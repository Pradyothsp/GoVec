package index

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// Recovery keeps the longest valid prefix of the WAL, as Redis and etcd do:
// a bad last record is a write torn by a crash and is cut off; anything else
// that can't be replayed is corruption and must stop recovery.

var replayEngines = []struct {
	name      string
	newEngine func(t *testing.T) Engine
}{
	{"brute", func(t *testing.T) Engine { return newTestIndex(t) }},
	{"hnsw", func(t *testing.T) Engine { return newTestHNSWIndex(t) }},
}

// walRecord encodes entry as the WAL writes it.
func walRecord(t *testing.T, entry *WALEntry) []byte {
	t.Helper()
	record, err := appendWALRecord(nil, entry)
	require.NoError(t, err)
	return record
}

// framedRecord frames an arbitrary payload with a valid header and CRC, for
// payloads the writer itself would refuse to produce.
func framedRecord(payload []byte) []byte {
	record := []byte{walRecordVersion}
	record = binary.LittleEndian.AppendUint32(record, uint32(len(payload))) //nolint:gosec // small test payloads
	record = binary.LittleEndian.AppendUint32(record, crc32.Checksum(payload, walCRCTable))
	return append(record, payload...)
}

// withFlippedPayloadByte returns record with its last byte changed, so its
// CRC no longer matches.
func withFlippedPayloadByte(record []byte) []byte {
	damaged := bytes.Clone(record)
	damaged[len(damaged)-1] ^= 0xFF
	return damaged
}

func goodWALRecord(t *testing.T) []byte {
	return walRecord(t, &WALEntry{Action: WALActionInsert, ID: "before", Vector: []float32{1, 0, 0}})
}

func writeWAL(t *testing.T, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.wal")
	require.NoError(t, os.WriteFile(path, content, 0o600))
	return path
}

func TestReplayWAL_TornLastRecord_IsCutOffSoTheNextWriteSurvives(t *testing.T) {
	unacked := walRecord(t, &WALEntry{Action: WALActionInsert, ID: "unacked", Vector: []float32{0, 1, 0}})
	tails := map[string][]byte{
		"incomplete_header":  unacked[:5],
		"incomplete_payload": unacked[:len(unacked)-3],
		"crc_mismatch":       withFlippedPayloadByte(unacked),
		"zero_filled_tail":   make([]byte, 32),
		"zero_filled_short":  make([]byte, 4),
		// Stale bytes some filesystems leave at the end of a file after a crash.
		"garbage": []byte("CORRUPTED DATA\n"),
	}
	for _, e := range replayEngines {
		for tailName, tail := range tails {
			t.Run(e.name+"/"+tailName, func(t *testing.T) {
				good := goodWALRecord(t)
				walPath := writeWAL(t, append(bytes.Clone(good), tail...))

				require.NoError(t, e.newEngine(t).ReplayWAL(walPath), "a torn last record is expected after a crash")

				data, err := os.ReadFile(walPath)
				require.NoError(t, err)
				assert.Equal(t, good, data, "the torn record must be cut off the file")

				// The server keeps appending to this file after recovery. The
				// next write must start a record of its own, not follow the torn one.
				wal, err := NewWAL(walPath)
				require.NoError(t, err)
				require.NoError(t, wal.WriteEntry(context.Background(), &WALEntry{Action: WALActionInsert, ID: "after", Vector: []float32{0, 0, 1}}))
				require.NoError(t, wal.Close())

				recovered := e.newEngine(t)
				require.NoError(t, recovered.ReplayWAL(walPath))
				assert.Equal(t, 2, recovered.Len())
				_, err = recovered.GetByID(context.Background(), "after")
				assert.NoError(t, err, "the write made after recovery must survive the next restart")
			})
		}
	}
}

func TestReplayWAL_CorruptRecordBeforeTheEnd_ReturnsError(t *testing.T) {
	later := walRecord(t, &WALEntry{Action: WALActionInsert, ID: "later", Vector: []float32{0, 1, 0}})
	damaged := map[string][]byte{
		"crc_mismatch": withFlippedPayloadByte(walRecord(t, &WALEntry{Action: WALActionInsert, ID: "bad", Vector: []float32{1, 1, 0}})),
		"bad_header":   {7, 1, 2, 3, 4, 5, 6, 7, 8},
	}
	for _, e := range replayEngines {
		for name, bad := range damaged {
			t.Run(e.name+"/"+name, func(t *testing.T) {
				content := bytes.Join([][]byte{goodWALRecord(t), bad, later}, nil)
				walPath := writeWAL(t, content)

				err := e.newEngine(t).ReplayWAL(walPath)

				require.Error(t, err)
				assert.Contains(t, err.Error(), "record 2")
				data, readErr := os.ReadFile(walPath)
				require.NoError(t, readErr)
				assert.Equal(t, content, data, "a corrupt WAL must be left as found, for the operator to inspect")
			})
		}
	}
}

func TestReplayWAL_EntryThatCannotBeApplied_ReturnsError(t *testing.T) {
	entries := map[string][]byte{
		"empty_vector":       walRecord(t, &WALEntry{Action: WALActionInsert, ID: "novec"}),
		"dimension_mismatch": walRecord(t, &WALEntry{Action: WALActionInsert, ID: "wrongdims", Vector: []float32{1, 0}}),
		// Intact framing around a payload no writer produces.
		"unknown_action": framedRecord([]byte{9, 1, 'x', 0, 0, 0, 0}),
	}
	for _, e := range replayEngines {
		for entryName, entry := range entries {
			t.Run(e.name+"/"+entryName, func(t *testing.T) {
				// Put the bad entry before a good one, and as the last record: an
				// intact record that can't be applied is never a torn write.
				for _, content := range [][]byte{append(bytes.Clone(entry), goodWALRecord(t)...), append(goodWALRecord(t), entry...)} {
					err := e.newEngine(t).ReplayWAL(writeWAL(t, content))
					require.Error(t, err)
					assert.Contains(t, err.Error(), "record ")
				}
			})
		}
	}
}

func TestReplayWAL_DeleteOfMissingID_IsNotAnError(t *testing.T) {
	for _, e := range replayEngines {
		t.Run(e.name, func(t *testing.T) {
			content := append(walRecord(t, &WALEntry{Action: WALActionDelete, ID: "never-inserted"}), goodWALRecord(t)...)
			require.NoError(t, e.newEngine(t).ReplayWAL(writeWAL(t, content)), "replay must stay idempotent")
		})
	}
}

// TestReplayWAL_JSONWAL_RejectedAndLeftAlone covers a WAL written by GoVec
// 0.1.0: refuse it with a clear message rather than misreading it or cutting
// it off as torn.
func TestReplayWAL_JSONWAL_RejectedAndLeftAlone(t *testing.T) {
	content := []byte(`{"action":"INSERT","id":"old","vec":[1,0,0]}` + "\n")
	walPath := writeWAL(t, content)

	err := newTestIndex(t).ReplayWAL(walPath)

	require.ErrorIs(t, err, errLegacyWAL)
	data, readErr := os.ReadFile(walPath)
	require.NoError(t, readErr)
	assert.Equal(t, content, data)
}

func TestWALRecord_RoundTripsEveryField(t *testing.T) {
	entries := []*WALEntry{
		{
			Action: WALActionInsert,
			ID:     "full ✓",
			Vector: []float32{0.1, -2.5, 3.25e-7},
			Sparse: core.SparseVector{Indices: []uint32{3, 70000}, Values: []float32{0.5, 1.5}},
			Meta:   map[string]any{"tag": "a", "n": float64(3), "nested": map[string]any{"ok": true}, "list": []any{"x", float64(1)}},
		},
		{Action: WALActionInsert, ID: "bare", Vector: []float32{1, 0}},
		{Action: WALActionDelete, ID: "gone"},
	}

	stream := make([]byte, 0, 256)
	for _, entry := range entries {
		stream = append(stream, walRecord(t, entry)...)
	}
	var got []WALEntry
	require.NoError(t, replayWALFile(writeWAL(t, stream), func(e WALEntry) error {
		got = append(got, e)
		return nil
	}))

	require.Len(t, got, len(entries))
	for i, want := range entries {
		assert.Equal(t, *want, got[i])
	}
}

func TestAppendWALRecord_UnknownAction_Rejected(t *testing.T) {
	_, err := appendWALRecord(nil, &WALEntry{Action: "UPSERT", ID: "x", Vector: []float32{1}})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown WAL action")
}

// TestReplayWAL_NotAWALAtAll_RefusedAndLeftAlone guards a wal_path that names
// the wrong file: with no valid record anywhere, a torn-tail rule alone would
// truncate it to nothing.
func TestReplayWAL_NotAWALAtAll_RefusedAndLeftAlone(t *testing.T) {
	content := []byte("these are somebody else's bytes, not a WAL\n")
	walPath := writeWAL(t, content)

	err := newTestIndex(t).ReplayWAL(walPath)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "doesn't look like a GoVec WAL")
	data, readErr := os.ReadFile(walPath)
	require.NoError(t, readErr)
	assert.Equal(t, content, data)
}

// TestReplayWAL_TornFirstWrite_IsCutOff: a crash during the very first write
// leaves a record prefix at byte 0, which is torn, not foreign.
func TestReplayWAL_TornFirstWrite_IsCutOff(t *testing.T) {
	record := walRecord(t, &WALEntry{Action: WALActionInsert, ID: "first", Vector: []float32{1, 0, 0}})
	walPath := writeWAL(t, record[:len(record)-4])

	require.NoError(t, newTestIndex(t).ReplayWAL(walPath))

	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Zero(t, info.Size())
}
