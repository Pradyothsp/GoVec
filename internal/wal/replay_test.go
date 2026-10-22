package wal

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// Recovery keeps the longest valid prefix of the log, as Redis and etcd do:
// a bad last record is a write torn by a crash and is cut off; anything else
// that can't be replayed is corruption and must stop recovery.

// encodeRecord encodes entry as the log writes it.
func encodeRecord(t *testing.T, entry *Entry) []byte {
	t.Helper()
	record, err := appendRecord(nil, entry)
	require.NoError(t, err)
	return record
}

// framedRecord frames an arbitrary payload with a valid header and CRC, for
// payloads the writer itself would refuse to produce.
func framedRecord(payload []byte) []byte {
	record := []byte{recordVersion}
	record = binary.LittleEndian.AppendUint32(record, uint32(len(payload))) //nolint:gosec // small test payloads
	record = binary.LittleEndian.AppendUint32(record, crc32.Checksum(payload, crcTable))
	return append(record, payload...)
}

// withFlippedPayloadByte returns record with its last byte changed, so its
// CRC no longer matches.
func withFlippedPayloadByte(record []byte) []byte {
	damaged := bytes.Clone(record)
	damaged[len(damaged)-1] ^= 0xFF
	return damaged
}

func goodRecord(t *testing.T) []byte {
	return encodeRecord(t, &Entry{Action: ActionInsert, ID: "before", Vector: []float32{1, 0, 0}})
}

func writeFile(t *testing.T, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.wal")
	require.NoError(t, os.WriteFile(path, content, 0o600))
	return path
}

// replayIDs replays path and returns the IDs of the entries it applied.
func replayIDs(t *testing.T, path string) ([]string, error) {
	t.Helper()
	var ids []string
	err := Replay(path, func(e Entry) error {
		ids = append(ids, e.ID)
		return nil
	})
	return ids, err
}

// A missing log is a first start, and an empty one a clean shutdown: neither
// is an error, and neither replays anything.
func TestReplay_NothingToReplay(t *testing.T) {
	t.Parallel()

	tests := map[string]func(t *testing.T) string{
		"missing": func(t *testing.T) string { return filepath.Join(t.TempDir(), "none.wal") },
		"empty":   func(t *testing.T) string { return writeFile(t, nil) },
	}
	for name, path := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Arrange
			p := path(t)

			// Act
			ids, err := replayIDs(t, p)

			// Assert
			require.NoError(t, err)
			assert.Empty(t, ids)
		})
	}
}

func TestReplay_TornLastRecord_IsCutOffSoTheNextWriteSurvives(t *testing.T) {
	t.Parallel()

	unacked := encodeRecord(t, &Entry{Action: ActionInsert, ID: "unacked", Vector: []float32{0, 1, 0}})
	tails := map[string][]byte{
		"incomplete_header":  unacked[:5],
		"incomplete_payload": unacked[:len(unacked)-3],
		"crc_mismatch":       withFlippedPayloadByte(unacked),
		"zero_filled_tail":   make([]byte, 32),
		"zero_filled_short":  make([]byte, 4),
		// Stale bytes some filesystems leave at the end of a file after a crash.
		"garbage": []byte("CORRUPTED DATA\n"),
	}
	for name, tail := range tails {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Arrange
			good := goodRecord(t)
			path := writeFile(t, append(bytes.Clone(good), tail...))

			// Act
			ids, err := replayIDs(t, path)

			// Assert
			require.NoError(t, err, "a torn last record is expected after a crash")
			assert.Equal(t, []string{"before"}, ids)
			data, err := os.ReadFile(path)
			require.NoError(t, err)
			assert.Equal(t, good, data, "the torn record must be cut off the file")

			// The server keeps appending to this file after recovery. The next
			// write must start a record of its own, not follow the torn one.
			l, err := Open(path)
			require.NoError(t, err)
			require.NoError(t, l.WriteEntry(context.Background(), &Entry{Action: ActionInsert, ID: "after", Vector: []float32{0, 0, 1}}))
			require.NoError(t, l.Close())
			ids, err = replayIDs(t, path)
			require.NoError(t, err)
			assert.Equal(t, []string{"before", "after"}, ids, "the write made after recovery must survive the next restart")
		})
	}
}

func TestReplay_CorruptRecordBeforeTheEnd_ReturnsError(t *testing.T) {
	t.Parallel()

	later := encodeRecord(t, &Entry{Action: ActionInsert, ID: "later", Vector: []float32{0, 1, 0}})
	damaged := map[string][]byte{
		"crc_mismatch": withFlippedPayloadByte(encodeRecord(t, &Entry{Action: ActionInsert, ID: "bad", Vector: []float32{1, 1, 0}})),
		"bad_header":   {7, 1, 2, 3, 4, 5, 6, 7, 8},
	}
	for name, bad := range damaged {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Arrange
			content := bytes.Join([][]byte{goodRecord(t), bad, later}, nil)
			path := writeFile(t, content)

			// Act
			_, err := replayIDs(t, path)

			// Assert
			require.Error(t, err)
			assert.Contains(t, err.Error(), "record 2")
			data, readErr := os.ReadFile(path)
			require.NoError(t, readErr)
			assert.Equal(t, content, data, "a corrupt log must be left as found, for the operator to inspect")
		})
	}
}

// An intact record that can't be applied is never a torn write, even as the
// last record: replay stops and names it.
func TestReplay_EntryThatCannotBeApplied_ReturnsError(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		record []byte
		apply  func(Entry) error
	}{
		"rejected by apply": {
			record: encodeRecord(t, &Entry{Action: ActionInsert, ID: "rejected", Vector: []float32{1, 0}}),
			apply: func(e Entry) error {
				if e.ID == "rejected" {
					return errors.New("rejected")
				}
				return nil
			},
		},
		// Intact framing around a payload no writer produces.
		"unknown action": {
			record: framedRecord([]byte{9, 1, 'x', 0, 0, 0, 0}),
			apply:  func(Entry) error { return nil },
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			for _, content := range [][]byte{
				append(bytes.Clone(tt.record), goodRecord(t)...),
				append(goodRecord(t), tt.record...),
			} {
				// Arrange
				path := writeFile(t, content)

				// Act
				err := Replay(path, tt.apply)

				// Assert
				require.Error(t, err)
				assert.Contains(t, err.Error(), "record ")
			}
		})
	}
}

// A log written by GoVec 0.1.0 is refused with a clear message rather than
// misread or cut off as torn.
func TestReplay_JSONLog_RejectedAndLeftAlone(t *testing.T) {
	t.Parallel()

	// Arrange
	content := []byte(`{"action":"INSERT","id":"old","vec":[1,0,0]}` + "\n")
	path := writeFile(t, content)

	// Act
	_, err := replayIDs(t, path)

	// Assert
	require.ErrorIs(t, err, errLegacyFormat)
	data, readErr := os.ReadFile(path)
	require.NoError(t, readErr)
	assert.Equal(t, content, data)
}

// A wal_path that names the wrong file: with no valid record anywhere, a
// torn-tail rule alone would truncate it to nothing.
func TestReplay_NotALogAtAll_RefusedAndLeftAlone(t *testing.T) {
	t.Parallel()

	// Arrange
	content := []byte("these are somebody else's bytes, not a WAL\n")
	path := writeFile(t, content)

	// Act
	_, err := replayIDs(t, path)

	// Assert
	require.Error(t, err)
	assert.Contains(t, err.Error(), "doesn't look like a GoVec WAL")
	data, readErr := os.ReadFile(path)
	require.NoError(t, readErr)
	assert.Equal(t, content, data)
}

// A crash during the very first write leaves a record prefix at byte 0, which
// is torn, not foreign.
func TestReplay_TornFirstWrite_IsCutOff(t *testing.T) {
	t.Parallel()

	// Arrange
	record := encodeRecord(t, &Entry{Action: ActionInsert, ID: "first", Vector: []float32{1, 0, 0}})
	path := writeFile(t, record[:len(record)-4])

	// Act
	ids, err := replayIDs(t, path)

	// Assert
	require.NoError(t, err)
	assert.Empty(t, ids)
	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Zero(t, info.Size())
}

func TestRecord_RoundTripsEveryField(t *testing.T) {
	t.Parallel()

	// Arrange
	entries := []*Entry{
		{
			Action: ActionInsert,
			ID:     "full ✓",
			Vector: []float32{0.1, -2.5, 3.25e-7},
			Sparse: core.SparseVector{Indices: []uint32{3, 70000}, Values: []float32{0.5, 1.5}},
			Meta:   map[string]any{"tag": "a", "n": float64(3), "nested": map[string]any{"ok": true}, "list": []any{"x", float64(1)}},
		},
		{Action: ActionInsert, ID: "bare", Vector: []float32{1, 0}},
		{Action: ActionDelete, ID: "gone"},
	}
	stream := make([]byte, 0, 256)
	for _, entry := range entries {
		stream = append(stream, encodeRecord(t, entry)...)
	}

	// Act
	var got []Entry
	err := Replay(writeFile(t, stream), func(e Entry) error {
		got = append(got, e)
		return nil
	})

	// Assert
	require.NoError(t, err)
	require.Len(t, got, len(entries))
	for i, want := range entries {
		assert.Equal(t, *want, got[i])
	}
}

func TestAppendRecord_UnknownAction_Rejected(t *testing.T) {
	t.Parallel()

	// Act
	_, err := appendRecord(nil, &Entry{Action: "UPSERT", ID: "x", Vector: []float32{1}})

	// Assert
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown WAL action")
}
