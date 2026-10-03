package index

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Recovery keeps the longest valid prefix of the WAL, as Redis and etcd do:
// a bad last line is a write torn by a crash and is cut off; anything else
// that can't be replayed is corruption and must stop recovery.

var replayEngines = []struct {
	name      string
	newEngine func(t *testing.T) Engine
}{
	{"brute", func(t *testing.T) Engine { return newTestIndex(t) }},
	{"hnsw", func(t *testing.T) Engine { return newTestHNSWIndex(t) }},
}

const goodWALLine = `{"action":"INSERT","id":"before","vec":[1,0,0]}` + "\n"

func writeWAL(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.wal")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	return path
}

func TestReplayWAL_TornLastLine_IsCutOffSoTheNextWriteSurvives(t *testing.T) {
	tails := map[string]string{
		"unterminated":     `{"action":"INSERT","id":"tor`,
		"terminated_junk":  "{this is: not json\n",
		"unterminated_ok":  `{"action":"INSERT","id":"unacked","vec":[0,1,0]}`,
		"zero_filled_tail": "\x00\x00\x00\x00",
	}
	for _, e := range replayEngines {
		for tailName, tail := range tails {
			t.Run(e.name+"/"+tailName, func(t *testing.T) {
				walPath := writeWAL(t, goodWALLine+tail)

				require.NoError(t, e.newEngine(t).ReplayWAL(walPath), "a torn last line is expected after a crash")

				data, err := os.ReadFile(walPath)
				require.NoError(t, err)
				assert.Equal(t, goodWALLine, string(data), "the torn line must be cut off the file")

				// The server keeps appending to this file after recovery. The
				// next write must start on its own line, not merge into the torn one.
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

func TestReplayWAL_CorruptLineBeforeTheEnd_ReturnsError(t *testing.T) {
	content := goodWALLine + "{this is: not json\n" + `{"action":"INSERT","id":"later","vec":[0,1,0]}` + "\n"
	for _, e := range replayEngines {
		t.Run(e.name, func(t *testing.T) {
			walPath := writeWAL(t, content)

			err := e.newEngine(t).ReplayWAL(walPath)

			require.Error(t, err)
			assert.Contains(t, err.Error(), "line 2")
			data, readErr := os.ReadFile(walPath)
			require.NoError(t, readErr)
			assert.Equal(t, content, string(data), "a corrupt WAL must be left as found, for the operator to inspect")
		})
	}
}

func TestReplayWAL_EntryThatCannotBeApplied_ReturnsError(t *testing.T) {
	entries := map[string]string{
		"empty_vector":       `{"action":"INSERT","id":"novec"}` + "\n",
		"dimension_mismatch": `{"action":"INSERT","id":"wrongdims","vec":[1,0]}` + "\n",
		"unknown_action":     `{"action":"UPSERT","id":"x","vec":[1,0,0]}` + "\n",
	}
	for _, e := range replayEngines {
		for entryName, entry := range entries {
			t.Run(e.name+"/"+entryName, func(t *testing.T) {
				// Put the bad entry before a good one, and as the last line:
				// a valid line that can't be applied is never a torn write.
				for _, content := range []string{entry + goodWALLine, goodWALLine + entry} {
					err := e.newEngine(t).ReplayWAL(writeWAL(t, content))
					require.Error(t, err, "WAL:\n%s", content)
					assert.Contains(t, err.Error(), "line ")
				}
			})
		}
	}
}

func TestReplayWAL_DeleteOfMissingID_IsNotAnError(t *testing.T) {
	for _, e := range replayEngines {
		t.Run(e.name, func(t *testing.T) {
			walPath := writeWAL(t, `{"action":"DELETE","id":"never-inserted"}`+"\n"+goodWALLine)
			require.NoError(t, e.newEngine(t).ReplayWAL(walPath), "replay must stay idempotent")
		})
	}
}
