package index

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/test/fixtures"
	"github.com/Pradyothsp/govec/internal/wal"
)

// The log's own rules (torn tails, corruption, old formats) are tested in
// package wal. These cover what the engines do with the entries it replays.

// writeLog writes entries to a new log file and returns its path.
func writeLog(t *testing.T, entries ...wal.Entry) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.wal")
	walLog, err := wal.Open(path)
	require.NoError(t, err)
	for _, e := range entries {
		require.NoError(t, walLog.WriteEntry(context.Background(), &e))
	}
	require.NoError(t, walLog.Close())
	return path
}

var goodEntry = wal.Entry{Action: wal.ActionInsert, ID: "good", Vector: []float32{1, 0, 0}}

// An intact entry the engine refuses stops replay, wherever it sits: it was
// acknowledged once, so skipping it would lose a write.
func TestReplayWAL_EntryThatCannotBeApplied_ReturnsError(t *testing.T) {
	bad := map[string]wal.Entry{
		"empty_vector":       {Action: wal.ActionInsert, ID: "novec"},
		"dimension_mismatch": {Action: wal.ActionInsert, ID: "wrongdims", Vector: []float32{1, 0}},
	}
	for _, e := range bothEngines {
		for name, entry := range bad {
			t.Run(e.name+"/"+name, func(t *testing.T) {
				for _, order := range [][]wal.Entry{{entry, goodEntry}, {goodEntry, entry}} {
					// Arrange
					idx, _ := e.open(t)
					path := writeLog(t, order...)

					// Act
					err := idx.ReplayWAL(path)

					// Assert
					require.Error(t, err)
					assert.Contains(t, err.Error(), "record ")
				}
			})
		}
	}
}

// Writes refuse NaN now, but a log written before they did may hold one. It
// was acknowledged then, so it must still replay rather than stop the server.
func TestReplayWAL_NonFiniteEntryFromAnOlderServer_StillLoads(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			old := wal.Entry{Action: wal.ActionInsert, ID: "old", Vector: []float32{1, float32(math.NaN()), 0}}
			path := writeLog(t, goodEntry, old)

			// Act
			err := idx.ReplayWAL(path)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, 2, idx.Len())
		})
	}
}

func TestReplayWAL_DeleteOfMissingID_IsNotAnError(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			path := writeLog(t, wal.Entry{Action: wal.ActionDelete, ID: "never-inserted"}, goodEntry)

			// Act
			err := idx.ReplayWAL(path)

			// Assert
			require.NoError(t, err, "replay must stay idempotent")
			assert.Equal(t, 1, idx.Len())
		})
	}
}

// Insert, overwrite and delete replay to the state they left.
func TestReplayWAL_ComplexSequence(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			path := writeLog(t,
				wal.Entry{Action: wal.ActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple, Meta: fixtures.MetaSimple},
				wal.Entry{Action: wal.ActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
				wal.Entry{Action: wal.ActionInsert, ID: "v1", Vector: fixtures.Vec3dThird, Meta: fixtures.MetaNested},
				wal.Entry{Action: wal.ActionDelete, ID: "v2"},
			)

			// Act
			err := idx.ReplayWAL(path)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, 1, idx.Len())
			v1, err := idx.GetByID(context.Background(), "v1")
			require.NoError(t, err)
			assert.InDeltaSlice(t, fixtures.Vec3dThird, v1.Vector, 1e-6, "the overwrite must win")
			assert.Contains(t, v1.Metadata, "nested")
			_, err = idx.GetByID(context.Background(), "v2")
			assert.Error(t, err, "v2 was deleted")
		})
	}
}
