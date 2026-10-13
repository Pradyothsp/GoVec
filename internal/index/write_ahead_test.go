package index

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// A write is committed only once it is in the WAL: recovery rebuilds the index
// from the snapshot and the WAL, so anything memory holds that the WAL doesn't
// would vanish at the next restart.

// TestWALWriteFails_ChangesNothing: a write the WAL can't record must not
// reach memory, for a single insert, a delete and a whole batch alike.
func TestWALWriteFails_ChangesNothing(t *testing.T) {
	ctx := context.Background()
	writes := map[string]func(FileEngine) error{
		"insert": func(e FileEngine) error {
			return e.Insert(ctx, "new", []float32{0, 1, 0}, core.SparseVector{}, nil)
		},
		"delete": func(e FileEngine) error {
			_, err := e.Delete(ctx, "kept")
			return err
		},
		"batch insert": func(e FileEngine) error {
			failures, err := e.BatchInsert(ctx, []BatchInsertItem{{ID: "new", Vector: []float32{0, 1, 0}}, {ID: "new2", Vector: []float32{0, 0, 1}}})
			assert.Nil(t, failures, "a batch that never reached the WAL has no per-item results")
			return err
		},
	}
	for _, e := range bothEngines {
		for name, write := range writes {
			t.Run(e.name+"/"+name, func(t *testing.T) {
				// Arrange
				idx, r := e.open(t)
				require.NoError(t, idx.Insert(ctx, "kept", []float32{1, 0, 0}, core.SparseVector{}, nil))
				require.NoError(t, r.wal.Close())

				// Act
				err := write(idx)

				// Assert
				require.Error(t, err)
				assert.Equal(t, 1, idx.Len())
				_, err = idx.GetByID(ctx, "kept")
				assert.NoError(t, err)
				_, err = idx.GetByID(ctx, "new")
				assert.Error(t, err)
			})
		}
	}
}

// Deleting an ID that isn't there changes nothing, so it logs nothing.
func TestDelete_UnknownID_WritesNothing(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)

			// Act
			deleted, err := idx.Delete(context.Background(), "never-inserted")

			// Assert
			require.NoError(t, err)
			assert.False(t, deleted)
			info, statErr := os.Stat(r.wal.Path())
			require.NoError(t, statErr)
			assert.Zero(t, info.Size())
		})
	}
}

// A save truncates the WAL only after the new snapshot is in place. If the
// truncate then fails, the WAL still holds entries the snapshot has too, which
// is safe because replay is idempotent: nothing may be lost either way.
func TestSaveToFile_TruncatesTheWALLast(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		for _, truncateFails := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/truncate_fails=%v", e.name, truncateFails), func(t *testing.T) {
				// Arrange
				idx, r := e.open(t)
				for i := range 5 {
					require.NoError(t, idx.Insert(ctx, fmt.Sprintf("v%d", i), []float32{1, float32(i), 0}, core.SparseVector{}, nil))
				}
				walPath, snapPath := r.wal.Path(), filepath.Join(t.TempDir(), "snapshot.bin")
				if truncateFails {
					require.NoError(t, r.wal.Close())
				}

				// Act
				saveErr := idx.SaveToFile(ctx, snapPath)

				// Assert
				info, err := os.Stat(walPath)
				require.NoError(t, err)
				if truncateFails {
					require.Error(t, saveErr)
					assert.Positive(t, info.Size(), "a failed truncate must leave the WAL as it was")
				} else {
					require.NoError(t, saveErr)
					assert.Zero(t, info.Size(), "everything in the WAL is now in the snapshot")
				}
				restarted, _ := e.open(t)
				require.NoError(t, restarted.LoadFromFile(ctx, snapPath))
				require.NoError(t, restarted.ReplayWAL(walPath))
				assert.Equal(t, 5, restarted.Len())
			})
		}
	}
}

// Replay takes the same lock as a write, so inserts arriving while a WAL is
// replayed neither race it nor get lost.
func TestReplayWAL_ConcurrentWithInserts(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange: a WAL of 10 inserts, written by another index.
			source, r := e.open(t)
			for i := range 10 {
				require.NoError(t, source.Insert(ctx, fmt.Sprintf("replayed%d", i), []float32{1, float32(i), 0}, core.SparseVector{}, nil))
			}
			idx, _ := e.open(t)

			// Act
			var wg sync.WaitGroup
			var replayErr error
			insertErrs := make(chan error, 10)
			wg.Add(2)
			go func() {
				defer wg.Done()
				replayErr = idx.ReplayWAL(r.wal.Path())
			}()
			go func() {
				defer wg.Done()
				for i := range 10 {
					insertErrs <- idx.Insert(ctx, fmt.Sprintf("inserted%d", i), []float32{0, 1, float32(i)}, core.SparseVector{}, nil)
				}
			}()
			wg.Wait()
			close(insertErrs)

			// Assert
			require.NoError(t, replayErr)
			for err := range insertErrs {
				require.NoError(t, err)
			}
			assert.Equal(t, 20, idx.Len(), "the replayed and inserted IDs are distinct, so all 20 must be present")
		})
	}
}
