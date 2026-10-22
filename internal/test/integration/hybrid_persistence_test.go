package integration

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/wal"
)

// TestHybridSearch_SurvivesRestart checks, for every snapshot format, that a
// hybrid query ranks the same after a restart from a snapshot and after one
// from the WAL alone: sparse scores come from the inverted index, which every
// load and replay path must rebuild from the stored sparse vectors. The
// posting counts themselves are asserted in internal/index
// (TestRebuildInvertedIndex_EveryFormat).
func TestHybridSearch_SurvivesRestart(t *testing.T) {
	ctx := context.Background()
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{1.0}}

	restarts := []struct {
		name string
		// sharedStore: the restarted engine opens the mmap store the first one
		// wrote, as a server does; a snapshot in mmap mode holds no vectors.
		sharedStore bool
		// restart starts the second engine with start, only once the first is
		// done writing, as a restarted server opens the store after the old
		// process stopped: a store reads its slot count when it opens.
		restart func(t *testing.T, before index.FileEngine, start func() index.FileEngine, dir string) index.FileEngine
	}{
		{"from_snapshot", true, func(t *testing.T, before index.FileEngine, start func() index.FileEngine, dir string) index.FileEngine {
			snapPath := filepath.Join(dir, "snapshot.bin")
			require.NoError(t, before.SaveToFile(ctx, snapPath))
			after := start()
			require.NoError(t, after.LoadFromFile(ctx, snapPath))
			return after
		}},
		{"from_wal", false, func(t *testing.T, _ index.FileEngine, start func() index.FileEngine, dir string) index.FileEngine {
			after := start()
			require.NoError(t, after.ReplayWAL(filepath.Join(dir, "before.wal")))
			return after
		}},
	}

	for _, f := range snapshotFormats {
		for _, r := range restarts {
			t.Run(f.name+"/"+r.name, func(t *testing.T) {
				// Arrange
				dir := t.TempDir()
				newEngine := func(walName, storeName string) index.FileEngine {
					t.Helper()
					walLog, err := wal.Open(filepath.Join(dir, walName))
					require.NoError(t, err)
					t.Cleanup(func() { _ = walLog.Close() })
					engine, err := index.NewEngine(
						config.EngineConfig{
							Quantization:       config.QuantizationNone,
							DistanceMetric:     config.DistanceMetricCosine,
							IndexType:          f.indexType,
							Dimensions:         3,
							EnableHybridSearch: true,
						},
						config.StorageConfig{EnableMmap: f.mmap, MmapStorePath: filepath.Join(dir, storeName)},
						walLog,
					)
					require.NoError(t, err)
					return engine
				}

				before := newEngine("before.wal", "mmap")
				// doc2 is the closer dense match; only a working sparse boost on
				// token 10 can put doc1 above it.
				require.NoError(t, before.Insert(ctx, "doc1", []float32{0.9, 0.1, 0}, core.SparseVector{Indices: []uint32{10}, Values: []float32{10.0}}, nil))
				require.NoError(t, before.Insert(ctx, "doc2", []float32{1, 0, 0}, core.SparseVector{}, nil))
				require.NoError(t, before.Insert(ctx, "doc3", []float32{0, 1, 0}, core.SparseVector{Indices: []uint32{99}, Values: []float32{9.0}}, nil))
				ranked, err := before.Search(ctx, []float32{1, 0, 0}, sparseQuery, 2, nil)
				require.NoError(t, err)
				require.Equal(t, "doc1", ranked[0].ID, "sanity check: the ranking must depend on the sparse boost")
				afterStore := "mmap-after"
				if r.sharedStore {
					afterStore = "mmap"
				}
				start := func() index.FileEngine { return newEngine("after.wal", afterStore) }

				// Act
				after := r.restart(t, before, start, dir)
				results, err := after.Search(ctx, []float32{1, 0, 0}, sparseQuery, 2, nil)

				// Assert
				require.NoError(t, err)
				require.Equal(t, 3, after.Len())
				require.Len(t, results, 2)
				require.Equal(t, "doc1", results[0].ID, "the sparse boost must outrank doc2's better dense score after a restart")
			})
		}
	}
}
