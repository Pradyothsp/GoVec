package integration

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// TestHybridSearch_SurvivesSnapshotReload checks, for every snapshot format,
// that a hybrid query ranks the same after a save and reload: sparse scores
// come from each node's stored sparse vector, so the vectors must round-trip.
//
// It does not check the inverted index. Search doesn't read the posting lists
// yet, so this test passes with or without the rebuild on load; the posting
// counts are asserted in internal/index (TestLoadFromFile_RebuildsInvertedIndex
// and TestRebuildInvertedIndex_EveryFormat).
func TestHybridSearch_SurvivesSnapshotReload(t *testing.T) {
	formats := []struct {
		name      string
		indexType config.IndexType
		mmap      bool
	}{
		{"brute", config.IndexTypeBrute, false},
		{"hnsw", config.IndexTypeHNSW, false},
		{"brute_mmap", config.IndexTypeBrute, true},
		{"hnsw_mmap", config.IndexTypeHNSW, true},
	}

	for _, f := range formats {
		t.Run(f.name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			snapPath := filepath.Join(dir, "snapshot.bin")

			engineCfg := config.EngineConfig{
				Quantization:       config.QuantizationNone,
				DistanceMetric:     "cosine",
				IndexType:          f.indexType,
				Dimensions:         3,
				EnableHybridSearch: true,
			}
			storageCfg := config.StorageConfig{
				EnableMmap:    f.mmap,
				MmapStorePath: filepath.Join(dir, "mmap"),
			}

			newEngine := func(walName string) index.Engine {
				t.Helper()
				wal, err := index.NewWAL(filepath.Join(dir, walName))
				require.NoError(t, err)
				t.Cleanup(func() { _ = wal.Close() })
				engine, err := index.NewEngine(engineCfg, storageCfg, wal)
				require.NoError(t, err)
				return engine
			}

			original := newEngine("original.wal")
			// doc2 is the closer dense match; only a working sparse boost on
			// token 10 can put doc1 above it.
			require.NoError(t, original.Insert(ctx, "doc1", []float32{0.9, 0.1, 0}, core.SparseVector{
				Indices: []uint32{10}, Values: []float32{10.0},
			}, nil))
			require.NoError(t, original.Insert(ctx, "doc2", []float32{1, 0, 0}, core.SparseVector{}, nil))
			require.NoError(t, original.Insert(ctx, "doc3", []float32{0, 1, 0}, core.SparseVector{
				Indices: []uint32{99}, Values: []float32{9.0},
			}, nil))

			sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{1.0}}
			assertSparseBoostWins := func(t *testing.T, engine index.Engine) {
				t.Helper()
				results, err := engine.Search(ctx, []float32{1, 0, 0}, sparseQuery, 2, nil)
				require.NoError(t, err)
				require.Len(t, results, 2)
				require.Equal(t, "doc1", results[0].ID, "the sparse boost must outrank doc2's better dense score")
			}

			// Sanity check: the ranking really does depend on the sparse boost.
			assertSparseBoostWins(t, original)

			require.NoError(t, original.SaveToFile(ctx, snapPath))

			reloaded := newEngine("reloaded.wal")
			require.NoError(t, reloaded.LoadFromFile(ctx, snapPath))
			require.Equal(t, 3, reloaded.Len())
			assertSparseBoostWins(t, reloaded)
		})
	}
}
