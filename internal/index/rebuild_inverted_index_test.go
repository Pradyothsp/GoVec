package index

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/storage"
)

// TestRebuildInvertedIndex_EveryFormat checks that each load path rebuilds the
// posting lists. Snapshots don't store the inverted index, and hybrid search
// reads only it for sparse scores, so a load path that skipped the rebuild
// would silently drop every keyword match after a restart.
func TestRebuildInvertedIndex_EveryFormat(t *testing.T) {
	type hybridIndex interface {
		FileEngine
		invertedIndex() *core.InvertedIndex
	}

	formats := []struct {
		name string
		// open builds an index with hybrid search on. mmapDir is shared between
		// the saving and the loading index, as it is across a restart.
		open func(t *testing.T, mmapDir string) hybridIndex
	}{
		{"brute", func(t *testing.T, _ string) hybridIndex {
			return bruteHybrid{newTestIndexWithHybrid(t)}
		}},
		{"hnsw", func(t *testing.T, _ string) hybridIndex {
			return hnswHybrid{newTestHNSWIndexWithHybrid(t)}
		}},
		{"brute_mmap", func(t *testing.T, mmapDir string) hybridIndex {
			return bruteHybrid{NewVectorIndex(Options[[]float32]{WAL: newTestWAL(t), Metric: testCosineMetric, HybridSearch: true, VectorStore: newTestMmapStore(t, mmapDir)})}
		}},
		{"hnsw_mmap", func(t *testing.T, mmapDir string) hybridIndex {
			return hnswHybrid{NewHNSWIndex(Options[[]float32]{WAL: newTestWAL(t), Metric: testCosineMetric, HybridSearch: true, MetadataIndex: true, VectorStore: newTestMmapStore(t, mmapDir)}, hnsw.Config{M: 16, EfSearch: 20, EfConstruction: 200})}
		}},
	}

	for _, f := range formats {
		t.Run(f.name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			mmapDir := filepath.Join(dir, "mmap")

			original := f.open(t, mmapDir)
			// 3 distinct tokens, 4 postings: token 10 appears in two documents.
			require.NoError(t, original.Insert(ctx, "doc1", []float32{1, 0, 0}, core.SparseVector{
				Indices: []uint32{10, 20}, Values: []float32{1, 2},
			}, nil))
			require.NoError(t, original.Insert(ctx, "doc2", []float32{0, 1, 0}, core.SparseVector{
				Indices: []uint32{10, 30}, Values: []float32{3, 4},
			}, nil))
			require.NoError(t, original.Insert(ctx, "doc3", []float32{0, 0, 1}, core.SparseVector{}, nil))

			require.Equal(t, 3, original.invertedIndex().Tokens(), "sanity check before saving")

			path := filepath.Join(dir, "snapshot.bin")
			require.NoError(t, original.SaveToFile(ctx, path))

			loaded := f.open(t, mmapDir)
			require.NoError(t, loaded.LoadFromFile(ctx, path))

			assert.Equal(t, 3, loaded.invertedIndex().Tokens(), "every token must have its posting list back")
			assert.Equal(t, 4, loaded.invertedIndex().Postings(), "every posting must be rebuilt")
			assert.Len(t, loaded.invertedIndex().Scores(core.SparseVector{Indices: []uint32{10}, Values: []float32{1}}), 2, "token 10 is in doc1 and doc2")
		})
	}
}

type bruteHybrid struct{ *VectorIndex[[]float32] }

func (b bruteHybrid) invertedIndex() *core.InvertedIndex { return b.inverted }

type hnswHybrid struct{ *HNSWIndex[[]float32] }

func (h hnswHybrid) invertedIndex() *core.InvertedIndex { return h.inverted }

func newTestWAL(t *testing.T) *WAL {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })
	return wal
}

func newTestMmapStore(t *testing.T, dir string) *storage.MmapStore {
	t.Helper()
	store, err := storage.NewMmapStore(dir, 3, false)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return store
}
