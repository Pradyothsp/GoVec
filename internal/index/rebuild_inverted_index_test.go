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
// posting lists. Snapshots don't store the inverted index, and search doesn't
// read it yet (the sparse score scans every node's sparse vector), so a load
// path that skipped the rebuild would pass every ranking test. Only counting
// the postings catches it -- this is the guard for the planned work that makes
// search use them.
func TestRebuildInvertedIndex_EveryFormat(t *testing.T) {
	type hybridIndex interface {
		Engine
		postings() map[uint32][]core.Posting
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
			return bruteHybrid{NewVectorIndex[[]float32](
				newTestWAL(t), make(map[uint32][]core.Posting), core.NewIDMapper(),
				func(v []float32) []float32 { return v }, core.CosineSimilarity,
				newTestMmapStore(t, mmapDir))}
		}},
		{"hnsw_mmap", func(t *testing.T, mmapDir string) hybridIndex {
			return hnswHybrid{NewHNSWIndex[[]float32](
				newTestWAL(t), make(map[uint32][]core.Posting), core.NewIDMapper(),
				func(v []float32) []float32 { return v }, core.CosineSimilarity,
				hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200,
				core.NewMetadataIndex(), newTestMmapStore(t, mmapDir))}
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

			require.Len(t, original.postings(), 3, "sanity check before saving")

			path := filepath.Join(dir, "snapshot.bin")
			require.NoError(t, original.SaveToFile(ctx, path))

			loaded := f.open(t, mmapDir)
			require.NoError(t, loaded.LoadFromFile(ctx, path))

			assert.Len(t, loaded.postings(), 3, "every token must have its posting list back")
			assert.Equal(t, 4, countPostings(loaded.postings()), "every posting must be rebuilt")
			assert.Len(t, loaded.postings()[10], 2, "token 10 is in doc1 and doc2")
		})
	}
}

type bruteHybrid struct{ *VectorIndex[[]float32] }

func (b bruteHybrid) postings() map[uint32][]core.Posting { return b.InvertedIndex }

type hnswHybrid struct{ *HNSWIndex[[]float32] }

func (h hnswHybrid) postings() map[uint32][]core.Posting { return h.InvertedIndex }

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
