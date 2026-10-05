package index

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

// =============================================================================
// HNSW Test Helpers
// =============================================================================

func newTestHNSWIndex(t testing.TB) *HNSWIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "hnsw_test.wal"))
	if err != nil {
		t.Fatalf("newTestHNSWIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	return NewHNSWIndex[[]float32](wal, nil, idMapper, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, core.NewMetadataIndex(), nil)
}

func newTestHNSWIndexWithHybrid(t testing.TB) *HNSWIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "hnsw_hybrid_test.wal"))
	if err != nil {
		t.Fatalf("newTestHNSWIndexWithHybrid: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	invertedIndex := make(map[uint32][]core.Posting)
	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	return NewHNSWIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, core.NewMetadataIndex(), nil)
}

// =============================================================================
// Basic Insert / Search / Delete
// =============================================================================

func TestHNSWIndex_InsertAndLen(t *testing.T) {
	idx := newTestHNSWIndex(t)
	assert.Equal(t, 0, idx.Len())

	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	assert.Equal(t, 1, idx.Len())

	require.NoError(t, idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	assert.Equal(t, 2, idx.Len())
}

func TestHNSWIndex_Insert_Overwrite(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"version": 1}))
	assert.Equal(t, 1, idx.Len())

	// Overwrite same ID — count must not increase
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"version": 2}))
	assert.Equal(t, 1, idx.Len())

	// Metadata should reflect the latest insert
	internalID, err := idx.IDMapper.ToUint32ID("v1")
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"version": 2}, idx.Store[internalID].Metadata)
}

func TestHNSWIndex_Search_EmptyIndex(t *testing.T) {
	idx := newTestHNSWIndex(t)

	results, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 5, nil)
	require.NoError(t, err)
	assert.Nil(t, results)
}

func TestHNSWIndex_Search_EmptyQuery(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	_, err := idx.Search(context.Background(), []float32{}, core.SparseVector{}, 5, nil)
	assert.Error(t, err)
}

func TestHNSWIndex_Delete_ExistingVector(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	assert.Equal(t, 2, idx.Len())

	internalID, err := idx.IDMapper.ToUint32ID("v1")
	require.NoError(t, err)

	deleted, err := idx.Delete(context.Background(), "v1")
	require.NoError(t, err)
	assert.True(t, deleted)
	assert.Equal(t, 1, idx.Len())

	// ID is gone from metadata and is tombstoned
	assert.NotContains(t, idx.Store, internalID)
	assert.True(t, idx.IDMapper.IsTombstone(internalID))
}

func TestHNSWIndex_Delete_NonExistentVector(t *testing.T) {
	idx := newTestHNSWIndex(t)

	deleted, err := idx.Delete(context.Background(), "does-not-exist")
	require.NoError(t, err)
	assert.False(t, deleted)
}

// =============================================================================
// Search ordering: distance → similarity score
// =============================================================================

func TestHNSWIndex_SearchOrdering_ByCosineSimilarity(t *testing.T) {
	idx := newTestHNSWIndex(t)

	// Vectors with predictable angular distances from [1, 0, 0]:
	//   doc1: [1,0,0] — identical to query (cosine = 1.0, score = 1.0)
	//   doc2: [0.9,0.1,0] — very close (score ≈ 0.99)
	//   doc3: [0,1,0] — orthogonal (cosine = 0.0, score = 0.0)
	require.NoError(t, idx.Insert(context.Background(), "doc3", []float32{0, 1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0.9, 0.1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{}, nil))

	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 3, nil)
	require.NoError(t, err)
	require.Len(t, results, 3)

	// Scores must be descending
	assert.GreaterOrEqual(t, results[0].Score, results[1].Score)
	assert.GreaterOrEqual(t, results[1].Score, results[2].Score)

	// Top result must be the closest vector
	assert.Equal(t, "doc1", results[0].ID)

	// All scores must be in [−1, 1] (cosine similarity range)
	for _, r := range results {
		assert.GreaterOrEqual(t, r.Score, float32(-1.0))
		assert.LessOrEqual(t, r.Score, float32(1.0)+1e-5)
	}
}

func TestHNSWIndex_Search_RespectLimit(t *testing.T) {
	idx := newTestHNSWIndex(t)

	for i, vec := range [][]float32{
		{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.7, 0.7, 0}, {0.5, 0.5, 0.5},
	} {
		require.NoError(t, idx.Insert(context.Background(), fixtures.GenerateMetadata(1)[0]["id"].(string)+string(rune('A'+i)), vec, core.SparseVector{}, nil))
	}

	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 2, nil)
	require.NoError(t, err)
	assert.Len(t, results, 2, "result count must equal k")
}

// =============================================================================
// Metadata filtering
// =============================================================================

func TestHNSWIndex_Search_WithFilter(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{"category": "A"}))
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0.9, 0.1, 0}, core.SparseVector{}, map[string]any{"category": "B"}))
	require.NoError(t, idx.Insert(context.Background(), "doc3", []float32{0.8, 0.2, 0}, core.SparseVector{}, map[string]any{"category": "A"}))
	require.NoError(t, idx.Insert(context.Background(), "doc4", []float32{0, 1, 0}, core.SparseVector{}, map[string]any{"category": "B"}))
	require.NoError(t, idx.Insert(context.Background(), "doc5", []float32{0, 0, 1}, core.SparseVector{}, map[string]any{"category": "A"}))

	// Over-fetching kicks in (k*5 candidates fetched internally, then filtered)
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 10, map[string]interface{}{"category": "A"})
	require.NoError(t, err)

	for _, r := range results {
		assert.Equal(t, "A", r.Meta["category"], "filter must exclude category B docs")
	}
	assert.LessOrEqual(t, len(results), 3, "at most 3 category-A docs exist")
}

func TestHNSWIndex_Search_NoMetadataMatch(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{"category": "A"}))
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0, 1, 0}, core.SparseVector{}, map[string]any{"category": "A"}))

	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 5, map[string]interface{}{"category": "Z"})
	require.NoError(t, err)
	assert.Empty(t, results)
}

// =============================================================================
// Hybrid search (dense + sparse)
// =============================================================================

func TestHNSWIndex_HybridSearch_SparseBoostsRanking(t *testing.T) {
	idx := newTestHNSWIndexWithHybrid(t)

	// doc1: great dense match, weak sparse match
	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{0.1},
	}, nil))

	// doc2: slightly weaker dense match, very strong sparse match
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0.9, 0.1, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{10.0},
	}, nil))

	require.NoError(t, idx.Insert(context.Background(), "doc3", []float32{0, 1, 0}, core.SparseVector{}, nil))

	query := []float32{1, 0, 0}
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{5.0}}

	results, err := idx.Search(context.Background(), query, sparseQuery, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)

	// doc2 should rank first because strong sparse match overcomes slight dense gap
	assert.Equal(t, "doc2", results[0].ID, "sparse boost should elevate doc2 above doc1")
}

func TestHNSWIndex_HybridSearch_EmptySparseQueryFallsBackToDense(t *testing.T) {
	idx := newTestHNSWIndexWithHybrid(t)

	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{5.0},
	}, nil))
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0, 1, 0}, core.SparseVector{}, nil))

	// Empty sparse query → dense-only path
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)

	assert.Equal(t, "doc1", results[0].ID, "without sparse query, dense similarity governs ranking")
}

// =============================================================================
// Persistence round-trip
// =============================================================================

func TestHNSWIndex_PersistenceRoundTrip(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 0, 0}, core.SparseVector{}, fixtures.MetaSimple))
	require.NoError(t, idx.Insert(context.Background(), "v2", []float32{0, 1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v3", []float32{0, 0, 1}, core.SparseVector{}, fixtures.MetaNested))

	path := filepath.Join(t.TempDir(), "hnsw.bin")
	require.NoError(t, idx.SaveToFile(context.Background(), path))

	// Verify no temp file leaked
	_, err := os.Stat(path + ".tmp")
	assert.True(t, os.IsNotExist(err), "temp file must be cleaned up after save")

	// Load into a fresh index
	idx2 := newTestHNSWIndex(t)
	require.NoError(t, idx2.LoadFromFile(context.Background(), path))

	assert.Equal(t, 3, idx2.Len())
	assert.Equal(t, idx.IDMapper.Count(), idx2.IDMapper.Count())
	assert.Equal(t, idx.IDMapper.NextID(), idx2.IDMapper.NextID())

	// Search still returns the right top result
	results, err := idx2.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "v1", results[0].ID)
}

func TestHNSWIndex_LoadFromFile_MissingFile(t *testing.T) {
	idx := newTestHNSWIndex(t)
	err := idx.LoadFromFile(context.Background(), "/nonexistent/hnsw.bin")
	assert.NoError(t, err, "missing file must be handled gracefully")
	assert.Equal(t, 0, idx.Len())
}

func TestHNSWIndex_LoadFromFile_CorruptedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	require.NoError(t, os.WriteFile(path, []byte("not valid gob data"), 0o600))

	idx := newTestHNSWIndex(t)
	err := idx.LoadFromFile(context.Background(), path)
	assert.Error(t, err)
}

func TestHNSWIndex_LoadFromFile_BruteForceSnapshot(t *testing.T) {
	// Save a brute-force (v3) snapshot
	bruteIdx := newTestIndex(t)
	require.NoError(t, bruteIdx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	path := filepath.Join(t.TempDir(), "brute.bin")
	require.NoError(t, bruteIdx.SaveToFile(context.Background(), path))

	// Attempting to load it as an HNSW snapshot must fail with a clear message
	hnswIdx := newTestHNSWIndex(t)
	err := hnswIdx.LoadFromFile(context.Background(), path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "brute", "error must hint that the snapshot is brute-force")
}

func TestHNSWIndex_SaveToFile_InvalidPath(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	err := idx.SaveToFile(context.Background(), filepath.Join(uncreatableDir(t), "hnsw.bin"))
	assert.Error(t, err)
}

// Verify that a VectorIndex refuses to load an HNSW snapshot
func TestVectorIndex_LoadFromFile_RejectsHNSWSnapshot(t *testing.T) {
	hnswIdx := newTestHNSWIndex(t)
	require.NoError(t, hnswIdx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	path := filepath.Join(t.TempDir(), "hnsw.bin")
	require.NoError(t, hnswIdx.SaveToFile(context.Background(), path))

	bruteIdx := newTestIndex(t)
	err := bruteIdx.LoadFromFile(context.Background(), path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "hnsw", "error must hint that the snapshot is HNSW")
}

// =============================================================================
// WAL replay recovery
// =============================================================================

func TestHNSWIndex_WALReplay_Recovery(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "wal_replay.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	idx := NewHNSWIndex[[]float32](wal, nil, idMapper, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, nil, nil)

	// Insert writes go to WAL first
	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v2", []float32{0, 1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v3", []float32{0, 0, 1}, core.SparseVector{}, nil))

	// Simulate crash: create a fresh index and replay the WAL
	wal2, err := NewWAL(filepath.Join(t.TempDir(), "dummy.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal2.Close() })

	idMapper2 := core.NewIDMapper()
	recovered := NewHNSWIndex[[]float32](wal2, nil, idMapper2, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, nil, nil)

	require.NoError(t, recovered.ReplayWAL(walPath))
	assert.Equal(t, 3, recovered.Len(), "all 3 inserts must be recovered from WAL")
}

func TestHNSWIndex_WALReplay_DeleteIsReplayed(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "wal_delete.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	idx := NewHNSWIndex[[]float32](wal, nil, idMapper, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, nil, nil)

	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v2", []float32{0, 1, 0}, core.SparseVector{}, nil))
	deleted, err := idx.Delete(context.Background(), "v1")
	require.NoError(t, err)
	require.True(t, deleted)

	// Replay on fresh index
	wal2, err := NewWAL(filepath.Join(t.TempDir(), "dummy.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal2.Close() })

	idMapper2 := core.NewIDMapper()
	recovered := NewHNSWIndex[[]float32](wal2, nil, idMapper2, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, nil, nil)

	require.NoError(t, recovered.ReplayWAL(walPath))
	assert.Equal(t, 1, recovered.Len(), "v1 was deleted in WAL, only v2 survives")
}

func TestHNSWIndex_WALReplay_MissingFile(t *testing.T) {
	idx := newTestHNSWIndex(t)
	err := idx.ReplayWAL("/nonexistent/wal.wal")
	assert.NoError(t, err, "missing WAL file must not be an error")
}

// =============================================================================
// Clear
// =============================================================================

func TestHNSWIndex_Clear(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	assert.Equal(t, 2, idx.Len())

	idx.Clear()
	assert.Equal(t, 0, idx.Len())
	assert.Empty(t, idx.Store)
}

// TestHNSWIndex_Clear_ResetsIDMapperAndWAL verifies Clear wipes ID mappings/tombstones
// and truncates the WAL, not just the graph/metadata -- these are easy to miss since
// they're separate fields the caller could forget to reset.
func TestHNSWIndex_Clear_ResetsIDMapperAndWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "hnsw_clear_test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	idx := NewHNSWIndex[[]float32](
		wal, nil, idMapper, identityFunc,
		hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, core.NewMetadataIndex(), nil,
	)

	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	_, err = idx.Delete(context.Background(), "v2")
	require.NoError(t, err)

	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Positive(t, info.Size(), "WAL should have entries before Clear")

	idx.Clear()

	assert.Equal(t, 0, idx.Len())
	assert.Equal(t, 0, idx.IDMapper.Count(), "IDMapper should have no active mappings after Clear")
	assert.Equal(t, uint32(0), idx.IDMapper.NextID(), "IDMapper's ID counter should restart at 0 after Clear")

	info, err = os.Stat(walPath)
	require.NoError(t, err)
	assert.Zero(t, info.Size(), "WAL should be truncated after Clear")
}

func TestHNSWIndex_Clear_AllowsReinsertion(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	idx.Clear()

	// Should be able to insert again after Clear
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	assert.Equal(t, 1, idx.Len())

	results, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "v1", results[0].ID)
}

// =============================================================================
// Filtered Search (MetadataIndex + Selectivity Strategy)
// =============================================================================

// TestHNSWIndex_FilteredSearch_NilFilter_Regression verifies that nil filters
// still return the nearest neighbors unchanged — backward compatibility.
func TestHNSWIndex_FilteredSearch_NilFilter_Regression(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "free"}))
	require.NoError(t, idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"tier": "premium"}))

	results, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 2, nil)
	require.NoError(t, err)
	assert.Len(t, results, 2, "nil filter must return all nearest neighbors")
}

// TestHNSWIndex_FilteredSearch_SelectiveFilter verifies that a highly selective filter
// still returns the correct k results using the graph-allowlist path.
func TestHNSWIndex_FilteredSearch_SelectiveFilter(t *testing.T) {
	idx := newTestHNSWIndex(t)

	vecs := fixtures.GenerateVectors(50, 3)
	// Insert 50 vectors: only 2 are "rare"
	for i, vec := range vecs {
		tier := "common"
		if i == 10 || i == 20 {
			tier = "rare"
		}
		require.NoError(t, idx.Insert(context.Background(),
			fmt.Sprintf("v%d", i), vec, core.SparseVector{}, map[string]any{"tier": tier},
		))
	}

	results, err := idx.Search(context.Background(), vecs[10], core.SparseVector{}, 2, map[string]any{"tier": "rare"})
	require.NoError(t, err)
	require.Len(t, results, 2)
	for _, r := range results {
		assert.Equal(t, "rare", r.Meta["tier"], "all results must have tier=rare")
	}
}

// TestHNSWIndex_FilteredSearch_NoMatch_ReturnsNil verifies the short-circuit path
// when the filter matches zero documents.
func TestHNSWIndex_FilteredSearch_NoMatch_ReturnsNil(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "free"}))

	results, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "nonexistent"})
	require.NoError(t, err)
	assert.Nil(t, results, "filter matching nothing must short-circuit and return nil")
}

// TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnOverwrite verifies that overwriting
// a vector updates the MetadataIndex so old filters no longer match.
func TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnOverwrite(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "free"}))

	// Overwrite with new metadata
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "premium"}))

	old, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "free"})
	require.NoError(t, err)
	assert.Nil(t, old, "old tier must no longer match after overwrite")

	updated, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "premium"})
	require.NoError(t, err)
	require.Len(t, updated, 1)
	assert.Equal(t, "v1", updated[0].ID)
}

// TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnDelete verifies that deleting
// a vector removes it from the MetadataIndex.
func TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnDelete(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "premium"}))
	require.NoError(t, idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"tier": "free"}))

	deleted, err := idx.Delete(context.Background(), "v1")
	require.NoError(t, err)
	require.True(t, deleted)

	results, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "premium"})
	require.NoError(t, err)
	assert.Nil(t, results, "deleted vector must not appear in filtered results")
}

// =============================================================================
// Persistence/WAL verification: InvertedIndex and metadata GOB round-trip
// =============================================================================

// TestHNSWIndex_SaveLoad_HybridRankingSurvives verifies that each
// document's sparse vector survives SaveToFile/LoadFromFile, so hybrid search
// ranks the same after a reload. Sparse scores come from the stored sparse
// vectors, not the InvertedIndex postings (nothing reads those yet); the
// postings rebuild is covered by rebuild_inverted_index_test.go.
func TestHNSWIndex_SaveLoad_HybridRankingSurvives(t *testing.T) {
	idx := newTestHNSWIndexWithHybrid(t)

	// doc1: close dense match with a strong sparse hit on term 10.
	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{10.0},
	}, nil))
	// doc2: the exact dense match for the query below, with no sparse hit.
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0.99, 0.1, 0}, core.SparseVector{}, nil))
	// doc3: orthogonal to query, strong sparse on an unrelated term.
	require.NoError(t, idx.Insert(context.Background(), "doc3", []float32{0, 1, 0}, core.SparseVector{
		Indices: []uint32{99}, Values: []float32{9.0},
	}, nil))

	path := filepath.Join(t.TempDir(), "hnsw_hybrid.bin")
	require.NoError(t, idx.SaveToFile(context.Background(), path))

	// Load into a fresh hybrid-enabled index.
	idx2 := newTestHNSWIndexWithHybrid(t)
	require.NoError(t, idx2.LoadFromFile(context.Background(), path))
	require.Equal(t, 3, idx2.Len())

	// The dense query is doc2's own vector, so doc2 wins on dense score alone.
	// doc1 can only rank first if its sparse vector survived the reload.
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{5.0}}
	results, err := idx2.Search(context.Background(), []float32{0.99, 0.1, 0}, sparseQuery, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, "doc1", results[0].ID, "doc1's sparse vector must survive the reload and outrank doc2's dense match")
}

// TestHNSWIndex_WALReplay_HybridRankingSurvives verifies that replaying a WAL
// restores each document's sparse vector from its WAL entry, so hybrid search
// ranks the same on an index recovered from the WAL alone.
func TestHNSWIndex_WALReplay_HybridRankingSurvives(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "hybrid_wal.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	invertedIndex := make(map[uint32][]core.Posting)
	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	idx := NewHNSWIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, nil, nil)

	// Insert with sparse data — WAL entries carry the full SparseVector.
	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{10.0},
	}, nil))
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0.99, 0.1, 0}, core.SparseVector{}, nil))

	// Simulate crash: close WAL without saving a snapshot.
	require.NoError(t, wal.Close())

	// Recovery: create a fresh hybrid-enabled index and replay the WAL.
	wal2, err := NewWAL(filepath.Join(t.TempDir(), "dummy.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal2.Close() })

	invertedIndex2 := make(map[uint32][]core.Posting)
	idMapper2 := core.NewIDMapper()
	recovered := NewHNSWIndex[[]float32](wal2, invertedIndex2, idMapper2, identityFunc, hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, nil, nil)

	require.NoError(t, recovered.ReplayWAL(walPath))
	require.Equal(t, 2, recovered.Len(), "both docs must be recovered from WAL")

	// The dense query is doc2's own vector, so doc2 wins on dense score alone.
	// doc1 can only rank first if replay restored its sparse vector.
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{5.0}}
	results, err := recovered.Search(context.Background(), []float32{0.99, 0.1, 0}, sparseQuery, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, "doc1", results[0].ID, "replay must restore doc1's sparse vector so it outranks doc2's dense match")
}

// TestHNSWIndex_Metadata_GOBRoundTrip_PreservesFilterability verifies that
// metadata of various types (string, int, float64, bool) survives a SaveToFile →
// LoadFromFile round-trip and that all filter types continue to match correctly.
// Note: metaindex.normalise() unifies numeric types via toFloat(), so int(42) and
// float64(42) produce the same filter key — the test exercises both.
func TestHNSWIndex_Metadata_GOBRoundTrip_PreservesFilterability(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(context.Background(), "doc1", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{
		"label":  "alpha",
		"count":  42,
		"score":  3.14,
		"active": true,
	}))
	require.NoError(t, idx.Insert(context.Background(), "doc2", []float32{0, 1, 0}, core.SparseVector{}, map[string]any{
		"label":  "beta",
		"count":  99,
		"active": false,
	}))

	path := filepath.Join(t.TempDir(), "hnsw_meta.bin")
	require.NoError(t, idx.SaveToFile(context.Background(), path))

	idx2 := newTestHNSWIndex(t)
	require.NoError(t, idx2.LoadFromFile(context.Background(), path))
	require.Equal(t, 2, idx2.Len())

	// String filter must survive GOB round-trip verbatim.
	results, err := idx2.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 2, map[string]any{"label": "alpha"})
	require.NoError(t, err)
	require.Len(t, results, 1, "string filter must work after GOB round-trip")
	assert.Equal(t, "doc1", results[0].ID)

	// Numeric filter: normalise() converts int/float64 to the same key, so 42 matches
	// regardless of whether GOB preserves the exact int type or widens to int64/float64.
	results, err = idx2.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 2, map[string]any{"count": 42})
	require.NoError(t, err)
	require.Len(t, results, 1, "numeric filter must work after GOB round-trip")
	assert.Equal(t, "doc1", results[0].ID)

	// Bool filter must survive GOB round-trip.
	results, err = idx2.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 2, map[string]any{"active": true})
	require.NoError(t, err)
	require.Len(t, results, 1, "bool filter must work after GOB round-trip")
	assert.Equal(t, "doc1", results[0].ID)

	// Float filter.
	results, err = idx2.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, 2, map[string]any{"score": 3.14})
	require.NoError(t, err)
	require.Len(t, results, 1, "float filter must work after GOB round-trip")
	assert.Equal(t, "doc1", results[0].ID)
}

func TestHNSWIndex_DistanceToScore_Cosine(t *testing.T) {
	idx := newTestHNSWIndex(t)
	idx.distanceMetric = "cosine"

	got := idx.distanceToScore(0.25)
	want := float32(0.75)
	assert.InDelta(t, want, got, 0.0001)
}

func TestHNSWIndex_DistanceToScore_Euclidean(t *testing.T) {
	idx := newTestHNSWIndex(t)
	idx.distanceMetric = "euclidean"

	got := idx.distanceToScore(4)
	want := float32(0.2) // 1 / (1 + 4)
	assert.InDelta(t, want, got, 0.0001)
}

func TestHNSWIndex_DistanceToScore_Euclidean_FartherIsLowerScore(t *testing.T) {
	idx := newTestHNSWIndex(t)
	idx.distanceMetric = "euclidean"

	near := idx.distanceToScore(1)
	far := idx.distanceToScore(100)
	assert.Greater(t, near, far)
}
