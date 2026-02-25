package index

import (
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
	return NewHNSWIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity, hnsw.CosineDistanceFloat32, 16, 20)
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
	return NewHNSWIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, core.CosineSimilarity, hnsw.CosineDistanceFloat32, 16, 20)
}

// =============================================================================
// Basic Insert / Search / Delete
// =============================================================================

func TestHNSWIndex_InsertAndLen(t *testing.T) {
	idx := newTestHNSWIndex(t)
	assert.Equal(t, 0, idx.Len())

	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	assert.Equal(t, 1, idx.Len())

	require.NoError(t, idx.Insert("v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	assert.Equal(t, 2, idx.Len())
}

func TestHNSWIndex_Insert_Overwrite(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"version": 1}))
	assert.Equal(t, 1, idx.Len())

	// Overwrite same ID — count must not increase
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"version": 2}))
	assert.Equal(t, 1, idx.Len())

	// Metadata should reflect the latest insert
	internalID, err := idx.IDMapper.ToUint32ID("v1")
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"version": 2}, idx.metadata[internalID].Metadata)
}

func TestHNSWIndex_Search_EmptyIndex(t *testing.T) {
	idx := newTestHNSWIndex(t)

	results, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 5, nil)
	require.NoError(t, err)
	assert.Nil(t, results)
}

func TestHNSWIndex_Search_EmptyQuery(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	_, err := idx.Search([]float32{}, core.SparseVector{}, 5, nil)
	assert.Error(t, err)
}

func TestHNSWIndex_Delete_ExistingVector(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	assert.Equal(t, 2, idx.Len())

	internalID, err := idx.IDMapper.ToUint32ID("v1")
	require.NoError(t, err)

	deleted, err := idx.Delete("v1")
	require.NoError(t, err)
	assert.True(t, deleted)
	assert.Equal(t, 1, idx.Len())

	// ID is gone from metadata and is tombstoned
	assert.NotContains(t, idx.metadata, internalID)
	assert.True(t, idx.IDMapper.IsTombstone(internalID))
}

func TestHNSWIndex_Delete_NonExistentVector(t *testing.T) {
	idx := newTestHNSWIndex(t)

	deleted, err := idx.Delete("does-not-exist")
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
	require.NoError(t, idx.Insert("doc3", []float32{0, 1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("doc2", []float32{0.9, 0.1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("doc1", []float32{1, 0, 0}, core.SparseVector{}, nil))

	results, err := idx.Search([]float32{1, 0, 0}, core.SparseVector{}, 3, nil)
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
		require.NoError(t, idx.Insert(fixtures.GenerateMetadata(1)[0]["id"].(string)+string(rune('A'+i)), vec, core.SparseVector{}, nil))
	}

	results, err := idx.Search([]float32{1, 0, 0}, core.SparseVector{}, 2, nil)
	require.NoError(t, err)
	assert.Len(t, results, 2, "result count must equal k")
}

// =============================================================================
// Metadata filtering
// =============================================================================

func TestHNSWIndex_Search_WithFilter(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert("doc1", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{"category": "A"}))
	require.NoError(t, idx.Insert("doc2", []float32{0.9, 0.1, 0}, core.SparseVector{}, map[string]any{"category": "B"}))
	require.NoError(t, idx.Insert("doc3", []float32{0.8, 0.2, 0}, core.SparseVector{}, map[string]any{"category": "A"}))
	require.NoError(t, idx.Insert("doc4", []float32{0, 1, 0}, core.SparseVector{}, map[string]any{"category": "B"}))
	require.NoError(t, idx.Insert("doc5", []float32{0, 0, 1}, core.SparseVector{}, map[string]any{"category": "A"}))

	// Over-fetching kicks in (k*5 candidates fetched internally, then filtered)
	results, err := idx.Search([]float32{1, 0, 0}, core.SparseVector{}, 10, map[string]interface{}{"category": "A"})
	require.NoError(t, err)

	for _, r := range results {
		assert.Equal(t, "A", r.Meta["category"], "filter must exclude category B docs")
	}
	assert.LessOrEqual(t, len(results), 3, "at most 3 category-A docs exist")
}

func TestHNSWIndex_Search_NoMetadataMatch(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert("doc1", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{"category": "A"}))
	require.NoError(t, idx.Insert("doc2", []float32{0, 1, 0}, core.SparseVector{}, map[string]any{"category": "A"}))

	results, err := idx.Search([]float32{1, 0, 0}, core.SparseVector{}, 5, map[string]interface{}{"category": "Z"})
	require.NoError(t, err)
	assert.Empty(t, results)
}

// =============================================================================
// Hybrid search (dense + sparse)
// =============================================================================

func TestHNSWIndex_HybridSearch_SparseBoostsRanking(t *testing.T) {
	idx := newTestHNSWIndexWithHybrid(t)

	// doc1: great dense match, weak sparse match
	require.NoError(t, idx.Insert("doc1", []float32{1, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{0.1},
	}, nil))

	// doc2: slightly weaker dense match, very strong sparse match
	require.NoError(t, idx.Insert("doc2", []float32{0.9, 0.1, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{10.0},
	}, nil))

	require.NoError(t, idx.Insert("doc3", []float32{0, 1, 0}, core.SparseVector{}, nil))

	query := []float32{1, 0, 0}
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{5.0}}

	results, err := idx.Search(query, sparseQuery, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)

	// doc2 should rank first because strong sparse match overcomes slight dense gap
	assert.Equal(t, "doc2", results[0].ID, "sparse boost should elevate doc2 above doc1")
}

func TestHNSWIndex_HybridSearch_EmptySparseQueryFallsBackToDense(t *testing.T) {
	idx := newTestHNSWIndexWithHybrid(t)

	require.NoError(t, idx.Insert("doc1", []float32{1, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{5.0},
	}, nil))
	require.NoError(t, idx.Insert("doc2", []float32{0, 1, 0}, core.SparseVector{}, nil))

	// Empty sparse query → dense-only path
	results, err := idx.Search([]float32{1, 0, 0}, core.SparseVector{}, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)

	assert.Equal(t, "doc1", results[0].ID, "without sparse query, dense similarity governs ranking")
}

// =============================================================================
// Persistence round-trip
// =============================================================================

func TestHNSWIndex_PersistenceRoundTrip(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert("v1", []float32{1, 0, 0}, core.SparseVector{}, fixtures.MetaSimple))
	require.NoError(t, idx.Insert("v2", []float32{0, 1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("v3", []float32{0, 0, 1}, core.SparseVector{}, fixtures.MetaNested))

	path := filepath.Join(t.TempDir(), "hnsw.bin")
	require.NoError(t, idx.SaveToFile(path))

	// Verify no temp file leaked
	_, err := os.Stat(path + ".tmp")
	assert.True(t, os.IsNotExist(err), "temp file must be cleaned up after save")

	// Load into a fresh index
	idx2 := newTestHNSWIndex(t)
	require.NoError(t, idx2.LoadFromFile(path))

	assert.Equal(t, 3, idx2.Len())
	assert.Equal(t, idx.IDMapper.Count(), idx2.IDMapper.Count())
	assert.Equal(t, idx.IDMapper.NextID(), idx2.IDMapper.NextID())

	// Search still returns the right top result
	results, err := idx2.Search([]float32{1, 0, 0}, core.SparseVector{}, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "v1", results[0].ID)
}

func TestHNSWIndex_LoadFromFile_MissingFile(t *testing.T) {
	idx := newTestHNSWIndex(t)
	err := idx.LoadFromFile("/nonexistent/hnsw.bin")
	assert.NoError(t, err, "missing file must be handled gracefully")
	assert.Equal(t, 0, idx.Len())
}

func TestHNSWIndex_LoadFromFile_CorruptedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	require.NoError(t, os.WriteFile(path, []byte("not valid gob data"), 0o600))

	idx := newTestHNSWIndex(t)
	err := idx.LoadFromFile(path)
	assert.Error(t, err)
}

func TestHNSWIndex_LoadFromFile_BruteForceSnapshot(t *testing.T) {
	// Save a brute-force (v3) snapshot
	bruteIdx := newTestIndex(t)
	require.NoError(t, bruteIdx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	path := filepath.Join(t.TempDir(), "brute.bin")
	require.NoError(t, bruteIdx.SaveToFile(path))

	// Attempting to load it as an HNSW snapshot must fail with a clear message
	hnswIdx := newTestHNSWIndex(t)
	err := hnswIdx.LoadFromFile(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "brute", "error must hint that the snapshot is brute-force")
}

func TestHNSWIndex_SaveToFile_InvalidPath(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	err := idx.SaveToFile("/nonexistent/dir/hnsw.bin")
	assert.Error(t, err)
}

// Verify that a VectorIndex refuses to load an HNSW snapshot
func TestVectorIndex_LoadFromFile_RejectsHNSWSnapshot(t *testing.T) {
	hnswIdx := newTestHNSWIndex(t)
	require.NoError(t, hnswIdx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

	path := filepath.Join(t.TempDir(), "hnsw.bin")
	require.NoError(t, hnswIdx.SaveToFile(path))

	bruteIdx := newTestIndex(t)
	err := bruteIdx.LoadFromFile(path)
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
	idx := NewHNSWIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity, hnsw.CosineDistanceFloat32, 16, 20)

	// Insert writes go to WAL first
	require.NoError(t, idx.Insert("v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("v2", []float32{0, 1, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("v3", []float32{0, 0, 1}, core.SparseVector{}, nil))

	// Simulate crash: create a fresh index and replay the WAL
	wal2, err := NewWAL(filepath.Join(t.TempDir(), "dummy.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal2.Close() })

	idMapper2 := core.NewIDMapper()
	recovered := NewHNSWIndex[[]float32](wal2, nil, idMapper2, identityFunc, core.CosineSimilarity, hnsw.CosineDistanceFloat32, 16, 20)

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
	idx := NewHNSWIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity, hnsw.CosineDistanceFloat32, 16, 20)

	require.NoError(t, idx.Insert("v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("v2", []float32{0, 1, 0}, core.SparseVector{}, nil))
	deleted, err := idx.Delete("v1")
	require.NoError(t, err)
	require.True(t, deleted)

	// Replay on fresh index
	wal2, err := NewWAL(filepath.Join(t.TempDir(), "dummy.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal2.Close() })

	idMapper2 := core.NewIDMapper()
	recovered := NewHNSWIndex[[]float32](wal2, nil, idMapper2, identityFunc, core.CosineSimilarity, hnsw.CosineDistanceFloat32, 16, 20)

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

	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert("v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
	assert.Equal(t, 2, idx.Len())

	idx.Clear()
	assert.Equal(t, 0, idx.Len())
	assert.Empty(t, idx.metadata)
}

func TestHNSWIndex_Clear_AllowsReinsertion(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	idx.Clear()

	// Should be able to insert again after Clear
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	assert.Equal(t, 1, idx.Len())

	results, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 1, nil)
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
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "free"}))
	require.NoError(t, idx.Insert("v2", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"tier": "premium"}))

	results, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 2, nil)
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
		require.NoError(t, idx.Insert(
			fmt.Sprintf("v%d", i), vec, core.SparseVector{}, map[string]any{"tier": tier},
		))
	}

	results, err := idx.Search(vecs[10], core.SparseVector{}, 2, map[string]any{"tier": "rare"})
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
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "free"}))

	results, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "nonexistent"})
	require.NoError(t, err)
	assert.Nil(t, results, "filter matching nothing must short-circuit and return nil")
}

// TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnOverwrite verifies that overwriting
// a vector updates the MetadataIndex so old filters no longer match.
func TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnOverwrite(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "free"}))

	// Overwrite with new metadata
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "premium"}))

	old, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "free"})
	require.NoError(t, err)
	assert.Nil(t, old, "old tier must no longer match after overwrite")

	updated, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "premium"})
	require.NoError(t, err)
	require.Len(t, updated, 1)
	assert.Equal(t, "v1", updated[0].ID)
}

// TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnDelete verifies that deleting
// a vector removes it from the MetadataIndex.
func TestHNSWIndex_FilteredSearch_MetaIndex_UpdatedOnDelete(t *testing.T) {
	idx := newTestHNSWIndex(t)
	require.NoError(t, idx.Insert("v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"tier": "premium"}))
	require.NoError(t, idx.Insert("v2", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"tier": "free"}))

	deleted, err := idx.Delete("v1")
	require.NoError(t, err)
	require.True(t, deleted)

	results, err := idx.Search(fixtures.Vec3dSimple, core.SparseVector{}, 5, map[string]any{"tier": "premium"})
	require.NoError(t, err)
	assert.Nil(t, results, "deleted vector must not appear in filtered results")
}
