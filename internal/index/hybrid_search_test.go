package index

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// newTestIndexWithHybrid creates a VectorIndex with hybrid search enabled
func newTestIndexWithHybrid(t testing.TB) *VectorIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("newTestIndexWithHybrid: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	// Create inverted index (enables hybrid search)
	invertedIndex := make(map[uint32][]core.Posting)

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	return NewVectorIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, core.CosineSimilarity, nil)
}

func TestHybridSearch_DenseOnly(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	// Insert documents with only dense vectors (no sparse)
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{}, map[string]any{"type": "doc1"})
	_ = idx.Insert("doc2", []float32{0.9, 0.1}, core.SparseVector{}, map[string]any{"type": "doc2"})
	_ = idx.Insert("doc3", []float32{0.0, 1.0}, core.SparseVector{}, map[string]any{"type": "doc3"})

	// Search with dense query, no sparse query
	query := []float32{1.0, 0.0}
	results, err := idx.Search(query, core.SparseVector{}, 3, nil)

	require.NoError(t, err)
	require.Len(t, results, 3)

	// Verify ranking by dense similarity (doc1 should be first)
	assert.Equal(t, "doc1", results[0].ID, "Dense-only search should rank by cosine similarity")
	assert.Equal(t, "doc2", results[1].ID)
	assert.Equal(t, "doc3", results[2].ID)
}

func TestHybridSearch_SparseBoostsRanking(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	// Insert documents with both dense and sparse vectors
	_ = idx.Insert("doc1", []float32{0.5, 0.5}, core.SparseVector{
		Indices: []uint32{10, 20},
		Values:  []float32{0.1, 0.2},
	}, nil)

	_ = idx.Insert("doc2", []float32{0.4, 0.6}, core.SparseVector{
		Indices: []uint32{10, 30},
		Values:  []float32{5.0, 1.0}, // Strong match on token 10
	}, nil)

	// Search with hybrid query
	query := []float32{0.5, 0.5}
	sparseQuery := core.SparseVector{
		Indices: []uint32{10, 40},
		Values:  []float32{10.0, 2.0}, // Query for token 10
	}

	results, err := idx.Search(query, sparseQuery, 2, nil)

	require.NoError(t, err)
	require.Len(t, results, 2)

	// doc2 should rank higher due to strong sparse match on token 10
	// even though doc1 has slightly better dense similarity
	assert.Equal(t, "doc2", results[0].ID, "Sparse match should boost ranking")
	assert.Equal(t, "doc1", results[1].ID)
}

func TestHybridSearch_MixedDocuments(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	// Insert documents: some with sparse, some without
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{100},
		Values:  []float32{5.0},
	}, nil)

	_ = idx.Insert("doc2", []float32{0.9, 0.1}, core.SparseVector{}, nil) // No sparse

	_ = idx.Insert("doc3", []float32{0.8, 0.2}, core.SparseVector{
		Indices: []uint32{200},
		Values:  []float32{3.0},
	}, nil)

	// Search with sparse query matching doc1
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{100},
		Values:  []float32{2.0},
	}

	results, err := idx.Search(query, sparseQuery, 3, nil)

	require.NoError(t, err)
	require.Len(t, results, 3)

	// doc1 should rank first (perfect dense match + sparse boost)
	assert.Equal(t, "doc1", results[0].ID)
}

func TestHybridSearch_EmptySparseQuery(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{5.0},
	}, nil)

	_ = idx.Insert("doc2", []float32{0.0, 1.0}, core.SparseVector{
		Indices: []uint32{20},
		Values:  []float32{3.0},
	}, nil)

	// Search with empty sparse query (should fall back to dense-only)
	query := []float32{1.0, 0.0}
	results, err := idx.Search(query, core.SparseVector{}, 2, nil)

	require.NoError(t, err)
	require.Len(t, results, 2)

	// Should rank by dense similarity only
	assert.Equal(t, "doc1", results[0].ID)
}

func TestHybridSearch_NoOverlappingTokens(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10, 20},
		Values:  []float32{1.0, 2.0},
	}, nil)

	_ = idx.Insert("doc2", []float32{0.9, 0.1}, core.SparseVector{
		Indices: []uint32{30, 40},
		Values:  []float32{1.0, 2.0},
	}, nil)

	// Search with sparse query that doesn't match any document tokens
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{50, 60}, // No overlap with doc tokens
		Values:  []float32{5.0, 3.0},
	}

	results, err := idx.Search(query, sparseQuery, 2, nil)

	require.NoError(t, err)
	require.Len(t, results, 2)

	// Should rank by dense similarity (sparse contributes 0.0)
	assert.Equal(t, "doc1", results[0].ID)
}

func TestHybridSearch_WithFilters(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{5.0},
	}, map[string]any{"category": "books"})

	_ = idx.Insert("doc2", []float32{0.9, 0.1}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{8.0}, // Stronger sparse match
	}, map[string]any{"category": "music"})

	// Search with filter (should exclude doc2)
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{2.0},
	}
	filters := map[string]interface{}{"category": "books"}

	results, err := idx.Search(query, sparseQuery, 5, nil)

	require.NoError(t, err)

	// Without filter, doc2 would rank higher due to sparse boost
	// Verify doc2 is included when no filter
	assert.Len(t, results, 2)

	// With filter, only doc1 should be returned
	resultsFiltered, err := idx.Search(query, sparseQuery, 5, filters)
	require.NoError(t, err)
	require.Len(t, resultsFiltered, 1)
	assert.Equal(t, "doc1", resultsFiltered[0].ID)
}

func TestHybridSearch_MultipleTokenOverlap(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	// Document with multiple matching tokens
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10, 20, 30},
		Values:  []float32{1.0, 2.0, 3.0},
	}, nil)

	// Document with single matching token but high weight
	_ = idx.Insert("doc2", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{10.0},
	}, nil)

	// Query matching multiple tokens in doc1
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{10, 20, 30},
		Values:  []float32{1.0, 1.0, 1.0},
	}

	results, err := idx.Search(query, sparseQuery, 2, nil)

	require.NoError(t, err)
	require.Len(t, results, 2)

	// doc1 should rank higher with multiple token matches
	// Sparse score for doc1: (1*1) + (2*1) + (3*1) = 6.0
	// Sparse score for doc2: (10*1) = 10.0
	// But doc2 has higher sparse score, so it should rank first
	assert.Equal(t, "doc2", results[0].ID)
}

func TestHybridSearch_ScoreCombination(t *testing.T) {
	idx := newTestIndexWithHybrid(t)

	// Insert document with known dense and sparse vectors
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{1.0},
	}, nil)

	// Search with identical query (perfect match on both)
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{1.0},
	}

	results, err := idx.Search(query, sparseQuery, 1, nil)

	require.NoError(t, err)
	require.Len(t, results, 1)

	// Dense score = 1.0 (perfect cosine similarity)
	// Sparse score = 1.0 (1.0 * 1.0)
	// Final score = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
	assert.InDelta(t, 1.0, results[0].Score, 0.01, "Perfect match should have score ~1.0")
}

func TestHybridSearch_NoInvertedIndex(t *testing.T) {
	// Create index WITHOUT inverted index (hybrid search disabled)
	idx := newTestIndex(t) // Uses nil invertedIndex

	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{5.0},
	}, nil)

	// Search with sparse query (should be ignored)
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{2.0},
	}

	results, err := idx.Search(query, sparseQuery, 1, nil)

	require.NoError(t, err)
	require.Len(t, results, 1)

	// Should fall back to dense-only search
	// Dense score = 1.0 (perfect match)
	assert.InDelta(t, 1.0, results[0].Score, 0.01)
}
