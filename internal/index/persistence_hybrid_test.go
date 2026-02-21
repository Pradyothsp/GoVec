package index

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

func TestPersistence_HybridSearch_SaveAndLoad(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/hybrid_snapshot.bin"

	// Insert documents with sparse vectors
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10, 20},
		Values:  []float32{1.5, 2.0},
	}, map[string]any{"category": "books"})

	_ = idx.Insert("doc2", []float32{0.0, 1.0}, core.SparseVector{
		Indices: []uint32{30, 40},
		Values:  []float32{3.0, 4.0},
	}, map[string]any{"category": "music"})

	// Save
	err := idx.SaveToFile(savePath)
	require.NoError(t, err, "Save should succeed")

	// Create new index and load
	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err, "Load should succeed")

	// Verify store was loaded
	require.Len(t, idx2.Store, 2, "Should load 2 documents")
	assert.Contains(t, idx2.Store, "doc1")
	assert.Contains(t, idx2.Store, "doc2")

	// Verify sparse vectors were loaded
	doc1 := idx2.Store["doc1"]
	assert.Equal(t, []uint32{10, 20}, doc1.Sparse.Indices)
	assert.Equal(t, []float32{1.5, 2.0}, doc1.Sparse.Values)

	doc2 := idx2.Store["doc2"]
	assert.Equal(t, []uint32{30, 40}, doc2.Sparse.Indices)
	assert.Equal(t, []float32{3.0, 4.0}, doc2.Sparse.Values)

	// Verify inverted index was loaded
	require.NotNil(t, idx2.InvertedIndex, "InvertedIndex should be loaded")
	assert.Contains(t, idx2.InvertedIndex, uint32(10), "Token 10 should be in inverted index")
	assert.Contains(t, idx2.InvertedIndex, uint32(20), "Token 20 should be in inverted index")
	assert.Contains(t, idx2.InvertedIndex, uint32(30), "Token 30 should be in inverted index")
	assert.Contains(t, idx2.InvertedIndex, uint32(40), "Token 40 should be in inverted index")

	// Verify posting lists
	postings10 := idx2.InvertedIndex[10]
	require.Len(t, postings10, 1)
	assert.Equal(t, "doc1", postings10[0].DocID)
	assert.Equal(t, float32(1.5), postings10[0].Weight)
}

func TestPersistence_HybridSearch_EmptyInvertedIndex(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/empty_hybrid_snapshot.bin"

	// Insert document with NO sparse vector
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{}, nil)

	// Save
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Load
	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Verify loaded correctly
	require.Len(t, idx2.Store, 1)
	assert.True(t, idx2.Store["doc1"].Sparse.IsEmpty())

	// Inverted index should be empty but not nil
	require.NotNil(t, idx2.InvertedIndex)
	assert.Empty(t, idx2.InvertedIndex)
}

func TestPersistence_HybridSearch_BackwardCompatibility_V1(t *testing.T) {
	// Create index WITHOUT hybrid search
	idx := newTestIndex(t) // No inverted index
	savePath := t.TempDir() + "/v1_snapshot.bin"

	// Insert documents (will create v1 snapshot without inverted index)
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{}, nil)

	// Save (should create v1 snapshot since InvertedIndex is nil)
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Create new index WITH hybrid search enabled
	idx2 := newTestIndexWithHybrid(t)

	// Load v1 snapshot into hybrid-enabled index (should work)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Verify data loaded
	require.Len(t, idx2.Store, 1)
	assert.Contains(t, idx2.Store, "doc1")

	// Inverted index should still exist (just empty)
	require.NotNil(t, idx2.InvertedIndex)
	assert.Empty(t, idx2.InvertedIndex)
}

func TestPersistence_HybridSearch_LoadV2IntoNonHybridIndex(t *testing.T) {
	// Create index WITH hybrid search
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/v2_hybrid_snapshot.bin"

	// Insert with sparse vectors
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{5.0},
	}, nil)

	// Save v2 snapshot with inverted index
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Create index WITHOUT hybrid search
	idx2 := newTestIndex(t) // nil InvertedIndex

	// Load v2 snapshot (should skip inverted index data)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Verify data loaded (but inverted index skipped)
	require.Len(t, idx2.Store, 1)
	assert.Contains(t, idx2.Store, "doc1")

	// Sparse vector should still be in the node (it's part of VectorNode)
	doc1 := idx2.Store["doc1"]
	assert.Equal(t, []uint32{10}, doc1.Sparse.Indices)

	// InvertedIndex should be nil (not populated)
	assert.Nil(t, idx2.InvertedIndex)
}

func TestPersistence_HybridSearch_SearchAfterLoad(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/search_after_load.bin"

	// Insert documents
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10, 20},
		Values:  []float32{2.0, 3.0},
	}, nil)

	_ = idx.Insert("doc2", []float32{0.0, 1.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{5.0},
	}, nil)

	// Save
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Load into new index
	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Perform hybrid search
	query := []float32{1.0, 0.0}
	sparseQuery := core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{1.0},
	}

	results, err := idx2.Search(query, sparseQuery, 2, nil)
	require.NoError(t, err)
	require.Len(t, results, 2)

	// doc2 should rank first (sparse boost outweighs dense difference)
	// doc1: 0.7*1.0 + 0.3*2.0 = 1.3
	// doc2: 0.7*0.0 + 0.3*5.0 = 1.5
	assert.Equal(t, "doc2", results[0].ID)
}

func TestPersistence_HybridSearch_MultipleTokens(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/multi_token_snapshot.bin"

	// Insert document with many tokens
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{1, 5, 10, 50, 100, 500, 1000},
		Values:  []float32{0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0},
	}, nil)

	// Save and load
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Verify all tokens in inverted index
	assert.Len(t, idx2.InvertedIndex, 7, "Should have 7 unique tokens")

	for _, tokenID := range []uint32{1, 5, 10, 50, 100, 500, 1000} {
		assert.Contains(t, idx2.InvertedIndex, tokenID)
		postings := idx2.InvertedIndex[tokenID]
		require.Len(t, postings, 1)
		assert.Equal(t, "doc1", postings[0].DocID)
	}
}

func TestPersistence_HybridSearch_LargeInvertedIndex(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/large_inverted_index.bin"

	// Insert many documents with overlapping tokens
	for i := 0; i < 100; i++ {
		id := string(rune('a'+i%26)) + string(rune('0'+i/26))
		vec := []float32{float32(i) * 0.01, float32(100-i) * 0.01}

		// Each doc has 3 tokens, some overlap
		sparse := core.SparseVector{
			Indices: []uint32{uint32(i % 10), uint32(i % 20), uint32(i % 30)},
			Values:  []float32{1.0, 2.0, 3.0},
		}

		_ = idx.Insert(id, vec, sparse, nil)
	}

	// Save
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Load
	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Verify all documents loaded
	assert.Len(t, idx2.Store, 100)

	// Verify inverted index has expected number of unique tokens
	//  are from i%10, i%20, i%30 for i=0..99
	// Unique values: 0-9 (from %10), 0-19 (from %20), 0-29 (from %30)
	// Total unique: max(10, 20, 30) = 30
	assert.LessOrEqual(t, len(idx2.InvertedIndex), 30)
	assert.GreaterOrEqual(t, len(idx2.InvertedIndex), 10)

	// Verify search works
	results, err := idx2.Search([]float32{0.5, 0.5}, core.SparseVector{
		Indices: []uint32{5},
		Values:  []float32{1.0},
	}, 10, nil)

	require.NoError(t, err)
	assert.LessOrEqual(t, len(results), 10)
}

func TestPersistence_HybridSearch_UpdateAfterLoad(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/update_after_load.bin"

	// Insert and save
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{1.0},
	}, nil)
	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Load
	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Update the document with a new sparse vector
	_ = idx2.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{20},
		Values:  []float32{2.0},
	}, nil)

	// Verify inverted index updated
	assert.NotContains(t, idx2.InvertedIndex, uint32(10), "Old token should be removed")
	assert.Contains(t, idx2.InvertedIndex, uint32(20), "New token should be added")
}

func TestPersistence_HybridSearch_DeleteAfterLoad(t *testing.T) {
	idx := newTestIndexWithHybrid(t)
	savePath := t.TempDir() + "/delete_after_load.bin"

	// Insert and save
	_ = idx.Insert("doc1", []float32{1.0, 0.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{1.0},
	}, nil)
	_ = idx.Insert("doc2", []float32{0.0, 1.0}, core.SparseVector{
		Indices: []uint32{10},
		Values:  []float32{2.0},
	}, nil)

	err := idx.SaveToFile(savePath)
	require.NoError(t, err)

	// Load
	idx2 := newTestIndexWithHybrid(t)
	err = idx2.LoadFromFile(savePath)
	require.NoError(t, err)

	// Verify token 10 has 2 postings
	require.Len(t, idx2.InvertedIndex[10], 2)

	// Delete doc1
	success, err := idx2.Delete("doc1")
	require.NoError(t, err)
	assert.True(t, success)

	// Verify inverted index updated (token 10 should have only 1 posting now)
	require.Len(t, idx2.InvertedIndex[10], 1)
	assert.Equal(t, "doc2", idx2.InvertedIndex[10][0].DocID)
}
