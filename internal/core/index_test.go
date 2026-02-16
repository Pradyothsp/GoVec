package core

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewVectorIndex(t *testing.T) {
	idx := NewVectorIndex()

	require.NotNil(t, idx, "NewVectorIndex should return non-nil")
	require.NotNil(t, idx.Store, "Store should be initialized")
	assert.Empty(t, idx.Store, "Store should be empty initially")
	assert.Equal(t, 0, len(idx.Store), "Store length should be 0")
}

func TestVectorIndex_Insert(t *testing.T) {
	tests := []struct {
		name     string
		id       string
		vector   []float32
		metadata map[string]any
	}{
		{
			name:     "insert_with_metadata",
			id:       "vec1",
			vector:   []float32{1.0, 2.0, 3.0},
			metadata: map[string]any{"label": "test", "count": 42},
		},
		{
			name:     "insert_without_metadata",
			id:       "vec2",
			vector:   []float32{4.0, 5.0},
			metadata: nil,
		},
		{
			name:     "insert_with_empty_metadata",
			id:       "vec3",
			vector:   []float32{1.5, 2.5, 3.5},
			metadata: map[string]any{},
		},
		{
			name:     "insert_with_nested_metadata",
			id:       "vec4",
			vector:   []float32{7.0, 8.0, 9.0},
			metadata: map[string]any{
				"category": "test",
				"nested": map[string]any{
					"level": 2,
					"active": true,
				},
				"tags": []string{"go", "vector", "db"},
			},
		},
		{
			name:     "insert_empty_vector",
			id:       "vec5",
			vector:   []float32{},
			metadata: map[string]any{"type": "empty"},
		},
		{
			name:     "insert_large_vector",
			id:       "vec6",
			vector:   make([]float32, 1536), // OpenAI embedding size
			metadata: map[string]any{"model": "text-embedding-ada-002"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := NewVectorIndex()
			idx.Insert(tt.id, tt.vector, tt.metadata)

			require.Contains(t, idx.Store, tt.id, "vector should be in store")

			node := idx.Store[tt.id]
			assert.Equal(t, tt.id, node.ID, "ID should match")
			assert.Equal(t, tt.vector, node.Vector, "Vector should match")
			assert.Equal(t, tt.metadata, node.Metadata, "Metadata should match")
		})
	}
}

func TestVectorIndex_Insert_Overwrite(t *testing.T) {
	idx := NewVectorIndex()

	// First insert
	idx.Insert("vec1", []float32{1.0, 2.0, 3.0}, map[string]any{"version": 1})

	require.Contains(t, idx.Store, "vec1")
	assert.Equal(t, []float32{1.0, 2.0, 3.0}, idx.Store["vec1"].Vector)
	assert.Equal(t, map[string]any{"version": 1}, idx.Store["vec1"].Metadata)

	// Overwrite with new vector
	idx.Insert("vec1", []float32{7.0, 8.0, 9.0}, map[string]any{"version": 2, "updated": true})

	require.Contains(t, idx.Store, "vec1")
	assert.Equal(t, []float32{7.0, 8.0, 9.0}, idx.Store["vec1"].Vector, "Vector should be updated")
	assert.Equal(t, map[string]any{"version": 2, "updated": true}, idx.Store["vec1"].Metadata, "Metadata should be updated")
	assert.Len(t, idx.Store, 1, "Should still have only one vector")
}

func TestVectorIndex_Insert_MultipleVectors(t *testing.T) {
	idx := NewVectorIndex()

	vectors := []struct {
		id     string
		vector []float32
		meta   map[string]any
	}{
		{"v1", []float32{1.0, 2.0}, map[string]any{"label": "first"}},
		{"v2", []float32{3.0, 4.0}, map[string]any{"label": "second"}},
		{"v3", []float32{5.0, 6.0}, nil},
		{"v4", []float32{7.0, 8.0}, map[string]any{"label": "fourth"}},
	}

	for _, v := range vectors {
		idx.Insert(v.id, v.vector, v.meta)
	}

	assert.Len(t, idx.Store, len(vectors), "Should have all vectors")

	for _, v := range vectors {
		require.Contains(t, idx.Store, v.id, "Vector %s should exist", v.id)
		assert.Equal(t, v.id, idx.Store[v.id].ID)
		assert.Equal(t, v.vector, idx.Store[v.id].Vector)
		assert.Equal(t, v.meta, idx.Store[v.id].Metadata)
	}
}

func TestVectorIndex_Insert_VariousMetadataTypes(t *testing.T) {
	idx := NewVectorIndex()

	testCases := []struct {
		name     string
		id       string
		metadata map[string]any
	}{
		{
			name: "string_values",
			id:   "v1",
			metadata: map[string]any{
				"text": "hello world",
				"name": "test",
			},
		},
		{
			name: "numeric_values",
			id:   "v2",
			metadata: map[string]any{
				"int":     42,
				"float":   3.14,
				"int64":   int64(9223372036854775807),
				"float32": float32(2.71),
			},
		},
		{
			name: "boolean_values",
			id:   "v3",
			metadata: map[string]any{
				"active":  true,
				"deleted": false,
			},
		},
		{
			name: "array_values",
			id:   "v4",
			metadata: map[string]any{
				"tags":    []string{"go", "database", "vector"},
				"numbers": []int{1, 2, 3, 4, 5},
			},
		},
		{
			name: "mixed_types",
			id:   "v5",
			metadata: map[string]any{
				"name":   "mixed",
				"count":  100,
				"active": true,
				"tags":   []string{"test"},
				"config": map[string]any{"setting": "value"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			vec := []float32{1.0, 2.0, 3.0}
			idx.Insert(tc.id, vec, tc.metadata)

			require.Contains(t, idx.Store, tc.id)
			assert.Equal(t, tc.metadata, idx.Store[tc.id].Metadata)
		})
	}
}

func TestVectorIndex_Insert_Concurrency(t *testing.T) {
	idx := NewVectorIndex()
	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vecID := fmt.Sprintf("vec%d", id)
			vec := []float32{float32(id), float32(id * 2)}
			meta := map[string]any{"id": id}
			idx.Insert(vecID, vec, meta)
		}(i)
	}

	wg.Wait()

	assert.Len(t, idx.Store, numGoroutines, "All vectors should be inserted")

	for i := 0; i < numGoroutines; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		require.Contains(t, idx.Store, vecID, "Vector %s should exist", vecID)

		node := idx.Store[vecID]
		assert.Equal(t, vecID, node.ID)
		assert.Len(t, node.Vector, 2)
		assert.Equal(t, i, node.Metadata["id"])
	}
}

func TestVectorIndex_Insert_ConcurrentOverwrites(t *testing.T) {
	idx := NewVectorIndex()
	var wg sync.WaitGroup
	numGoroutines := 50
	vecID := "shared_vector"

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(iteration int) {
			defer wg.Done()
			vec := []float32{float32(iteration)}
			meta := map[string]any{"iteration": iteration}
			idx.Insert(vecID, vec, meta)
		}(i)
	}

	wg.Wait()

	require.Contains(t, idx.Store, vecID, "Shared vector should exist")
	assert.NotNil(t, idx.Store[vecID], "Vector should not be nil")
	assert.NotNil(t, idx.Store[vecID].Vector, "Vector data should not be nil")
}

func TestVectorIndex_Insert_EdgeCases(t *testing.T) {
	t.Run("empty_id", func(t *testing.T) {
		idx := NewVectorIndex()
		idx.Insert("", []float32{1.0, 2.0}, nil)

		require.Contains(t, idx.Store, "", "Empty ID should be allowed")
		assert.Equal(t, "", idx.Store[""].ID)
	})

	t.Run("special_characters_in_id", func(t *testing.T) {
		idx := NewVectorIndex()
		specialIDs := []string{
			"vec-with-dashes",
			"vec_with_underscores",
			"vec.with.dots",
			"vec:with:colons",
			"vec/with/slashes",
			"vec with spaces",
			"vec@special!chars#",
			"中文ID",
			"emoji🚀vector",
		}

		for _, id := range specialIDs {
			idx.Insert(id, []float32{1.0}, nil)
			require.Contains(t, idx.Store, id, "ID with special characters should be allowed: %s", id)
		}

		assert.Len(t, idx.Store, len(specialIDs))
	})

	t.Run("nil_vector", func(t *testing.T) {
		idx := NewVectorIndex()
		idx.Insert("nil_vec", nil, map[string]any{"type": "nil"})

		require.Contains(t, idx.Store, "nil_vec")
		assert.Nil(t, idx.Store["nil_vec"].Vector)
	})

	t.Run("very_large_dimension", func(t *testing.T) {
		idx := NewVectorIndex()
		largeVec := make([]float32, 4096)
		for i := range largeVec {
			largeVec[i] = float32(i)
		}

		idx.Insert("large", largeVec, nil)

		require.Contains(t, idx.Store, "large")
		assert.Len(t, idx.Store["large"].Vector, 4096)
	})
}

func TestVectorIndex_Search_EmptyIndex(t *testing.T) {
	idx := NewVectorIndex()
	query := []float32{1.0, 2.0, 3.0}

	results, err := idx.Search(query, 5)

	require.NoError(t, err)
	assert.Empty(t, results, "Search on empty index should return empty results")
}

func TestVectorIndex_Search_SingleVector(t *testing.T) {
	idx := NewVectorIndex()
	idx.Insert("v1", []float32{1.0, 2.0, 3.0}, map[string]any{"label": "test"})

	query := []float32{1.0, 2.0, 3.0}
	results, err := idx.Search(query, 5)

	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "v1", results[0].ID)
	assert.InDelta(t, 1.0, results[0].Score, 0.0001, "Identical vector should have score 1.0")
	assert.Equal(t, "test", results[0].Meta["label"])
}

func TestVectorIndex_Search_MultipleVectors(t *testing.T) {
	idx := NewVectorIndex()

	// Insert vectors with different similarities to query
	idx.Insert("identical", []float32{1.0, 0.0, 0.0}, map[string]any{"type": "identical"})
	idx.Insert("similar", []float32{0.9, 0.1, 0.0}, map[string]any{"type": "similar"})
	idx.Insert("orthogonal", []float32{0.0, 1.0, 0.0}, map[string]any{"type": "orthogonal"})
	idx.Insert("opposite", []float32{-1.0, 0.0, 0.0}, map[string]any{"type": "opposite"})

	query := []float32{1.0, 0.0, 0.0}
	results, err := idx.Search(query, 10)

	require.NoError(t, err)
	require.Len(t, results, 4)

	// Verify results are sorted by score (descending)
	assert.Equal(t, "identical", results[0].ID)
	assert.InDelta(t, 1.0, results[0].Score, 0.001)

	assert.Equal(t, "similar", results[1].ID)
	assert.Greater(t, results[1].Score, float32(0.8))

	assert.Equal(t, "orthogonal", results[2].ID)
	assert.InDelta(t, 0.0, results[2].Score, 0.001)

	assert.Equal(t, "opposite", results[3].ID)
	assert.InDelta(t, -1.0, results[3].Score, 0.001)
}

func TestVectorIndex_Search_WithLimit(t *testing.T) {
	idx := NewVectorIndex()

	// Insert 10 vectors (start from 1 to avoid zero vector)
	for i := 1; i <= 10; i++ {
		vec := []float32{float32(i), float32(i * 2)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, map[string]any{"index": i})
	}

	query := []float32{5.0, 10.0}

	tests := []struct {
		name          string
		limit         int
		expectedCount int
	}{
		{"limit_1", 1, 1},
		{"limit_3", 3, 3},
		{"limit_5", 5, 5},
		{"limit_10", 10, 10},
		{"limit_greater_than_total", 20, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.Search(query, tt.limit)
			require.NoError(t, err)
			assert.Len(t, results, tt.expectedCount)
		})
	}
}

func TestVectorIndex_Search_WithZeroLimit(t *testing.T) {
	idx := NewVectorIndex()

	for i := 1; i <= 5; i++ {
		vec := []float32{float32(i), float32(i * 2)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, nil)
	}

	query := []float32{1.0, 2.0}
	results, err := idx.Search(query, 0)

	require.NoError(t, err)
	assert.Len(t, results, 5, "Zero limit should return all results")
}

func TestVectorIndex_Search_EmptyQuery(t *testing.T) {
	idx := NewVectorIndex()
	idx.Insert("v1", []float32{1.0, 2.0}, nil)

	query := []float32{}
	results, err := idx.Search(query, 5)

	require.Error(t, err)
	assert.Equal(t, "empty query vector", err.Error())
	assert.Nil(t, results)
}

func TestVectorIndex_Search_DimensionMismatch(t *testing.T) {
	idx := NewVectorIndex()
	idx.Insert("v1", []float32{1.0, 2.0, 3.0}, nil)
	idx.Insert("v2", []float32{4.0, 5.0, 6.0}, nil)

	// Query with different dimension
	query := []float32{1.0, 2.0}
	results, err := idx.Search(query, 5)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector dimensions mismatch")
	assert.Nil(t, results)
}

func TestVectorIndex_Search_ResultsIncludeMetadata(t *testing.T) {
	idx := NewVectorIndex()

	metadata := map[string]any{
		"category": "test",
		"count":    42,
		"active":   true,
		"tags":     []string{"go", "vector"},
	}

	idx.Insert("v1", []float32{1.0, 2.0, 3.0}, metadata)

	query := []float32{1.0, 2.0, 3.0}
	results, err := idx.Search(query, 1)

	require.NoError(t, err)
	require.Len(t, results, 1)

	assert.Equal(t, "test", results[0].Meta["category"])
	assert.Equal(t, 42, results[0].Meta["count"])
	assert.Equal(t, true, results[0].Meta["active"])
	assert.NotNil(t, results[0].Meta["tags"])
}

func TestVectorIndex_Search_SortingOrder(t *testing.T) {
	idx := NewVectorIndex()

	// Insert vectors with known similarity scores to query [1,0,0]
	idx.Insert("high", []float32{1.0, 0.0, 0.0}, nil)      // score = 1.0
	idx.Insert("medium", []float32{0.7, 0.7, 0.0}, nil)    // score ≈ 0.7
	idx.Insert("low", []float32{0.5, 0.866, 0.0}, nil)     // score ≈ 0.5
	idx.Insert("negative", []float32{-1.0, 0.0, 0.0}, nil) // score = -1.0

	query := []float32{1.0, 0.0, 0.0}
	results, err := idx.Search(query, 10)

	require.NoError(t, err)
	require.Len(t, results, 4)

	// Verify descending order by score
	for i := 0; i < len(results)-1; i++ {
		assert.GreaterOrEqual(t, results[i].Score, results[i+1].Score,
			"Results should be sorted by score descending")
	}

	assert.Equal(t, "high", results[0].ID)
	assert.Equal(t, "negative", results[len(results)-1].ID)
}

func TestVectorIndex_Search_IdenticalVectors(t *testing.T) {
	idx := NewVectorIndex()

	// Insert multiple identical vectors
	for i := 0; i < 3; i++ {
		idx.Insert(fmt.Sprintf("v%d", i), []float32{1.0, 2.0, 3.0}, map[string]any{"id": i})
	}

	query := []float32{1.0, 2.0, 3.0}
	results, err := idx.Search(query, 10)

	require.NoError(t, err)
	require.Len(t, results, 3)

	// All should have score 1.0
	for _, result := range results {
		assert.InDelta(t, 1.0, result.Score, 0.0001)
	}
}

func TestVectorIndex_Search_Concurrency(t *testing.T) {
	idx := NewVectorIndex()

	// Insert test vectors (start from 1 to avoid zero vector)
	for i := 1; i <= 100; i++ {
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, map[string]any{"index": i})
	}

	var wg sync.WaitGroup
	numGoroutines := 50
	query := []float32{50.0, 100.0, 150.0}

	// Run concurrent searches
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results, err := idx.Search(query, 10)
			assert.NoError(t, err)
			assert.Len(t, results, 10)

			// Verify sorting
			for j := 0; j < len(results)-1; j++ {
				assert.GreaterOrEqual(t, results[j].Score, results[j+1].Score)
			}
		}()
	}

	wg.Wait()
}

func TestVectorIndex_Search_ConcurrentInsertAndSearch(t *testing.T) {
	idx := NewVectorIndex()

	// Pre-populate with some vectors (start from 1 to avoid zero vector)
	for i := 1; i <= 50; i++ {
		vec := []float32{float32(i), float32(i * 2)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, nil)
	}

	var wg sync.WaitGroup
	query := []float32{25.0, 50.0}

	// Concurrent searches
	for i := 0; i < 25; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results, err := idx.Search(query, 5)
			assert.NoError(t, err)
			assert.NotNil(t, results)
		}()
	}

	// Concurrent inserts
	for i := 51; i <= 75; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vec := []float32{float32(id), float32(id * 2)}
			idx.Insert(fmt.Sprintf("v%d", id), vec, nil)
		}(i)
	}

	wg.Wait()
}

func TestVectorIndex_Search_HighDimensional(t *testing.T) {
	idx := NewVectorIndex()

	// Test with 1536-dimensional vectors (OpenAI embedding size)
	vec1 := make([]float32, 1536)
	vec2 := make([]float32, 1536)
	vec3 := make([]float32, 1536)

	for i := 0; i < 1536; i++ {
		vec1[i] = float32(i+1) * 0.001
		vec2[i] = float32(i+1) * 0.001
		// vec3 has different direction (reversed second half)
		if i < 768 {
			vec3[i] = float32(i+1) * 0.001
		} else {
			vec3[i] = float32(1536-i) * 0.001
		}
	}

	idx.Insert("v1", vec1, map[string]any{"type": "identical"})
	idx.Insert("v2", vec2, map[string]any{"type": "identical"})
	idx.Insert("v3", vec3, map[string]any{"type": "different"})

	results, err := idx.Search(vec1, 3)

	require.NoError(t, err)
	require.Len(t, results, 3)

	// First two should have very high similarity (identical)
	assert.InDelta(t, 1.0, results[0].Score, 0.001)
	assert.InDelta(t, 1.0, results[1].Score, 0.001)
	// Third should have lower similarity (different direction)
	assert.Less(t, results[2].Score, results[1].Score)
}

func TestVectorIndex_Search_NilMetadata(t *testing.T) {
	idx := NewVectorIndex()
	idx.Insert("v1", []float32{1.0, 2.0}, nil)
	idx.Insert("v2", []float32{2.0, 3.0}, nil)

	query := []float32{1.0, 2.0}
	results, err := idx.Search(query, 2)

	require.NoError(t, err)
	require.Len(t, results, 2)

	// Metadata should be nil, not cause errors
	for _, result := range results {
		assert.Nil(t, result.Meta)
	}
}
