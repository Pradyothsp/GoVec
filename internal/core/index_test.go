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
