package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVectorNode_Creation(t *testing.T) {
	t.Run("with_all_fields", func(t *testing.T) {
		id := "test_vector"
		vector := []float32{1.0, 2.0, 3.0}
		metadata := map[string]any{
			"label": "test",
			"count": 42,
		}

		node := &VectorNode[[]float32]{
			ID:       id,
			Vector:   vector,
			Metadata: metadata,
		}

		require.NotNil(t, node)
		assert.Equal(t, id, node.ID)
		assert.Equal(t, vector, node.Vector)
		assert.Equal(t, metadata, node.Metadata)
	})

	t.Run("with_nil_metadata", func(t *testing.T) {
		node := &VectorNode[[]float32]{
			ID:       "vec1",
			Vector:   []float32{4.0, 5.0},
			Metadata: nil,
		}

		require.NotNil(t, node)
		assert.Equal(t, "vec1", node.ID)
		assert.Equal(t, []float32{4.0, 5.0}, node.Vector)
		assert.Nil(t, node.Metadata)
	})

	t.Run("with_empty_metadata", func(t *testing.T) {
		node := &VectorNode[[]float32]{
			ID:       "vec2",
			Vector:   []float32{1.0},
			Metadata: map[string]any{},
		}

		require.NotNil(t, node)
		assert.Empty(t, node.Metadata)
		assert.NotNil(t, node.Metadata)
	})

	t.Run("with_nil_vector", func(t *testing.T) {
		node := &VectorNode[[]float32]{
			ID:       "vec3",
			Vector:   nil,
			Metadata: map[string]any{"type": "placeholder"},
		}

		require.NotNil(t, node)
		assert.Nil(t, node.Vector)
	})

	t.Run("with_empty_vector", func(t *testing.T) {
		node := &VectorNode[[]float32]{
			ID:       "vec4",
			Vector:   []float32{},
			Metadata: nil,
		}

		require.NotNil(t, node)
		assert.Empty(t, node.Vector)
		assert.NotNil(t, node.Vector)
	})

	t.Run("zero_value", func(t *testing.T) {
		var node VectorNode[[]float32]

		assert.Equal(t, "", node.ID)
		assert.Nil(t, node.Vector)
		assert.Nil(t, node.Metadata)
	})
}

func TestVectorNode_FieldTypes(t *testing.T) {
	t.Run("id_is_string", func(t *testing.T) {
		node := &VectorNode[[]float32]{ID: "test"}
		assert.IsType(t, "", node.ID)
	})

	t.Run("vector_is_float32_slice", func(t *testing.T) {
		node := &VectorNode[[]float32]{Vector: []float32{1.0, 2.0}}
		assert.IsType(t, []float32{}, node.Vector)
	})

	t.Run("metadata_is_string_any_map", func(t *testing.T) {
		node := &VectorNode[[]float32]{Metadata: map[string]any{"key": "value"}}
		assert.IsType(t, map[string]any{}, node.Metadata)
	})
}

func TestVectorNode_MetadataTypes(t *testing.T) {
	tests := []struct {
		name     string
		metadata map[string]any
	}{
		{
			name: "string_values",
			metadata: map[string]any{
				"text":        "hello",
				"description": "test vector",
			},
		},
		{
			name: "numeric_values",
			metadata: map[string]any{
				"int":     42,
				"float":   3.14159,
				"int64":   int64(1000000),
				"float32": float32(2.71828),
			},
		},
		{
			name: "boolean_values",
			metadata: map[string]any{
				"active":    true,
				"processed": false,
			},
		},
		{
			name: "array_values",
			metadata: map[string]any{
				"tags":       []string{"go", "vector", "db"},
				"dimensions": []int{128, 256, 512},
			},
		},
		{
			name: "nested_maps",
			metadata: map[string]any{
				"config": map[string]any{
					"version": "1.0",
					"enabled": true,
				},
			},
		},
		{
			name: "mixed_types",
			metadata: map[string]any{
				"name":      "test",
				"count":     100,
				"active":    true,
				"tags":      []string{"test"},
				"timestamp": int64(1234567890),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &VectorNode[[]float32]{
				ID:       "test",
				Vector:   []float32{1.0, 2.0},
				Metadata: tt.metadata,
			}

			require.NotNil(t, node.Metadata)
			assert.Equal(t, tt.metadata, node.Metadata)

			for key, expectedValue := range tt.metadata {
				actualValue, exists := node.Metadata[key]
				require.True(t, exists, "Key %s should exist", key)
				assert.Equal(t, expectedValue, actualValue, "Value for key %s should match", key)
			}
		})
	}
}

func TestVectorNode_VectorDimensions(t *testing.T) {
	tests := []struct {
		name       string
		vector     []float32
		dimensions int
	}{
		{"dimension_2", []float32{1.0, 2.0}, 2},
		{"dimension_3", []float32{1.0, 2.0, 3.0}, 3},
		{"dimension_128", make([]float32, 128), 128},
		{"dimension_384", make([]float32, 384), 384},
		{"dimension_512", make([]float32, 512), 512},
		{"dimension_768", make([]float32, 768), 768},
		{"dimension_1536", make([]float32, 1536), 1536},
		{"dimension_4096", make([]float32, 4096), 4096},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &VectorNode[[]float32]{
				ID:       tt.name,
				Vector:   tt.vector,
				Metadata: nil,
			}

			assert.Len(t, node.Vector, tt.dimensions)
		})
	}
}
