package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestVectorNode_MetadataTypes validates that VectorNode can store various metadata types
// correctly. This is important for JSON serialization/deserialization edge cases where
// different value types (strings, numbers, booleans, arrays, nested maps) need to be
// preserved through persistence and WAL replay operations.
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
				ExternalID: "test",
				Vector:     []float32{1.0, 2.0},
				Metadata:   tt.metadata,
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
