package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCosineSimilarityInt8(t *testing.T) {
	// Arrange: the cases below, sharing one 1536-d vector.
	large := make([]int8, 1536)
	for i := range large {
		large[i] = int8((i % 127) - 63)
	}

	tests := []struct {
		name string
		a, b []int8
		want float32
	}{
		{"identical", []int8{127, 63, 31}, []int8{127, 63, 31}, 1},
		{"orthogonal", []int8{127, 0, 0}, []int8{0, 127, 0}, 0},
		{"opposite", []int8{127, 63, 31}, []int8{-127, -63, -31}, -1},
		{"similar", []int8{127, 100, 50}, []int8{120, 95, 48}, 0.9999},
		{"45 degrees", []int8{127, 0, 0}, []int8{90, 90, 0}, 0.7071},
		// No direction, so no meaningful similarity.
		{"zero vector", []int8{0, 0, 0}, []int8{127, 63, 31}, 0},
		// int64 sums: 1536 squared int8s overflow anything narrower.
		{"1536 dimensions", large, large, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			got, err := CosineSimilarityInt8(tt.a, tt.b)

			// Assert
			require.NoError(t, err)
			assert.InDelta(t, tt.want, got, 1e-3)
		})
	}

	t.Run("dimension mismatch", func(t *testing.T) {
		// Arrange
		a, b := []int8{127, 63}, []int8{127, 63, 31}

		// Act
		_, err := CosineSimilarityInt8(a, b)

		// Assert
		assert.Error(t, err)
	})
	t.Run("empty", func(t *testing.T) {
		// Arrange
		empty := []int8{}

		// Act
		_, err := CosineSimilarityInt8(empty, empty)

		// Assert
		assert.Error(t, err)
	})
}

// Quantizing to int8 costs some precision, but not the ranking signal.
func TestCosineSimilarityInt8_CloseToFloat32(t *testing.T) {
	// Arrange
	vecA := []float32{0.5, 0.3, 0.8, -0.2}
	vecB := []float32{0.6, 0.2, 0.7, -0.1}

	// Act
	scoreFloat, errFloat := CosineSimilarity(vecA, vecB)
	scoreInt8, errInt8 := CosineSimilarityInt8(QuantizeVector(vecA), QuantizeVector(vecB))

	// Assert
	require.NoError(t, errFloat)
	require.NoError(t, errInt8)
	assert.InDelta(t, scoreFloat, scoreInt8, 0.1)
}

func BenchmarkCosineSimilarityInt8(b *testing.B) {
	vec1 := make([]int8, 1536)
	vec2 := make([]int8, 1536)
	for i := range vec1 {
		vec1[i] = int8(i % 127)
		vec2[i] = int8((i + 1) % 127)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = CosineSimilarityInt8(vec1, vec2)
	}
}

func BenchmarkCosineSimilarityFloat32(b *testing.B) {
	vec1 := make([]float32, 1536)
	vec2 := make([]float32, 1536)
	for i := range vec1 {
		vec1[i] = float32(i) / 1536.0
		vec2[i] = float32(i+1) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = CosineSimilarity(vec1, vec2)
	}
}
