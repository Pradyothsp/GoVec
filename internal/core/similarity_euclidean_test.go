package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Euclidean similarity is 1/(1+distance): 1 for identical vectors, falling
// toward 0 as they move apart.

func TestEuclideanSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 1},
		{"3-4-5 triangle", []float32{0, 0}, []float32{3, 4}, 1.0 / 6},
		{"near", []float32{0, 0}, []float32{1, 0}, 1.0 / 2},
		{"far", []float32{0, 0}, []float32{10, 0}, 1.0 / 11},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: the case's vectors.

			// Act
			ab, errAB := EuclideanSimilarity(tt.a, tt.b)
			ba, errBA := EuclideanSimilarity(tt.b, tt.a)

			// Assert
			require.NoError(t, errAB)
			require.NoError(t, errBA)
			assert.InDelta(t, tt.want, ab, 1e-6)
			assert.Equal(t, ab, ba, "must be symmetric")
		})
	}

	t.Run("dimension mismatch", func(t *testing.T) {
		// Arrange
		a, b := []float32{1, 2}, []float32{1, 2, 3}

		// Act
		_, err := EuclideanSimilarity(a, b)

		// Assert
		assert.Error(t, err)
	})
	t.Run("empty", func(t *testing.T) {
		// Arrange
		empty := []float32{}

		// Act
		_, err := EuclideanSimilarity(empty, empty)

		// Assert
		assert.Error(t, err)
	})
}

func TestEuclideanSimilarityInt8(t *testing.T) {
	tests := []struct {
		name string
		a, b []int8
		want float32
	}{
		{"identical", []int8{127, 63, 31}, []int8{127, 63, 31}, 1},
		{"3-4-5 triangle", []int8{0, 0}, []int8{3, 4}, 1.0 / 6},
		// int64 arithmetic: the difference of two int8s overflows int8.
		{"full range", []int8{-127}, []int8{127}, 1.0 / 255},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: the case's vectors.

			// Act
			got, err := EuclideanSimilarityInt8(tt.a, tt.b)

			// Assert
			require.NoError(t, err)
			assert.InDelta(t, tt.want, got, 1e-6)
		})
	}

	t.Run("dimension mismatch", func(t *testing.T) {
		// Arrange
		a, b := []int8{127, 63}, []int8{127, 63, 31}

		// Act
		_, err := EuclideanSimilarityInt8(a, b)

		// Assert
		assert.Error(t, err)
	})
	t.Run("empty", func(t *testing.T) {
		// Arrange
		empty := []int8{}

		// Act
		_, err := EuclideanSimilarityInt8(empty, empty)

		// Assert
		assert.Error(t, err)
	})
}
