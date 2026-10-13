package metric

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Cosine similarity is the cosine of the angle between two vectors: 1 for the
// same direction, whatever the lengths, 0 at right angles, -1 opposite. The
// unrolled loop is checked against the float64 formula on every length in
// similarity_reference_test.go.
func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, -2, 3}, []float32{1, -2, 3}, 1},
		{"same direction, different length", []float32{1, 1}, []float32{2, 2}, 1},
		{"opposite", []float32{1, 2, 3}, []float32{-1, -2, -3}, -1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"45 degrees", []float32{1, 0, 0}, []float32{1, 1, 0}, float32(1 / math.Sqrt2)},
		{"tiny values", []float32{0.0001, 0.0002}, []float32{0.0001, 0.0002}, 1},
		// A zero vector has no direction, so no meaningful similarity: 0, not NaN.
		{"zero vector", []float32{0, 0, 0}, []float32{1, 2, 3}, 0},
		{"both zero", []float32{0, 0}, []float32{0, 0}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			ab, errAB := CosineSimilarity(tt.a, tt.b)
			ba, errBA := CosineSimilarity(tt.b, tt.a)

			// Assert
			require.NoError(t, errAB)
			require.NoError(t, errBA)
			assert.InDelta(t, tt.want, ab, 1e-5)
			assert.Equal(t, ab, ba, "must be symmetric")
		})
	}

	t.Run("dimension mismatch", func(t *testing.T) {
		// Act
		_, err := CosineSimilarity([]float32{1, 2}, []float32{1, 2, 3})

		// Assert
		assert.EqualError(t, err, "vector dimensions mismatch")
	})
	t.Run("empty", func(t *testing.T) {
		// Act
		_, err := CosineSimilarity([]float32{}, []float32{})

		// Assert
		assert.Error(t, err)
	})
}
