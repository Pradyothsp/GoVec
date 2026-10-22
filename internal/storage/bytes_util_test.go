package storage

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The helpers reinterpret memory without copying, so a vector must come back
// bit for bit, and the bytes must alias the vector rather than copy it.
func TestVectorBytes_RoundTripWithoutCopying(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		// Arrange
		vec := []float32{1.5, -2, 0, 3.25e-7}

		// Act
		b := VectorToBytes(vec)
		back := BytesToVector[[]float32](b)

		// Assert
		assert.Len(t, b, 16)
		assert.Equal(t, vec, back)

		vec[0] = 9
		assert.Equal(t, float32(9), back[0], "the bytes must alias the vector, not copy it")
	})

	t.Run("int8", func(t *testing.T) {
		// Arrange
		vec := []int8{-128, -1, 0, 127}

		// Act
		b := VectorToBytes(vec)
		back := BytesToVector[[]int8](b)

		// Assert
		assert.Len(t, b, 4)
		assert.Equal(t, vec, back)
	})

	t.Run("empty", func(t *testing.T) {
		// Act
		b := VectorToBytes([]float32{})
		back := BytesToVector[[]float32](nil)

		// Assert
		assert.Nil(t, b)
		assert.Nil(t, back)
	})
}
