package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSparseDotProduct_EmptyVectors(t *testing.T) {
	emptyA := SparseVector{}
	emptyB := SparseVector{}

	result := SparseDotProduct(emptyA, emptyB)
	assert.Equal(t, float32(0.0), result, "Dot product of two empty vectors should be 0")
}

func TestSparseDotProduct_OneEmptyVector(t *testing.T) {
	empty := SparseVector{}
	nonEmpty := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{1.0, 2.0, 3.0},
	}

	result1 := SparseDotProduct(empty, nonEmpty)
	assert.Equal(t, float32(0.0), result1, "Dot product with empty first vector should be 0")

	result2 := SparseDotProduct(nonEmpty, empty)
	assert.Equal(t, float32(0.0), result2, "Dot product with empty second vector should be 0")
}

func TestSparseDotProduct_NoOverlap(t *testing.T) {
	// No common indices
	a := SparseVector{
		Indices: []uint32{1, 3, 5},
		Values:  []float32{1.0, 2.0, 3.0},
	}
	b := SparseVector{
		Indices: []uint32{2, 4, 6},
		Values:  []float32{4.0, 5.0, 6.0},
	}

	result := SparseDotProduct(a, b)
	assert.Equal(t, float32(0.0), result, "Dot product with no overlapping indices should be 0")
}

func TestSparseDotProduct_CompleteOverlap(t *testing.T) {
	// All indices match
	a := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{2.0, 3.0, 4.0},
	}
	b := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{1.0, 2.0, 3.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (2*1) + (3*2) + (4*3) = 2 + 6 + 12 = 20
	expected := float32(2.0*1.0 + 3.0*2.0 + 4.0*3.0)
	assert.Equal(t, expected, result, "Dot product should sum all matching indices")
}

func TestSparseDotProduct_PartialOverlap(t *testing.T) {
	// Some indices match, some don't
	a := SparseVector{
		Indices: []uint32{1, 3, 5, 7},
		Values:  []float32{1.0, 2.0, 3.0, 4.0},
	}
	b := SparseVector{
		Indices: []uint32{2, 3, 6, 7},
		Values:  []float32{5.0, 6.0, 7.0, 8.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (2*6) + (4*8) = 12 + 32 = 44
	// Only indices 3 and 7 match
	expected := float32(2.0*6.0 + 4.0*8.0)
	assert.Equal(t, expected, result, "Dot product should only sum matching indices")
}

func TestSparseDotProduct_SingleElement(t *testing.T) {
	// Single element vectors
	a := SparseVector{
		Indices: []uint32{5},
		Values:  []float32{3.0},
	}
	b := SparseVector{
		Indices: []uint32{5},
		Values:  []float32{4.0},
	}

	result := SparseDotProduct(a, b)
	assert.Equal(t, float32(12.0), result, "Dot product of single matching elements")
}

func TestSparseDotProduct_DifferentLengths(t *testing.T) {
	// Different length vectors with some overlap
	a := SparseVector{
		Indices: []uint32{1, 2, 3, 4, 5},
		Values:  []float32{1.0, 2.0, 3.0, 4.0, 5.0},
	}
	b := SparseVector{
		Indices: []uint32{3, 5},
		Values:  []float32{10.0, 20.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (3*10) + (5*20) = 30 + 100 = 130
	expected := float32(3.0*10.0 + 5.0*20.0)
	assert.Equal(t, expected, result, "Dot product should work with different length vectors")
}

func TestSparseDotProduct_NegativeValues(t *testing.T) {
	// Test with negative values
	a := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{-1.0, 2.0, -3.0},
	}
	b := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{2.0, -3.0, 4.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (-1*2) + (2*-3) + (-3*4) = -2 + -6 + -12 = -20
	expected := float32(-1.0*2.0 + 2.0*-3.0 + -3.0*4.0)
	assert.Equal(t, expected, result, "Dot product should handle negative values correctly")
}

func TestSparseDotProduct_ZeroValues(t *testing.T) {
	// Test with zero values (although sparse vectors shouldn't have these in practice)
	a := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{0.0, 2.0, 3.0},
	}
	b := SparseVector{
		Indices: []uint32{1, 5, 10},
		Values:  []float32{1.0, 0.0, 4.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (0*1) + (2*0) + (3*4) = 0 + 0 + 12 = 12
	expected := float32(0.0*1.0 + 2.0*0.0 + 3.0*4.0)
	assert.Equal(t, expected, result, "Dot product should handle zero values")
}

func TestSparseDotProduct_LargeIndices(t *testing.T) {
	// Test with large index values (simulating large vocabulary)
	a := SparseVector{
		Indices: []uint32{100, 1000, 10000},
		Values:  []float32{1.5, 2.5, 3.5},
	}
	b := SparseVector{
		Indices: []uint32{100, 1000, 10000},
		Values:  []float32{2.0, 3.0, 4.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (1.5*2.0) + (2.5*3.0) + (3.5*4.0) = 3.0 + 7.5 + 14.0 = 24.5
	expected := float32(1.5*2.0 + 2.5*3.0 + 3.5*4.0)
	assert.Equal(t, expected, result, "Dot product should work with large indices")
}

func TestSparseDotProduct_BM25Example(t *testing.T) {
	// Realistic BM25 example from documentation
	// IMPORTANT: Indices must be sorted in ascending order!
	query := SparseVector{
		Indices: []uint32{12, 452}, // "battery", "fix" (sorted by token ID)
		Values:  []float32{0.9, 1.8},
	}
	doc := SparseVector{
		Indices: []uint32{12, 452, 8911}, // "battery", "fix", "iphone" (sorted)
		Values:  []float32{0.8, 1.5, 2.1},
	}

	result := SparseDotProduct(query, doc)
	// Expected: (0.9*0.8) + (1.8*1.5) = 0.72 + 2.7 = 3.42
	expected := float32(0.9*0.8 + 1.8*1.5)
	assert.InDelta(t, expected, result, 0.0001, "BM25 example should compute correct score")
}

func TestSparseDotProduct_Commutative(t *testing.T) {
	// Dot product should be commutative: a·b = b·a
	a := SparseVector{
		Indices: []uint32{1, 3, 5, 7},
		Values:  []float32{1.0, 2.0, 3.0, 4.0},
	}
	b := SparseVector{
		Indices: []uint32{2, 3, 6, 7},
		Values:  []float32{5.0, 6.0, 7.0, 8.0},
	}

	result1 := SparseDotProduct(a, b)
	result2 := SparseDotProduct(b, a)

	assert.Equal(t, result1, result2, "Dot product should be commutative")
}

func TestSparseDotProduct_FirstVectorLonger(t *testing.T) {
	// Ensure algorithm works when first vector is longer
	a := SparseVector{
		Indices: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		Values:  []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	}
	b := SparseVector{
		Indices: []uint32{3, 7},
		Values:  []float32{5.0, 10.0},
	}

	result := SparseDotProduct(a, b)
	// Expected: (1*5) + (1*10) = 15
	expected := float32(15.0)
	assert.Equal(t, expected, result, "Should handle first vector being longer")
}

func TestSparseDotProduct_SecondVectorLonger(t *testing.T) {
	// Ensure algorithm works when second vector is longer
	a := SparseVector{
		Indices: []uint32{3, 7},
		Values:  []float32{5.0, 10.0},
	}
	b := SparseVector{
		Indices: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		Values:  []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	}

	result := SparseDotProduct(a, b)
	// Expected: (5*1) + (10*1) = 15
	expected := float32(15.0)
	assert.Equal(t, expected, result, "Should handle second vector being longer")
}
