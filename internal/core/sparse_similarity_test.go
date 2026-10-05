package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func sv(indices []uint32, values []float32) SparseVector {
	return SparseVector{Indices: indices, Values: values}
}

// Every case runs with the arguments both ways round: the dot product is
// commutative, and the two-pointer merge must finish correctly whichever
// vector runs out first.
func TestSparseDotProduct(t *testing.T) {
	tests := []struct {
		name string
		a, b SparseVector
		want float32
	}{
		{"both empty", SparseVector{}, SparseVector{}, 0},
		{"one empty", SparseVector{}, sv([]uint32{1, 5, 10}, []float32{1, 2, 3}), 0},
		{"no overlap", sv([]uint32{1, 3, 5}, []float32{1, 2, 3}), sv([]uint32{2, 4, 6}, []float32{4, 5, 6}), 0},
		{"complete overlap", sv([]uint32{1, 5, 10}, []float32{2, 3, 4}), sv([]uint32{1, 5, 10}, []float32{1, 2, 3}), 2*1 + 3*2 + 4*3},
		{"partial overlap", sv([]uint32{1, 3, 5, 7}, []float32{1, 2, 3, 4}), sv([]uint32{2, 3, 6, 7}, []float32{5, 6, 7, 8}), 2*6 + 4*8},
		{"different lengths", sv([]uint32{1, 2, 3, 4, 5}, []float32{1, 2, 3, 4, 5}), sv([]uint32{3, 5}, []float32{10, 20}), 3*10 + 5*20},
		{"one vector much longer", sv([]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}), sv([]uint32{3, 7}, []float32{5, 10}), 15},
		{"negative values", sv([]uint32{1, 5, 10}, []float32{-1, 2, -3}), sv([]uint32{1, 5, 10}, []float32{2, -3, 4}), -1*2 + 2*-3 + -3*4},
		{"large token IDs", sv([]uint32{100, 1000, 10000}, []float32{1.5, 2.5, 3.5}), sv([]uint32{100, 1000, 10000}, []float32{2, 3, 4}), 1.5*2 + 2.5*3 + 3.5*4},
		// A realistic BM25 query against a document ("battery", "fix", "iphone").
		{"bm25 example", sv([]uint32{12, 452}, []float32{0.9, 1.8}), sv([]uint32{12, 452, 8911}, []float32{0.8, 1.5, 2.1}), 0.9*0.8 + 1.8*1.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: the case's vectors.

			// Act
			ab := SparseDotProduct(tt.a, tt.b)
			ba := SparseDotProduct(tt.b, tt.a)

			// Assert
			assert.InDelta(t, tt.want, ab, 1e-5)
			assert.InDelta(t, tt.want, ba, 1e-5, "must be commutative")
		})
	}
}
