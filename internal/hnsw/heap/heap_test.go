//nolint:revive // Package name matches directory, not stdlib conflict in practice
package heap

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type Int int

func (i Int) Less(j Int) bool {
	return i < j
}

// trickyOrder leaves the maximum outside the last array slot of a min-heap:
// container/heap only guarantees parent <= children, so Max and PopLast must
// search for it rather than take data[len-1].
var trickyOrder = []Int{1, 3, 2, 7, 4, 5, 6}

func randomInts(n int) []Int {
	rng := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // deterministic test data
	values := make([]Int, n)
	for i := range values {
		values[i] = Int(rng.IntN(100))
	}
	return values
}

func TestHeap(t *testing.T) {
	t.Run("Max and PopLast find the true maximum", func(t *testing.T) {
		// Arrange
		h := Heap[Int]{}
		h.Init(nil)
		for _, v := range trickyOrder {
			h.Push(v)
		}

		// Act
		peeked := h.Max()
		popped := h.PopLast()

		// Assert
		assert.Equal(t, Int(7), peeked)
		assert.Equal(t, Int(7), popped)
		remaining := h.Slice()
		slices.Sort(remaining)
		assert.Equal(t, []Int{1, 2, 3, 4, 5, 6}, remaining)
	})

	t.Run("Pop drains in ascending order", func(t *testing.T) {
		// Arrange
		h := Heap[Int]{}
		values := randomInts(50)
		for _, v := range values {
			h.Push(v)
		}

		// Act
		var drained []Int
		for h.Len() > 0 {
			drained = append(drained, h.Pop())
		}

		// Assert
		slices.Sort(values)
		require.Equal(t, values, drained)
	})
}

func TestMaxHeap(t *testing.T) {
	t.Run("Max finds the true maximum", func(t *testing.T) {
		// Arrange
		h := MaxHeap[Int]{}
		h.Init(nil)
		for _, v := range trickyOrder {
			h.Push(v)
		}

		// Act
		got := h.Max()

		// Assert
		assert.Equal(t, Int(7), got)
	})

	t.Run("PopLast drains in descending order", func(t *testing.T) {
		// Arrange
		h := MaxHeap[Int]{}
		values := randomInts(50)
		for _, v := range values {
			h.Push(v)
		}

		// Act
		var drained []Int
		for h.Len() > 0 {
			drained = append(drained, h.PopLast())
		}

		// Assert
		slices.Sort(values)
		slices.Reverse(values)
		require.Equal(t, values, drained)
	})
}
