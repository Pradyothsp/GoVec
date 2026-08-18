//nolint:revive // Package name matches directory, not stdlib conflict in practice
package heap

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

type Int int

func (i Int) Less(j Int) bool {
	return i < j
}

func TestHeap_Max_ReturnsTrueMaximum(t *testing.T) {
	h := Heap[Int]{}
	h.Init(nil)

	// Pushed in an order that produces a min-heap array where the true
	// maximum does not land in the last array slot (container/heap only
	// guarantees parent <= children, not that data[len-1] is the max).
	for _, v := range []Int{1, 3, 2, 7, 4, 5, 6} {
		h.Push(v)
	}

	require.Equal(t, Int(7), h.Max())
}

func TestHeap_PopLast_RemovesTrueMaximum(t *testing.T) {
	h := Heap[Int]{}
	h.Init(nil)

	for _, v := range []Int{1, 3, 2, 7, 4, 5, 6} {
		h.Push(v)
	}

	got := h.PopLast()

	require.Equal(t, Int(7), got)
	require.Equal(t, 6, h.Len())

	var remaining []Int
	for h.Len() > 0 {
		remaining = append(remaining, h.Pop())
	}
	require.ElementsMatch(t, []Int{1, 2, 3, 4, 5, 6}, remaining)
}

func TestHeap(t *testing.T) {
	h := Heap[Int]{}

	for i := 0; i < 20; i++ {
		h.Push(Int(rand.Int() % 100))
	}

	require.Equal(t, 20, h.Len())

	var inOrder []Int
	for h.Len() > 0 {
		inOrder = append(inOrder, h.Pop())
	}

	if !slices.IsSorted(inOrder) {
		t.Errorf("Heap did not return sorted elements: %+v", inOrder)
	}
}

func TestMaxHeap_Max_ReturnsTrueMaximum(t *testing.T) {
	h := MaxHeap[Int]{}
	h.Init(nil)

	// Pushed in an order that would leave the true maximum outside index 0
	// under a naive "last pushed" or "any leaf" assumption -- container/heap
	// only guarantees the root beats its children, which Init/Push must
	// still deliver correctly for every push order.
	for _, v := range []Int{1, 3, 2, 7, 4, 5, 6} {
		h.Push(v)
	}

	require.Equal(t, Int(7), h.Max())
}

func TestMaxHeap_PopLast_RemovesTrueMaximum(t *testing.T) {
	h := MaxHeap[Int]{}
	h.Init(nil)

	for _, v := range []Int{1, 3, 2, 7, 4, 5, 6} {
		h.Push(v)
	}

	got := h.PopLast()

	require.Equal(t, Int(7), got)
	require.Equal(t, 6, h.Len())

	// Popping the max repeatedly should drain the heap in descending order --
	// the max-heap analogue of TestHeap's ascending-order min-heap check.
	var descending []Int
	for h.Len() > 0 {
		descending = append(descending, h.PopLast())
	}
	require.Equal(t, []Int{6, 5, 4, 3, 2, 1}, descending)
}

func TestMaxHeap(t *testing.T) {
	h := MaxHeap[Int]{}

	for i := 0; i < 20; i++ {
		h.Push(Int(rand.Int() % 100))
	}

	require.Equal(t, 20, h.Len())

	var descending []Int
	for h.Len() > 0 {
		descending = append(descending, h.PopLast())
	}

	slices.Reverse(descending)
	if !slices.IsSorted(descending) {
		t.Errorf("MaxHeap did not return elements in descending order: %+v", descending)
	}
}
