package heap

import "container/heap"

// Lessable is an interface that allows a type to be compared to another of the same type.
// It is used to define the order of elements in the heap.
type Lessable[T any] interface {
	Less(T) bool
}

// innerHeap is a type that represents the heap data structure.
// it implements the std heap interface.
type innerHeap[T Lessable[T]] struct {
	data []T
}

func (h *innerHeap[T]) Len() int {
	return len(h.data)
}

func (h *innerHeap[T]) Less(i, j int) bool {
	return h.data[i].Less(h.data[j])
}

func (h *innerHeap[T]) Swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *innerHeap[T]) Push(x interface{}) {
	//nolint:errcheck // Type assertion safe: heap only contains T
	h.data = append(h.data, x.(T))
}

func (h *innerHeap[T]) Pop() interface{} {
	n := len(h.data)
	x := h.data[n-1]
	h.data = h.data[:n-1]
	return x
}

// Heap represents the heap data structure using a flat array to store the elements.
// It is a wrapper around the standard library's heap.
type Heap[T Lessable[T]] struct {
	inner innerHeap[T]
}

// Init establishes the heap invariants required by the other routines in this package.
// Init is idempotent with respect to the heap invariants
// and may be called whenever the heap invariants may have been invalidated.
// The complexity is O(n) where n = h.Len().
func (h *Heap[T]) Init(d []T) {
	h.inner.data = d
	heap.Init(&h.inner)
}

// Len returns the number of elements in the heap.
func (h *Heap[T]) Len() int {
	return h.inner.Len()
}

// Push pushes the element x onto the heap.
// The complexity is O(log n) where n = h.Len().
func (h *Heap[T]) Push(x T) {
	heap.Push(&h.inner, x)
}

// Pop removes and returns the minimum element (according to Less) from the heap.
// The complexity is O(log n) where n = h.Len().
// Pop is equivalent to Remove(h, 0).
func (h *Heap[T]) Pop() T {
	//nolint:errcheck // Type assertion safe: heap only contains T
	return heap.Pop(&h.inner).(T)
}

// PopLast removes and returns the maximum (worst) element from the heap.
func (h *Heap[T]) PopLast() T {
	return h.Remove(h.maxIndex())
}

// Remove removes and returns the element at index i from the heap.
// The complexity is O(log n) where n = h.Len().
func (h *Heap[T]) Remove(i int) T {
	//nolint:errcheck // Type assertion safe: heap only contains T
	return heap.Remove(&h.inner, i).(T)
}

// Min returns the minimum element in the heap.
func (h *Heap[T]) Min() T {
	return h.inner.data[0]
}

// Max returns the maximum element in the heap.
func (h *Heap[T]) Max() T {
	return h.inner.data[h.maxIndex()]
}

// maxIndex returns the index of the maximum element in the heap's flat array.
// A binary min-heap only guarantees parent <= children, so the maximum is not
// necessarily the last element -- it must be found by scanning every element.
func (h *Heap[T]) maxIndex() int {
	maxIdx := 0
	for i := 1; i < h.inner.Len(); i++ {
		if h.inner.data[maxIdx].Less(h.inner.data[i]) {
			maxIdx = i
		}
	}
	return maxIdx
}

// Slice returns a copy of the heap's underlying data as a slice.
func (h *Heap[T]) Slice() []T {
	return h.inner.data
}

// maxInnerHeap is the container/heap adapter for MaxHeap. Same Lessable[T]
// contract as innerHeap, but the comparison is inverted so the heap
// invariant puts the maximum element (by T.Less) at the root instead of the
// minimum.
type maxInnerHeap[T Lessable[T]] struct {
	data []T
}

func (h *maxInnerHeap[T]) Len() int {
	return len(h.data)
}

func (h *maxInnerHeap[T]) Less(i, j int) bool {
	// Inverted vs innerHeap.Less: data[j].Less(data[i]) is true when
	// data[i] is the larger of the two, so it's the one that belongs
	// closer to the root of a max-heap.
	return h.data[j].Less(h.data[i])
}

func (h *maxInnerHeap[T]) Swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *maxInnerHeap[T]) Push(x interface{}) {
	//nolint:errcheck // Type assertion safe: heap only contains T
	h.data = append(h.data, x.(T))
}

func (h *maxInnerHeap[T]) Pop() interface{} {
	n := len(h.data)
	x := h.data[n-1]
	h.data = h.data[:n-1]
	return x
}

// MaxHeap is a binary max-heap over T, ordered by T.Less: the element that
// no other element is Less than sits at the root. It exists for HNSW's
// bounded result set, where the hot-path operation is "what's the worst
// candidate kept so far, and can it be evicted" -- O(1) peek and O(log n)
// eviction here, versus Heap's O(n) Max()/PopLast() (Heap stays a min-heap,
// which is what candidate-set traversal actually wants -- see graph.go's
// search()).
type MaxHeap[T Lessable[T]] struct {
	inner maxInnerHeap[T]
}

// Init establishes the heap invariants required by the other routines in this package.
// The complexity is O(n) where n = h.Len().
func (h *MaxHeap[T]) Init(d []T) {
	h.inner.data = d
	heap.Init(&h.inner)
}

// Len returns the number of elements in the heap.
func (h *MaxHeap[T]) Len() int {
	return h.inner.Len()
}

// Push pushes the element x onto the heap.
// The complexity is O(log n) where n = h.Len().
func (h *MaxHeap[T]) Push(x T) {
	heap.Push(&h.inner, x)
}

// Max returns the maximum element in the heap without removing it.
// O(1): the max-heap invariant guarantees it's always the root.
func (h *MaxHeap[T]) Max() T {
	return h.inner.data[0]
}

// PopLast removes and returns the maximum element from the heap.
// The complexity is O(log n) where n = h.Len(), since the maximum is
// always the root under the max-heap invariant.
func (h *MaxHeap[T]) PopLast() T {
	//nolint:errcheck // Type assertion safe: heap only contains T
	return heap.Pop(&h.inner).(T)
}

// Slice returns a copy of the heap's underlying data as a slice.
func (h *MaxHeap[T]) Slice() []T {
	return h.inner.data
}
