package hnsw

import (
	"container/heap"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// referenceHeap is container/heap over the same candidates: the queues must
// pop in exactly its order, ties included, or searches would change results.
type referenceHeap struct {
	items    []searchCandidate[uint32, []float32]
	farthest bool
}

func (h *referenceHeap) Len() int { return len(h.items) }

func (h *referenceHeap) Less(i, j int) bool {
	if h.farthest {
		return h.items[j].dist < h.items[i].dist
	}

	return h.items[i].dist < h.items[j].dist
}

func (h *referenceHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *referenceHeap) Push(x any) {
	h.items = append(h.items, x.(searchCandidate[uint32, []float32]))
}

func (h *referenceHeap) Pop() any {
	n := len(h.items) - 1
	x := h.items[n]
	h.items = h.items[:n]

	return x
}

// Pushes and pops interleaved, with distances drawn from a few values so most
// of them tie: each queue must hand back the same node as container/heap at
// every pop.
func TestQueues_PopInContainerHeapOrder(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		farthest bool
	}{
		{"nearestFirst", false},
		{"farthestFirst", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Arrange
			rng := rand.New(rand.NewSource(1))
			ref := &referenceHeap{farthest: tt.farthest}
			var near nearestFirst[uint32, []float32]
			var far farthestFirst[uint32, []float32]

			var got, want []uint32

			// Act
			for i := range 2000 {
				c := searchCandidate[uint32, []float32]{
					node: &layerNode[uint32, []float32]{Node: Node[uint32, []float32]{Key: uint32(i)}},
					dist: float32(rng.Intn(8)),
				}

				heap.Push(ref, c)
				near.push(c)
				far.push(c)

				if rng.Intn(3) == 0 {
					want = append(want, heap.Pop(ref).(searchCandidate[uint32, []float32]).node.Key)

					if tt.farthest {
						got = append(got, far.popMax().node.Key)
					} else {
						got = append(got, near.pop().node.Key)
					}
				}
			}

			for ref.Len() > 0 {
				want = append(want, heap.Pop(ref).(searchCandidate[uint32, []float32]).node.Key)

				if tt.farthest {
					got = append(got, far.popMax().node.Key)
				} else {
					got = append(got, near.pop().node.Key)
				}
			}

			// Assert
			assert.Equal(t, want, got)
		})
	}
}

// min and max peek at the top without removing it.
func TestQueues_PeekLeavesTheTop(t *testing.T) {
	t.Parallel()

	// Arrange
	var near nearestFirst[uint32, []float32]
	var far farthestFirst[uint32, []float32]

	for _, d := range []float32{3, 1, 2} {
		c := searchCandidate[uint32, []float32]{dist: d}
		near.push(c)
		far.push(c)
	}

	// Act
	nearest, farthest := near.min(), far.max()

	// Assert
	assert.Equal(t, float32(1), nearest.dist)
	assert.Equal(t, float32(3), farthest.dist)
	assert.Equal(t, 3, near.len())
	assert.Equal(t, 3, far.len())
}
