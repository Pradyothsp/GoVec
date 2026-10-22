package hnsw

import "fmt"

// Key is the graph's key type. Keys must be integers because the visited set
// uses each key as an array index, and non-negative because a negative index
// can't be one. The index layer's keys are internal IDs (uint32) issued from
// 0 upward and never reused, so the arrays stay as long as the largest ID.
type Key interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// visitedSet records which nodes one traversal has already reached. It
// replaces a Go map per search: a map hashed every lookup and grew from empty
// on every layer of every query, about 30% of search time at 10k vectors.
// Here a lookup is one indexed load.
//
// Each slot holds the epoch of the last traversal that visited it, so a new
// traversal starts by bumping the epoch instead of clearing the array; every
// older tag stops matching at once. The array is cleared only when the epoch
// wraps, once every 65,535 traversals.
type visitedSet struct {
	tags  []uint16
	epoch uint16
}

// reset starts a new traversal: nothing counts as visited afterwards.
func (s *visitedSet) reset() {
	s.epoch++
	if s.epoch == 0 {
		// Wrapped: a tag written 65,536 traversals ago would match again.
		clear(s.tags)
		s.epoch = 1
	}
}

// visit marks the node whose key is i and reports whether this traversal
// hadn't reached it yet.
func (s *visitedSet) visit(i int) bool {
	if i < 0 {
		panic(fmt.Sprintf("hnsw: negative key %d", i))
	}
	if i >= len(s.tags) {
		s.grow(i)
	}
	if s.tags[i] == s.epoch {
		return false
	}
	s.tags[i] = s.epoch
	return true
}

// grow extends tags to cover index i, keeping this traversal's marks. The new
// slots are zero, which no epoch equals.
func (s *visitedSet) grow(i int) {
	tags := make([]uint16, max(i+1, 2*len(s.tags)))
	copy(tags, s.tags)
	s.tags = tags
}

// acquireVisited takes a visited set from the graph's pool. Each
// concurrent search or insert-preparing goroutine needs its own: a shared set
// would mix two traversals' marks and make each skip nodes the other reached.
// Return it with releaseVisited.
func (g *Graph[K, V]) acquireVisited() *visitedSet {
	s, ok := g.visitedPool.Get().(*visitedSet)
	if !ok {
		s = &visitedSet{}
	}
	return s
}

// releaseVisited returns s to the pool. The caller must not use it afterwards.
func (g *Graph[K, V]) releaseVisited(s *visitedSet) {
	g.visitedPool.Put(s)
}
