package hnsw

// The two priority queues a layer search keeps: the candidates still to
// expand, nearest first, and the best results so far, worst first so the one
// to evict is always on top.
//
// They are binary heaps on a typed slice, comparing dist directly. They
// replace a generic wrapper around container/heap, whose Push(any) and Pop()
// any boxed every candidate into a new heap object: over 95% of a search's
// allocations. The sift steps are container/heap's, unchanged, so equal
// distances still break the same way and searches return the same results.

// nearestFirst is a min-heap of candidates by distance.
type nearestFirst[K Key, V VectorType] struct {
	items []searchCandidate[K, V]
}

func (q *nearestFirst[K, V]) len() int {
	return len(q.items)
}

// min returns the nearest candidate without removing it.
func (q *nearestFirst[K, V]) min() searchCandidate[K, V] {
	return q.items[0]
}

func (q *nearestFirst[K, V]) push(c searchCandidate[K, V]) {
	q.items = append(q.items, c)
	q.up(len(q.items) - 1)
}

// pop removes and returns the nearest candidate.
func (q *nearestFirst[K, V]) pop() searchCandidate[K, V] {
	n := len(q.items) - 1
	q.items[0], q.items[n] = q.items[n], q.items[0]
	q.down(0, n)

	c := q.items[n]
	q.items = q.items[:n]

	return c
}

func (q *nearestFirst[K, V]) less(i, j int) bool {
	return q.items[i].dist < q.items[j].dist
}

func (q *nearestFirst[K, V]) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !q.less(j, i) {
			break
		}

		q.items[i], q.items[j] = q.items[j], q.items[i]
		j = i
	}
}

func (q *nearestFirst[K, V]) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}

		j := j1 // left child
		if j2 := j1 + 1; j2 < n && q.less(j2, j1) {
			j = j2 // right child
		}

		if !q.less(j, i) {
			break
		}

		q.items[i], q.items[j] = q.items[j], q.items[i]
		i = j
	}
}

// farthestFirst is a max-heap of candidates by distance.
type farthestFirst[K Key, V VectorType] struct {
	items []searchCandidate[K, V]
}

func (q *farthestFirst[K, V]) len() int {
	return len(q.items)
}

// max returns the farthest candidate without removing it.
func (q *farthestFirst[K, V]) max() searchCandidate[K, V] {
	return q.items[0]
}

func (q *farthestFirst[K, V]) push(c searchCandidate[K, V]) {
	q.items = append(q.items, c)
	q.up(len(q.items) - 1)
}

// popMax removes and returns the farthest candidate.
func (q *farthestFirst[K, V]) popMax() searchCandidate[K, V] {
	n := len(q.items) - 1
	q.items[0], q.items[n] = q.items[n], q.items[0]
	q.down(0, n)

	c := q.items[n]
	q.items = q.items[:n]

	return c
}

func (q *farthestFirst[K, V]) less(i, j int) bool {
	return q.items[j].dist < q.items[i].dist
}

func (q *farthestFirst[K, V]) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !q.less(j, i) {
			break
		}

		q.items[i], q.items[j] = q.items[j], q.items[i]
		j = i
	}
}

func (q *farthestFirst[K, V]) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}

		j := j1 // left child
		if j2 := j1 + 1; j2 < n && q.less(j2, j1) {
			j = j2 // right child
		}

		if !q.less(j, i) {
			break
		}

		q.items[i], q.items[j] = q.items[j], q.items[i]
		i = j
	}
}
