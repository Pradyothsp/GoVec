package hnsw

import (
	"cmp"
	"slices"
)

// layerNode is an internal representation of a node within a specific layer of the HNSW graph.
// It embeds a Node, adding layer-specific details like its neighbor list.
type layerNode[K Key, V VectorType] struct {
	Node[K, V]

	// neighbors stores the direct neighbors of this node within the same layer,
	// up to a bounded capacity (M). A flat slice, not a map: M is small (tens
	// of entries), and a Go map's per-entry bucket overhead dominates memory
	// at HNSW's scale (one map per node, times every node in the graph) far
	// more than a linear scan over a handful of pointers costs in CPU --
	// confirmed via heap profile, see govec-bench STATUS.md.
	neighbors []*layerNode[K, V]

	// precomputed is an opaque, metric-supplied value computed once via
	// Graph.Metric.Precompute when this node is created (on insert or persistence
	// import), and handed back to Graph.Metric.CachedDistance on every future
	// comparison involving this node -- letting the metric skip work that's
	// otherwise redundantly repeated on every single comparison (e.g.
	// cosine's per-vector norm: needed on both sides of every distance()
	// call, but it never changes once the vector is stored). layerNode
	// itself never interprets this value -- see distFunc's doc comment.
	//
	// float64, not float32 or a type derived from V (e.g. int64 for
	// []int8's integer sums): []int8's squared-norm accumulates in int64
	// (see metric.CosineSimilarityInt8) and can exceed float32's 2^24
	// exact-integer ceiling at realistic embedding dimensions (3072 dims x
	// 127^2 ~= 49.5M > 16.78M) -- float32 would silently lose precision
	// there. float64 holds any int64 this codebase will realistically ever
	// produce here exactly (up to 2^53), and that correctness is free:
	// math.Sqrt only operates on float64 anyway, so an int64-cached value
	// would still need this exact conversion before use, just one step
	// later. A per-V cache type would need a third generic type parameter
	// threaded through layerNode/Graph/HNSWIndex for zero benefit over just
	// using float64 everywhere -- not worth the complexity.
	//
	// Left at its zero value, unused, when the configured metric doesn't
	// opt in (Graph.Metric.Precompute == nil) -- e.g. Euclidean, whose current
	// single-pass implementation has no redundant per-comparison norm
	// computation to eliminate in the first place, or any custom-registered
	// metric that doesn't supply Precompute/CachedDistance.
	precomputed float64
}

// indexOfNeighbor returns the index of the neighbor with the given key in
// n.neighbors, or -1 if not present.
func (n *layerNode[K, V]) indexOfNeighbor(key K) int {
	for i, nb := range n.neighbors {
		if nb.Key == key {
			return i
		}
	}

	return -1
}

// removeNeighbor removes the neighbor with the given key, if present.
// No-op if the key isn't found. Order among remaining neighbors is not
// preserved (swap-with-last) -- nothing depends on neighbor slice order,
// search() iterates current.neighbors directly in whatever order it's
// stored, which is already deterministic within one constructed graph.
func (n *layerNode[K, V]) removeNeighbor(key K) {
	if i := n.indexOfNeighbor(key); i >= 0 {
		n.neighbors[i] = n.neighbors[len(n.neighbors)-1]
		n.neighbors = n.neighbors[:len(n.neighbors)-1]
	}
}

// addNeighbor attempts to add a new node as a neighbor to the current layerNode.
// If the current node already has 'm' neighbors (its maximum capacity), the full
// candidate set (existing neighbors + newNode) is re-evaluated with
// selectNeighborsHeuristic, which prefers diverse connections over simply the
// 'm' closest by raw distance -- see selectNeighborsHeuristic's doc comment.
// This can drop more than one existing neighbor, or newNode itself, though in
// practice it's usually exactly one (m+1 candidates in, up to m selected).
// 'dist' compares two stored nodes, using cached per-node values when the
// graph's metric supports it (see distFunc's doc comment).
func (n *layerNode[K, V]) addNeighbor(newNode *layerNode[K, V], m int, dist distFunc[V]) {
	// If newNode is already a neighbor, just update it in place -- matches
	// the map version's overwrite-on-existing-key semantics.
	if i := n.indexOfNeighbor(newNode.Key); i >= 0 {
		n.neighbors[i] = newNode
		return
	}

	if n.neighbors == nil {
		n.neighbors = make([]*layerNode[K, V], 0, m)
	}

	n.neighbors = append(n.neighbors, newNode)
	if len(n.neighbors) <= m {
		return
	}

	// Over capacity: re-run selection over the full candidate set, with
	// distances computed relative to n itself (not whatever query the new
	// node was originally inserted for).
	candidates := make([]searchCandidate[K, V], len(n.neighbors))
	for i, neighbor := range n.neighbors {
		candidates[i] = searchCandidate[K, V]{node: neighbor, dist: dist(neighbor.Value, neighbor.precomputed, n.Value, n.precomputed)}
	}

	selected := selectNeighborsHeuristic(candidates, m, dist)

	// Diff against the pre-selection set to find who got dropped. selected
	// is small (<= m, so <= 16-32 elements), so a linear scan is cheap and
	// allocation-free -- the same reasoning layerNode.neighbors itself is a
	// slice instead of a map for (see that field's doc comment), just
	// applied to this temporary set instead of a permanent field.
	var dropped []*layerNode[K, V]
	for _, neighbor := range n.neighbors {
		kept := false
		for _, s := range selected {
			if s.Key == neighbor.Key {
				kept = true
				break
			}
		}

		if !kept {
			dropped = append(dropped, neighbor)
		}
	}

	n.neighbors = selected
	for _, d := range dropped {
		// Remove the backlink only -- no repair pass. hnswlib's
		// mutuallyConnectNewElement never replenishes on insertion either;
		// evicted links are simply pruned, relying on HNSW's power-law
		// small-world connectivity to keep the graph reachable without
		// active healing on every eviction. replenish() itself is kept for
		// Delete(), where a node's neighbors lose a connection
		// permanently (not just to a redistribution the heuristic already
		// picked a diverse replacement set for) and have no other chance at
		// repair.
		d.removeNeighbor(n.Key)
	}
}

// selectNeighborsHeuristic picks up to m diverse candidates from a pool whose
// distances are already computed relative to a common query point (the new
// node being inserted, or an existing node being re-evaluated -- see the two
// call sites in Add and addNeighbor). This is SELECT-NEIGHBORS-HEURISTIC from
// Malkov & Yashunin, matching hnswlib's actual getNeighborsByHeuristic2
// exactly: no extendCandidates (no pulling in candidates' own neighbors to
// widen the pool) and no keepPrunedConnections (rejected candidates are not
// backfilled if fewer than m survive) -- hnswlib's production implementation
// doesn't use either, despite both appearing in the original paper.
//
// Candidates are visited nearest-to-query first. A candidate is accepted only
// if it is closer to the query than it is to every already-accepted
// candidate; otherwise it's considered redundant (already well-represented by
// a closer, already-selected neighbor) and dropped. This favors connections
// that reach new, useful regions of the graph over several connections
// clustered in the same direction, which raw "m closest by distance"
// selection (this function's caller before this change) is prone to.
//
// If there are fewer than m candidates, all of them are returned unpruned --
// nothing to select from. Note this can still return fewer than m even when
// there are m or more candidates: the heuristic has no backfill step, so
// highly redundant candidate sets can legitimately prune below m.
func selectNeighborsHeuristic[K Key, V VectorType](candidates []searchCandidate[K, V], m int, dist distFunc[V]) []*layerNode[K, V] {
	if len(candidates) < m {
		selected := make([]*layerNode[K, V], len(candidates))
		for i, c := range candidates {
			selected[i] = c.node
		}

		return selected
	}

	sorted := slices.Clone(candidates)
	slices.SortFunc(sorted, func(a, b searchCandidate[K, V]) int {
		if c := cmp.Compare(a.dist, b.dist); c != 0 {
			return c
		}

		return cmp.Compare(a.node.Key, b.node.Key) // deterministic tie-break
	})

	selected := make([]*layerNode[K, V], 0, m)
	for _, candidate := range sorted {
		if len(selected) >= m {
			break
		}

		good := true
		for _, kept := range selected {
			if dist(kept.Value, kept.precomputed, candidate.node.Value, candidate.node.precomputed) < candidate.dist {
				good = false
				break
			}
		}

		if good {
			selected = append(selected, candidate.node)
		}
	}

	return selected
}

// replenish attempts to restore connectivity for a node that might have lost neighbors,
// for instance, after a neighbor deletion. It gathers 2-hop candidates (its
// current neighbors' neighbors) and adds them back nearest-first, up to 'm' --
// not the first candidates encountered in traversal order. This matters more
// than it might look: replenish's candidates compete with the heuristic's own
// carefully-chosen diverse connections, so adding low-quality (merely "first
// found") candidates measurably diluted graph quality in practice.
func (n *layerNode[K, V]) replenish(m int, dist distFunc[V]) {
	if len(n.neighbors) >= m {
		return
	}

	seen := make(map[K]struct{}, len(n.neighbors))
	for _, neighbor := range n.neighbors {
		seen[neighbor.Key] = struct{}{}
	}

	seen[n.Key] = struct{}{}

	// Cap how many 2-hop candidates get gathered (and therefore sorted) --
	// at high M (e.g. the base layer's 2*M), a node's neighbors' neighbors
	// can number in the thousands, and this is a linear-scan/no-index repair
	// path, not a real graph search. A handful of candidates over the actual
	// shortfall is plenty to pick a good nearest one from; unbounded
	// gathering makes replenish (called on every over-capacity eviction, not
	// just deletes) the dominant construction cost at scale.
	need := m - len(n.neighbors)
	maxCandidates := 4 * need
	if maxCandidates < 16 {
		maxCandidates = 16
	}

	var candidates []searchCandidate[K, V]
	for _, neighbor := range n.neighbors {
		for _, candidate := range neighbor.neighbors {
			if len(candidates) >= maxCandidates {
				break
			}

			if _, dup := seen[candidate.Key]; dup {
				continue
			}

			seen[candidate.Key] = struct{}{}
			candidates = append(candidates, searchCandidate[K, V]{node: candidate, dist: dist(candidate.Value, candidate.precomputed, n.Value, n.precomputed)})
		}

		if len(candidates) >= maxCandidates {
			break
		}
	}

	slices.SortFunc(candidates, func(a, b searchCandidate[K, V]) int {
		return cmp.Compare(a.dist, b.dist)
	})

	for _, c := range candidates {
		n.addNeighbor(c.node, m, dist)
		if len(n.neighbors) >= m {
			return
		}
	}
}
