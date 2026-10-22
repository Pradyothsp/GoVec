package hnsw

import (
	"slices"

	"github.com/Pradyothsp/govec/internal/hnsw/heap"
)

// searchCandidate represents a node considered during the HNSW search process,
// paired with its calculated distance to the query vector.
type searchCandidate[K Key, V VectorType] struct {
	node *layerNode[K, V]
	dist float32
}

// Less implements the heap.Interface for searchCandidate, ordering by distance.
// Lower distance means higher priority in the min-heap.
func (s searchCandidate[K, V]) Less(o searchCandidate[K, V]) bool {
	return s.dist < o.dist
}

// search performs a greedy search within the current layer to find the 'k' closest
// nodes to the target vector. It explores nodes by iteratively expanding from
// the current best candidates, using 'efSearch' as a parameter to control
// the search's breadth and accuracy.
func (n *layerNode[K, V]) search(
	// k is the desired number of closest candidates to return.
	k int,
	// efSearch (exploration factor search) determines the maximum number of candidates
	// considered during the search. Higher values increase accuracy at the cost of speed.
	efSearch int,
	target V,
	distance distFunc[V],
	// targetCache is target's precomputed value (see metric.Metric's
	// Precompute), computed once by the caller -- target has no layerNode of
	// its own to carry a cached value on (it's an ephemeral query or
	// not-yet-inserted vector), so this is search()'s only way to get it. 0
	// when the configured metric has no cache; distance ignores it in that
	// case, same as every other cache argument in this file.
	targetCache float64,
	// allowlist restricts which nodes may appear in the result set.
	// nil means no filter (all nodes qualify). An empty map means no nodes qualify.
	// Non-matching nodes are still explored for graph connectivity.
	allowlist map[K]struct{},
	// visited tracks the nodes this call has reached. search resets it first,
	// so one set serves every layer of a query or insert. It must be private
	// to the calling goroutine: take it from Graph.acquireVisited.
	visited *visitedSet,
) []searchCandidate[K, V] {
	// resultCap is the breadth exploration actually operates at -- matches
	// hnswlib's searchBaseLayerST exactly: the result/frontier are bounded by
	// ef (here, efSearch), not k, for the entire search; only the very end
	// trims down to the k the caller actually asked for. Using k directly as
	// the exploration bound (the previous behavior) made the "is this
	// candidate worth exploring further" cutoff far too narrow for small k --
	// k=1 (recall@1) was badly hurt by this, since a single found candidate
	// would immediately close off further exploration; k=50 was barely
	// affected since it already exceeded the configured efSearch anyway.
	resultCap := k
	if efSearch > resultCap {
		resultCap = efSearch
	}

	// candidates is a min-heap of nodes to visit, prioritized by distance to the target.
	// Deliberately unbounded in size -- matches hnswlib's candidate_set, which
	// has no size cap at all, only a relevance filter on what gets pushed (see
	// below). Capping it by raw size (the previous behavior) discarded
	// genuinely useful not-yet-explored candidates purely because the
	// frontier got crowded, independent of whether they were worth visiting.
	candidates := heap.Heap[searchCandidate[K, V]]{}
	candidates.Init(make([]searchCandidate[K, V], 0, resultCap))
	candidates.Push(
		searchCandidate[K, V]{
			node: n,
			dist: distance(n.Value, n.precomputed, target, targetCache),
		},
	)
	// result is a max-heap storing the resultCap best (closest) found nodes so far.
	// The worst-kept candidate sits at the root, so checking/evicting it (the hot
	// path below) is O(1)/O(log n) instead of the O(n) scan a min-heap would need.
	result := heap.MaxHeap[searchCandidate[K, V]]{}
	result.Init(make([]searchCandidate[K, V], 0, resultCap))

	// A key can sit in several layers, so a mark left by the layer above
	// would wrongly skip it here.
	visited.reset()

	// Start with the initial node in the result set only if it passes the allowlist.
	// It is always in candidates so its neighbors are explored regardless.
	if allowlist == nil {
		result.Push(candidates.Min())
	} else if _, ok := allowlist[n.Key]; ok {
		result.Push(candidates.Min())
	}
	visited.visit(int(n.Key))

	for candidates.Len() > 0 {
		// Canonical SEARCH-LAYER termination (Malkov & Yashunin, Algorithm 2):
		// candidates is nearest-first, so once the closest remaining candidate
		// is already farther than our worst kept result, nothing still queued
		// can improve the result either — stop. This must be checked against
		// the *next* candidate, not "did the last expansion help": the last
		// expansion can be unproductive while an earlier, still-queued
		// candidate remains genuinely promising.
		if result.Len() >= resultCap && candidates.Min().dist > result.Max().dist {
			break
		}

		current := candidates.Pop().node // Get the closest node from candidates to explore

		// Iterate neighbors directly in their stored order. This used to clone
		// and sort by key first "for test consistency" -- but within one
		// constructed graph, current.neighbors' order is already fully
		// deterministic (a plain slice, not a map), so every search() call
		// against the same graph state produces the same traversal and the
		// same results; it just isn't canonically sorted by key. That clone +
		// sort cost an allocation and an O(m log m) sort on every single node
		// visited during traversal, purely to make tie-break outcomes match a
		// specific order in tests -- tests that care about ties now assert
		// membership among the tied candidates instead of a fixed winner.
		for _, neighbor := range current.neighbors {
			neighborID := neighbor.Key
			if !visited.visit(int(neighborID)) {
				continue
			}

			dist := distance(neighbor.Value, neighbor.precomputed, target, targetCache)

			// Only worth exploring further if it could plausibly improve the
			// result, or the result isn't full yet -- matches hnswlib's
			// searchBaseLayer/ST filter exactly. Skipped when an allowlist is
			// active: a non-matching node can still be a necessary stepping
			// stone to reach a distant matching one, so its own raw distance
			// isn't a safe filter for whether to keep exploring past it.
			if allowlist != nil || result.Len() < resultCap || dist < result.Max().dist {
				candidates.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			}

			// Only add to result if this node passes the allowlist filter.
			if allowlist != nil {
				if _, ok := allowlist[neighborID]; !ok {
					continue
				}
			}

			if result.Len() < resultCap {
				result.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist { // If new node is better than the worst in result set
				result.PopLast()                                               // Remove worst
				result.Push(searchCandidate[K, V]{node: neighbor, dist: dist}) // Add new best
			}
		}
	}

	// Trim down from resultCap (the exploration breadth) to the k results the
	// caller actually asked for -- matches hnswlib's post-search truncation.
	for result.Len() > k {
		result.PopLast()
	}

	// Drain into ascending (closest-first) order. result is a max-heap, so
	// draining it via PopLast() yields descending (worst-first) order in
	// O(k log k); reverse once. Callers (Graph.search, Graph.Search) return
	// this slice as-is with no further sort, so this drain is load-bearing,
	// not cosmetic -- it also fixes a latent gap in the old min-heap
	// Slice()-return: only index 0 was ever guaranteed to be the true
	// closest node (the min-heap root), positions 1..k-1 were raw heap-array
	// order, not actually sorted by distance.
	sorted := make([]searchCandidate[K, V], 0, result.Len())
	for result.Len() > 0 {
		sorted = append(sorted, result.PopLast())
	}
	slices.Reverse(sorted)
	return sorted
}

// Search finds the 'k' nearest neighbors from the target vector 'near'.
// It returns a slice of Node objects, ordered by increasing distance to 'near'.
// The underlying search algorithm utilizes the HNSW graph structure to efficiently
// navigate and find approximate nearest neighbors.
func (h *Graph[K, V]) Search(near V, k int) []Node[K, V] {
	sr := h.search(near, k, nil, 0)
	out := make([]Node[K, V], len(sr))
	for i, node := range sr {
		out[i] = node.Node
	}

	return out
}

// SearchWithDistance finds the 'k' nearest neighbors from the target vector 'near',
// returning them as SearchResult objects which include both the Node and its distance
// to the query vector. The results are ordered by increasing distance.
//
// allowlist restricts which nodes may appear in results — nil means no filter.
// efSearch overrides the graph's default exploration factor — 0 means use the default.
func (h *Graph[K, V]) SearchWithDistance(near V, k int, allowlist map[K]struct{}, efSearch int) []SearchResult[K, V] {
	return h.search(near, k, allowlist, efSearch)
}

// SearchResult represents a single result from a nearest neighbor search.
type SearchResult[K Key, V VectorType] struct {
	Node[K, V]
	Distance float32
}

func (h *Graph[K, V]) search(near V, k int, allowlist map[K]struct{}, efSearch int) []SearchResult[K, V] {
	h.assertDims(near) // Ensure query vector dimensions match graph dimensions.
	if len(h.layers) == 0 {
		return nil // No nodes in the graph to search.
	}

	if efSearch <= 0 {
		efSearch = h.EfSearch // Use the graph's configured EfSearch parameter.
	}

	// Built once per query, not per layer: near's cache doesn't change as
	// the search descends through layers, and rankDistFuncFor()'s result
	// doesn't depend on near at all. Every layer traversed here (including
	// the base layer below) only uses distances for ranking -- picking an
	// elevator, ordering/evicting search() candidates -- so this always
	// prefers the cheaper ranking-only dispatcher; the real distance is
	// recovered via fromRank, once, only for the base layer's final k
	// results below (the one place a value from this function actually
	// reaches a caller).
	distFn := h.rankDistFuncFor()
	nearCache := h.precompute(near)

	var elevator *K // Key of the node that serves as the entry point to the next lower layer.

	visited := h.acquireVisited()
	defer h.releaseVisited(visited)

	// Traverse the graph from the highest layer down to the base layer (0).
	// In higher layers, the search is primarily to find a good starting point (elevator)
	// for the next lower layer. In the base layer, the full search for 'k' neighbors is performed.
	for layerIdx := len(h.layers) - 1; layerIdx >= 0; layerIdx-- {
		currentLayer := h.layers[layerIdx]
		searchPoint := currentLayer.entry() // Default entry point for the current layer.
		if elevator != nil {
			// If an elevator from a higher layer is available, use it as the starting point.
			searchPoint = currentLayer.nodes[*elevator]
		}

		// For layers above the base layer, perform a limited search (k=1) to find the best
		// entry point for the layer below.
		if layerIdx > 0 {
			nodes := searchPoint.search(1, efSearch, near, distFn, nearCache, nil, visited)
			if len(nodes) == 0 {
				// This implies an issue in graph construction or an empty layer.
				// For robustness, if no nodes found, try to use the current searchPoint as elevator.
				elevator = ptr(searchPoint.Key)
			} else {
				elevator = ptr(nodes[0].node.Key)
			}

			continue // Move to the next lower layer.
		}

		// At the base layer (layerIdx == 0), perform the full search for 'k' nearest neighbors.
		nodes := searchPoint.search(k, efSearch, near, distFn, nearCache, allowlist, visited)
		out := make([]SearchResult[K, V], 0, len(nodes))

		for _, node := range nodes {
			out = append(out, SearchResult[K, V]{
				Node: node.node.Node,
				// fromRank converts back from rankDistFuncFor()'s
				// ranking-only value (e.g. squared Euclidean distance) to
				// the real distance -- identity when the configured metric
				// has no such shortcut (fromRank's doc comment). This is
				// the only place in this file a rank value is allowed to
				// reach a caller.
				Distance: h.fromRank(node.dist),
			})
		}

		return out // Results are only returned from the base layer search.
	}

	panic("unreachable: HNSW search loop should always return from base layer")
}
