package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/Pradyothsp/govec/internal/hnsw/heap"
)

// VectorType defines the supported vector storage types for HNSW.
// This constraint ensures compile-time type safety - only []float32 and []int8 are allowed.
// Add new types here when implementing additional quantization methods (e.g., []uint8 for binary).
type VectorType interface {
	[]float32 | []int8
}

// Node is a node in the graph.
// K is the key type (must be comparable/ordered, e.g., uint32, int, string).
// V is the vector type (must satisfy VectorType constraint: []float32 or []int8).
type Node[K cmp.Ordered, V VectorType] struct {
	Key   K
	Value V // Generic vector type
}

// MakeNode creates a new Node with the given key and vector.
func MakeNode[K cmp.Ordered, V VectorType](key K, vec V) Node[K, V] {
	return Node[K, V]{Key: key, Value: vec}
}

// layerNode is an internal representation of a node within a specific layer of the HNSW graph.
// It embeds a Node, adding layer-specific details like its neighbor list.
type layerNode[K cmp.Ordered, V VectorType] struct {
	Node[K, V]

	// neighbors stores the direct neighbors of this node within the same layer,
	// up to a bounded capacity (M). A flat slice, not a map: M is small (tens
	// of entries), and a Go map's per-entry bucket overhead dominates memory
	// at HNSW's scale (one map per node, times every node in the graph) far
	// more than a linear scan over a handful of pointers costs in CPU --
	// confirmed via heap profile, see govec-bench STATUS.md.
	neighbors []*layerNode[K, V]

	// precomputed is an opaque, metric-supplied value computed once via
	// Graph.Precompute when this node is created (on insert or persistence
	// import), and handed back to Graph.CachedDistance on every future
	// comparison involving this node -- letting the metric skip work that's
	// otherwise redundantly repeated on every single comparison (e.g.
	// cosine's per-vector norm: needed on both sides of every distance()
	// call, but it never changes once the vector is stored). layerNode
	// itself never interprets this value -- see distFunc's doc comment.
	//
	// float64, not float32 or a type derived from V (e.g. int64 for
	// []int8's integer sums): []int8's squared-norm accumulates in int64
	// (see core.CosineSimilarityInt8) and can exceed float32's 2^24
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
	// opt in (Graph.Precompute == nil) -- e.g. Euclidean, whose current
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
		// Delete()/isolate(), where a node's neighbors lose a connection
		// permanently (not just to a redistribution the heuristic already
		// picked a diverse replacement set for) and have no other chance at
		// repair.
		d.removeNeighbor(n.Key)
	}
}

// searchCandidate represents a node considered during the HNSW search process,
// paired with its calculated distance to the query vector.
type searchCandidate[K cmp.Ordered, V VectorType] struct {
	node *layerNode[K, V]
	dist float32
}

// Less implements the heap.Interface for searchCandidate, ordering by distance.
// Lower distance means higher priority in the min-heap.
func (s searchCandidate[K, V]) Less(o searchCandidate[K, V]) bool {
	return s.dist < o.dist
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
func selectNeighborsHeuristic[K cmp.Ordered, V VectorType](candidates []searchCandidate[K, V], m int, dist distFunc[V]) []*layerNode[K, V] {
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
	// targetCache is target's precomputed value (see Graph.Precompute's doc
	// comment), computed once by the caller -- target has no layerNode of
	// its own to carry a cached value on (it's an ephemeral query or
	// not-yet-inserted vector), so this is search()'s only way to get it. 0
	// when the configured metric has no cache; distance ignores it in that
	// case, same as every other cache argument in this file.
	targetCache float64,
	// allowlist restricts which nodes may appear in the result set.
	// nil means no filter (all nodes qualify). An empty map means no nodes qualify.
	// Non-matching nodes are still explored for graph connectivity.
	allowlist map[K]struct{},
	// visited is a caller-supplied, pre-cleared scratch map for dedup-tracking
	// nodes seen during this call. nil means "allocate a private one internally"
	// -- required for Graph.search() (the query path), where concurrent calls
	// share the graph under an RLock and can't safely share a map. Graph.Add()
	// (the insert path, always under an exclusive Lock, never concurrent with
	// itself) instead passes a single reused-and-cleared map across every
	// layer/node it processes, avoiding a fresh map allocation + incremental
	// bucket growth on every call -- the actual cost this parameter exists to
	// remove.
	visited map[K]bool,
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

	// visited tracks nodes already processed to avoid redundant work and loops.
	// nil means the caller didn't supply a reusable scratch map -- allocate a
	// private one (see the visited parameter's doc comment above).
	if visited == nil {
		visited = make(map[K]bool)
	}

	// Start with the initial node in the result set only if it passes the allowlist.
	// It is always in candidates so its neighbors are explored regardless.
	if allowlist == nil {
		result.Push(candidates.Min())
	} else if _, ok := allowlist[n.Key]; ok {
		result.Push(candidates.Min())
	}
	visited[n.Key] = true

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
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

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

// isolate removes all connections to and from this node within its layer.
// This is a prerequisite for deleting a node to ensure it's fully disconnected.
func (n *layerNode[K, V]) isolate(m int, dist distFunc[V]) {
	// Remove backlinks from its current neighbors.
	for _, neighbor := range n.neighbors {
		neighbor.removeNeighbor(n.Key)
	}

	// After removing 'n', its former neighbors might have lost a connection.
	// Attempt to replenish their connectivity.
	for _, neighbor := range n.neighbors {
		neighbor.replenish(m, dist)
	}
}

// layer represents a single layer within the HNSW graph, holding a collection of nodes.
// Nodes in higher layers are always a subset of nodes in lower layers.
type layer[K cmp.Ordered, V VectorType] struct {
	// nodes is a map of node IDs to the actual layerNode instances within this layer.
	// This map is exported for compatibility with encoding/gob for persistence.
	nodes map[K]*layerNode[K, V]
}

// entry returns an arbitrary entry node for the layer.
// The specific choice of entry node doesn't significantly impact the algorithm's
// correctness, as long as it is consistent within a search path.
// It returns nil if the layer is empty.
func (l *layer[K, V]) entry() *layerNode[K, V] {
	if l == nil {
		return nil
	}
	// Simply return the first node found in the map.
	for _, node := range l.nodes {
		return node
	}
	return nil
}

// size returns the number of nodes currently in this layer.
func (l *layer[K, V]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World graph.
// All public parameters must be set before adding nodes to the graph.
// K is cmp.Ordered instead of of comparable so that they can be sorted.
// V is the vector type (must satisfy VectorType constraint: []float32 or []int8).
type Graph[K cmp.Ordered, V VectorType] struct {
	// Distance is the distance function used to compare embeddings.
	Distance DistanceFunc[V]

	// Precompute and CachedDistance are an optional, opt-in pair that lets a
	// metric skip work it would otherwise redundantly repeat on every single
	// comparison. When both are set, Precompute(v) is called once per node
	// (on insert or persistence import) and the result is cached on that
	// node (layerNode.precomputed); CachedDistance is then used instead of
	// Distance for every comparison involving that node, receiving both
	// sides' cached values alongside the vectors themselves. Cosine is the
	// motivating case: its distance formula needs each vector's norm as a
	// value standalone from the comparison, and that norm never changes
	// once the vector is stored, so recomputing it on every comparison (as
	// plain Distance does) is pure waste. See CachedCosineDistanceFloat32.
	//
	// nil (the default) means "this metric has nothing worth caching" --
	// every internal comparison falls back to calling Distance directly,
	// identical to before this pair existed. Euclidean is the other metric
	// this package ships, and deliberately leaves these nil: its current
	// single-pass implementation (sum of squared diffs) never computes a
	// standalone per-vector norm in the first place, so there's nothing
	// here for it to cache. Any custom-registered DistanceFunc (see
	// RegisterDistanceFuncFloat32/Int8) that doesn't set these gets the
	// same zero-overhead fallback.
	//
	// CachedDistance must ignore cacheA/cacheB when Precompute is nil --
	// though in that case CachedDistance itself is also nil and never
	// called, so this is enforced by construction, not a runtime check.
	Precompute     func(v V) float64
	CachedDistance func(aVec V, aCache float64, bVec V, bCache float64) float32

	// SquaredDistance and FromSquaredDistance are a second, independent
	// opt-in pair: a cheaper distance-equivalent used only for internal
	// ranking/comparison (search/addNeighbor/replenish/
	// selectNeighborsHeuristic -- everywhere except the k results a query
	// actually returns), paired with a conversion back to the real
	// distance, applied once to just those k results. Euclidean is the
	// motivating case: sqrt is a monotonic transform (a < b implies
	// sqrt(a) < sqrt(b)), so ranking on squared distance produces identical
	// ordering to ranking on real distance, while skipping sqrt() on every
	// comparison except the handful that get returned. See
	// SquaredEuclideanDistanceFloat32/FromSquaredEuclideanDistance.
	//
	// nil (the default) means "no cheaper ranking equivalent for this
	// metric" -- every comparison falls back to Distance/CachedDistance
	// (distFuncFor's existing behavior), and FromSquaredDistance is never
	// called. Cosine deliberately leaves these nil: its distance isn't a
	// monotonic transform of some cheaper quantity the way Euclidean's
	// sqrt is, so there's no equivalent shortcut here for it to use --
	// Precompute/CachedDistance above is cosine's own, differently-shaped
	// optimization.
	//
	// FromSquaredDistance must be set whenever SquaredDistance is -- unlike
	// CachedDistance/Precompute's cache arguments (safely ignorable), a nil
	// FromSquaredDistance would leave squared values flowing out to a
	// caller uncorrected, since rankDistFuncFor's whole purpose is to
	// substitute SquaredDistance in wherever this pair is configured.
	SquaredDistance     DistanceFunc[V]
	FromSquaredDistance func(rank float32) float32

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// M is the maximum number of neighbors to keep for each node.
	// See DefaultM's doc comment for the reasoning behind its value.
	M int

	// Ml is the level generation factor.
	// E.g., for Ml = 0.25, each layer is 1/4 the size of the previous layer.
	Ml float64

	// EfSearch is the number of nodes to consider in the search phase.
	// See DefaultEfSearch's doc comment for the reasoning behind its value.
	// Higher values improve search accuracy (particularly at k > EfSearch) at
	// the expense of latency on every query, including cheap low-k ones.
	EfSearch int

	// EfConstruction is the number of nodes to consider when selecting neighbors
	// for a new node during graph construction (Add). It is independent of
	// EfSearch: a low EfConstruction produces a poorly-connected graph that no
	// amount of EfSearch or M can fully compensate for at query time. See
	// DefaultEfConstruction's doc comment for the reasoning behind its value;
	// higher values build a higher-quality graph at the cost of slower inserts.
	EfConstruction int

	// layers is a slice of layers in the graph.
	layers []*layer[K, V]

	// insertVisited is a reused scratch map for layerNode.search()'s visited
	// tracking during Add(), cleared and passed down instead of letting
	// search() allocate a fresh map on every layer/node it processes. Safe to
	// share across calls (unlike a per-node tag) because Add() always runs
	// under the caller's exclusive lock -- never concurrently with itself or
	// with a Search(). Left nil until Add()'s first call; Search() never
	// touches this field, since concurrent Search() calls (an RLock-permitted
	// pattern) can't safely share one map -- see visitedScratch's doc comment.
	insertVisited map[K]bool

	// BatchParallelism caps how many items' graph searches run concurrently
	// within one Add() call's round-based parallelism (see addRound). 0
	// (the default) resolves to runtime.GOMAXPROCS(0) -- see
	// resolveParallelism. Never touched by addSerial/applyLive.
	BatchParallelism int

	// BatchParallelThreshold is the minimum node count an Add() call needs
	// before round-based parallelism kicks in at all; below it, every node
	// runs through addSerial instead (goroutine/WaitGroup overhead isn't
	// worth it for a handful of items -- and a single-node Add(), the most
	// common call shape, must always resolve here regardless of this value).
	// 0 (the default) resolves to DefaultBatchParallelThreshold -- see
	// resolveParallelism.
	BatchParallelThreshold int
}

// visitedScratch returns g.insertVisited, lazily allocating it on first use
// and clearing it (not reallocating) on every later call -- the map's
// backing bucket array is grown once and then reused for the graph's whole
// lifetime, which is the actual allocation/rehashing cost this exists to
// remove from Add()'s hot path. Only call this from Add() or its helpers,
// which run under an exclusive lock; never from Graph.search() (the query
// path), which must remain safe under concurrent RLock-held calls.
func (g *Graph[K, V]) visitedScratch() map[K]bool {
	if g.insertVisited == nil {
		g.insertVisited = make(map[K]bool)
		return g.insertVisited
	}
	clear(g.insertVisited)
	return g.insertVisited
}

// distFunc computes the distance between two vectors, given each one's
// precomputed cache value (0 and ignored when the configured metric has no
// cache -- see Graph.Precompute's doc comment). It replaces passing a bare
// DistanceFunc[V] to search()/addNeighbor()/replenish()/
// selectNeighborsHeuristic() below: those need each side's cache, not just
// its vector. DistanceFunc[V] itself -- the public, name-registered
// contract used by persistence -- is unchanged; this is purely internal
// plumbing built fresh (via distFuncFor) for each top-level Add()/search()
// call.
type distFunc[V VectorType] func(aVec V, aCache float64, bVec V, bCache float64) float32

// distFuncFor builds this graph's distFunc[V]: CachedDistance when the
// configured metric opted in, or a thin wrapper around Distance (ignoring
// both cache arguments) otherwise. Call once per top-level Add()/search()
// call and thread the result down, rather than re-branching on every
// comparison.
func (g *Graph[K, V]) distFuncFor() distFunc[V] {
	if g.CachedDistance != nil {
		return distFunc[V](g.CachedDistance)
	}
	return noCacheDist(g.Distance)
}

// noCacheDist adapts a plain DistanceFunc[V] into a distFunc[V] that ignores
// both cache arguments -- the shape every internal call site needs, for a
// metric that never populates a cache. Shared by distFuncFor's fallback
// branch and any caller (e.g. tests) that wants to drive search()/
// addNeighbor()/etc. directly with a bare DistanceFunc[V] and no caching.
func noCacheDist[V VectorType](dist DistanceFunc[V]) distFunc[V] {
	return func(aVec V, _ float64, bVec V, _ float64) float32 {
		return dist(aVec, bVec)
	}
}

// precompute returns g.Precompute(v), or 0 if the configured metric has no
// cache (Precompute == nil) -- safe because distFuncFor's non-cached branch
// ignores the cache arguments entirely in that case.
func (g *Graph[K, V]) precompute(v V) float64 {
	if g.Precompute == nil {
		return 0
	}
	return g.Precompute(v)
}

// rankDistFuncFor builds a distFunc[V] for internal ranking/comparison use
// only -- values it produces must never reach a caller as-is; see
// Graph.SquaredDistance's doc comment. Prefers SquaredDistance when the
// metric opted in (cheaper: skips a monotonic transform every comparison
// would otherwise pay for); falls back to distFuncFor's existing behavior
// (CachedDistance, or plain Distance) when it hasn't. Callers whose
// distance values never leave the graph (Add, Delete, and everything they
// call) should always use this instead of distFuncFor -- only
// Graph.search()'s base-layer result needs the real distance, converted
// back via fromRank at that single boundary.
func (g *Graph[K, V]) rankDistFuncFor() distFunc[V] {
	if g.SquaredDistance != nil {
		return noCacheDist(g.SquaredDistance)
	}
	return g.distFuncFor()
}

// fromRank converts a rankDistFuncFor()-produced value back into the real,
// externally-correct distance -- the identity function when the configured
// metric has no cheaper ranking equivalent (SquaredDistance == nil, so
// rankDistFuncFor never substituted anything in the first place).
func (g *Graph[K, V]) fromRank(v float32) float32 {
	if g.FromSquaredDistance == nil {
		return v
	}
	return g.FromSquaredDistance(v)
}

// Default HNSW parameters, used by NewGraph() and mirrored by
// internal/index/factory.go's server-config fallback defaults so the
// standalone-library and server-config paths agree without hand-copying
// literals in two places. internal/config/config.go's DefaultConfig() keeps
// its own literal instead of importing this package -- config is meant to
// stay independent of any specific index implementation -- but its values
// are expected to match these; keep them in sync by hand if either changes.
const (
	// DefaultM is a good default for OpenAI-style embeddings.
	DefaultM = 16
	// DefaultEfSearch: see docs/architecture/HNSW_NEIGHBOR_SELECTION.md's
	// EfConstruction/EfSearch sweep for why 50 -- recall@1/5 plateau there;
	// below it, recall drops with no latency benefit, since resultCap =
	// max(k, EfSearch) already floors at k for k > EfSearch anyway.
	DefaultEfSearch = 50
	// DefaultEfConstruction of 200 is hnswlib's own default and a reasonable
	// starting point -- see docs/architecture/HNSW_EF_CONSTRUCTION.md.
	DefaultEfConstruction = 200
	// DefaultBatchParallelThreshold: below this many nodes in one Add() call,
	// round-based parallelism (see addRound) is skipped entirely in favor of
	// addSerial -- goroutine/WaitGroup overhead isn't worth paying for a
	// handful of items. Picked from a real wall-clock sweep on an 8-core
	// arm64 dev machine (M=16/EfConstruction=200/EfSearch=50, cosine,
	// 128-dim): n=10 parallel was ~19% *slower* than serial (goroutine
	// overhead dominates at that size), n=20 was already ~28% *faster* and
	// the margin only widens from there (n=50: ~55% faster) -- so 20 sits
	// just above the measured crossover, not in the middle of it. See
	// HNSW_BULK_INSERT_PLAN.md §8 for the full sweep.
	DefaultBatchParallelThreshold = 20
)

// defaultRand returns a new pseudo-random number generator initialized with the current time.
// It is used for probabilistic layer assignment during graph construction.
//
// Reproducibility can be achieved by seeding the Rng field of the Graph with a fixed source.
func defaultRand() *rand.Rand {
	//nolint:gosec // G404: Non-crypto RNG acceptable for HNSW layer generation
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

// NewGraph returns a new graph with default parameters, roughly designed for
// storing OpenAI embeddings.
// K is the key type (e.g., uint32).
// V is the vector type (e.g., []float32, []int8).
func NewGraph[K cmp.Ordered, V VectorType]() *Graph[K, V] {
	return &Graph[K, V]{
		M:              DefaultM,
		Ml:             0.25,
		EfSearch:       DefaultEfSearch,
		EfConstruction: DefaultEfConstruction,
		Rng:            defaultRand(),
	}
}

// maxLevel estimates an upper bound on the number of layers a node might be placed in.
// This is used to size the graph's layer slice initially and during level generation.
// It's based on the total number of nodes in the base layer and the Ml factor.
func maxLevel(ml float64, numNodes int) int {
	if ml == 0 {
		panic("ml must be greater than 0") // Ml of 0 would lead to infinite levels.
	}

	if numNodes == 0 {
		return 1 // A graph with no nodes still has a base level (level 0).
	}

	// The formula derives from the probability distribution of layer assignment:
	// P(level >= L) = Ml^L. So, if we want N nodes on average in base layer,
	// then 1/Ml^L = N => L = log(N) / log(1/Ml).
	l := math.Log(float64(numNodes))
	l /= math.Log(1 / ml)

	m := int(math.Round(l)) + 1

	return m
}

// randomLevel generates a random level for a new node based on the graph's Ml parameter.
// This probabilistic assignment ensures a hierarchical structure where higher layers
// contain fewer, more interconnected nodes.
func (h *Graph[K, V]) randomLevel() int {
	// maxLvl is dynamically calculated to provide a reasonable upper bound for level generation,
	// preventing excessive memory allocation for very sparse higher layers when the graph is small.
	maxLvl := 1
	if len(h.layers) > 0 {
		if h.Ml == 0 {
			panic("(*Graph).Ml must be greater than 0") // Should be caught by maxLevel.
		}
		maxLvl = maxLevel(h.Ml, h.layers[0].size())
	}

	for level := 0; level < maxLvl; level++ {
		if h.Rng == nil {
			h.Rng = defaultRand() // Initialize RNG if not set by user.
		}
		r := h.Rng.Float64()
		if r > h.Ml {
			return level
		}
	}

	return maxLvl // Fallback in case loop finishes without returning, typically very high levels.
}

// assertDims checks if the dimensions of the incoming vector 'n' match the
// dimensions of vectors already present in the graph. This ensures consistency.
// It panics if a mismatch is found.
func (g *Graph[K, V]) assertDims(n V) {
	// g.Len() (layer 0's actual node count), not len(g.layers) == 0: a round
	// (addRound)'s serial preamble can grow g.layers ahead of any node
	// actually being inserted into them (level assignment/layer growth for
	// every item in the round all happen before the round's first Apply) --
	// len(g.layers) == 0 would miss that intermediate state and call Dims()
	// against a layer that exists but has no entry yet, panicking.
	if g.Len() == 0 {
		return // No existing vectors to compare against, so no assertion needed.
	}
	hasDims := g.Dims()
	var nLen int
	// Use type assertion to get the length for both []float32 and []int8.
	switch v := any(n).(type) {
	case []float32:
		nLen = len(v)
	case []int8:
		nLen = len(v)
	default:
		// This case should ideally not be reached due to VectorType constraint.
		panic(fmt.Sprintf("unsupported vector type for dimension assertion: %T", n))
	}
	if hasDims != nLen {
		panic(fmt.Sprintf("embedding dimension mismatch: graph has %d dimensions, new vector has %d", hasDims, nLen))
	}
}

// Dims returns the number of dimensions of the vectors stored in the graph.
// It inspects an arbitrary vector in the base layer (if available) to determine its length.
// Returns 0 if the graph is empty.
func (g *Graph[K, V]) Dims() int {
	if len(g.layers) == 0 {
		return 0
	}
	val := g.layers[0].entry().Value
	// Use type assertion to get the length for both []float32 and []int8.
	switch v := any(val).(type) {
	case []float32:
		return len(v)
	case []int8:
		return len(v)
	default:
		// This case should ideally not be reached due to VectorType constraint.
		return 0
	}
}

// ptr is a helper function that returns a pointer to the given value.
// It is useful for creating pointers to literals or temporary values,
// especially in contexts where an addressable value is required.
func ptr[T any](v T) *T {
	return &v
}

// Add inserts one or more nodes into the HNSW graph.
// If a node with the same key already exists, its vector data is updated, and its
// connections are re-evaluated within the graph structure.
// The method ensures that the dimensionality of the incoming vectors matches existing
// vectors in the graph, panicking on a mismatch.
func (g *Graph[K, V]) Add(nodes ...Node[K, V]) {
	// Defensive floor, mirroring hnswlib's own ef_construction_ = max(ef_construction, M_):
	// selectNeighborsHeuristic needs a candidate pool at least as large as what
	// it's selecting from. Config-level validation (internal/config/validation.go)
	// catches this for the normal YAML-configured path, but a Graph constructed
	// directly (tests, or any future caller bypassing config) has no such guard --
	// an unset or too-small EfConstruction would otherwise silently starve every
	// insert's candidate pool instead of erroring.
	efConstruction := g.EfConstruction
	if efConstruction < g.M {
		efConstruction = g.M
	}

	// Built once for the whole batch, not per node/layer: rankDistFuncFor()'s
	// result doesn't depend on which node is being inserted, only on the
	// graph's configured metric. Add() never returns a distance value to a
	// caller, so it always prefers the cheaper ranking-only dispatcher over
	// distFuncFor() -- see rankDistFuncFor's doc comment.
	distFn := g.rankDistFuncFor()

	workers := g.resolveParallelism(len(nodes))
	if workers <= 1 {
		g.addSerial(nodes, efConstruction, distFn)
		return
	}

	// pool is created once per Add() call and reused across every round --
	// safe because addRound's barrier guarantees the previous round's
	// goroutine using a given slot has fully returned before the next
	// round's goroutine for that same slot starts (see visitedScratchPool's
	// doc comment).
	pool := newVisitedScratchPool[K](workers)
	for start := 0; start < len(nodes); start += workers {
		end := start + workers
		if end > len(nodes) {
			end = len(nodes)
		}
		g.addRound(nodes[start:end], efConstruction, distFn, pool)
	}
}

// resolveParallelism decides how many items should run through addRound's
// parallel Prepare phase concurrently for a batch of size n -- 1 means
// "don't parallelize at all, use addSerial" (always true for n <= 1, so a
// single-node Add() call, the most common shape, is unaffected by this
// machinery). Below BatchParallelThreshold, parallelism is skipped even for
// n > 1: goroutine/WaitGroup overhead isn't worth it for a handful of items.
func (g *Graph[K, V]) resolveParallelism(n int) int {
	threshold := g.BatchParallelThreshold
	if threshold <= 0 {
		threshold = DefaultBatchParallelThreshold
	}
	if n < threshold {
		return 1
	}

	workers := g.BatchParallelism
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > n {
		workers = n
	}
	if workers < 1 {
		workers = 1
	}
	return workers
}

// addSerial inserts nodes one at a time, each running fully through
// applyLive against the graph's live current state before the next node
// starts -- this is exactly Add()'s original, single-goroutine behavior,
// used both as the below-threshold path (see resolveParallelism) and,
// within addRound, for any item excluded from that round's parallel path
// (an upsert or an intra-round duplicate key -- see addRound's doc comment).
func (g *Graph[K, V]) addSerial(nodes []Node[K, V], efConstruction int, distFn distFunc[V]) {
	for _, node := range nodes {
		// vecCache is this node's precomputed value (see Graph.Precompute's
		// doc comment) -- computed once here and reused for every layer this
		// node touches below, instead of recomputing it on every layer.
		vecCache := g.precompute(node.Value)

		g.assertDims(node.Value)       // Ensure dimensional consistency.
		insertLevel := g.randomLevel() // Determine the highest layer this node will be added to.
		// Create layers that don't exist yet, up to the insertLevel.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K, V]{})
		}

		if insertLevel < 0 {
			panic("invalid level generated") // Should not happen with current randomLevel logic.
		}

		g.addSerialPrepared(node, insertLevel, vecCache, efConstruction, distFn)
	}
}

// addSerialPrepared is addSerial's per-item body, factored out so addRound's
// serial preamble (which already computes insertLevel/vecCache for every
// item in the round, upsert-or-not) can drive it directly without redoing
// that work.
func (g *Graph[K, V]) addSerialPrepared(node Node[K, V], insertLevel int, vecCache float64, efConstruction int, distFn distFunc[V]) {
	key := node.Key
	vec := node.Value

	var elevator *K // Tracks the entry point into lower layers from higher ones.

	preLen := g.Len() // Store graph length before insertion for invariant check.

	// Insert the node at each layer from the highest (insertLevel) down to the base layer (0).
	for i := len(g.layers) - 1; i >= 0; i-- {
		elevator = g.applyLive(i, key, vec, vecCache, insertLevel, elevator, efConstruction, distFn)
	}

	// Invariant check: ensure the node was successfully added to the graph's base layer.
	if g.Len() != preLen+1 {
		// If adding failed for some reason, and the highest layer is now empty, trim it.
		if len(g.layers) > 0 && g.layers[len(g.layers)-1].entry() == nil {
			g.layers = g.layers[:len(g.layers)-1]
		}
	}
}

// applyLive performs Add()'s original single-item, single-layer insert
// logic directly against the graph's current state -- no plan, no
// parallelism, just layer i's slice of what addSerial used to do inline.
// Returns the elevator key the layer below (i-1) should use as its search
// start point.
func (g *Graph[K, V]) applyLive(
	i int, key K, vec V, vecCache float64, insertLevel int,
	elevator *K, efConstruction int, distFn distFunc[V],
) *K {
	currentLayer := g.layers[i]
	newNode := &layerNode[K, V]{
		Node: Node[K, V]{
			Key:   key,
			Value: vec,
		},
		precomputed: vecCache,
	}

	// If the current layer is empty, the new node becomes the entry point.
	if currentLayer.entry() == nil {
		currentLayer.nodes = map[K]*layerNode[K, V]{key: newNode}
		return elevator
	}

	// Determine the starting point for search in the current layer.
	// For the highest layer, it's the layer's entry point. For lower layers,
	// it's the 'elevator' node found from the layer above.
	searchPoint := currentLayer.entry()
	if elevator != nil {
		searchPoint = currentLayer.nodes[*elevator]
	}

	if g.Distance == nil {
		panic("(*Graph).Distance must be set before adding nodes")
	}

	// If the current layer is at or below the node's insertLevel, the node
	// actually connects here: find a real candidate pool, bounded by
	// EfConstruction (not M) -- selectNeighborsHeuristic below needs a wide
	// pool to choose diverse connections from, not just the M closest.
	if insertLevel >= i {
		candidatePool := searchPoint.search(efConstruction, efConstruction, vec, distFn, vecCache, nil, g.visitedScratch())
		if len(candidatePool) == 0 {
			// This should ideally not happen as the searchPoint itself should be in the result set.
			panic("search returned no nodes")
		}

		// The new elevator for the layer below. It will be the closest node found.
		newElevator := ptr(candidatePool[0].node.Key)

		// If the node already exists at this key, delete it first to update its position and connections.
		if _, ok := currentLayer.nodes[key]; ok {
			g.Delete(key) // This handles isolating the old node.
		}

		currentLayer.nodes[key] = newNode
		// Select up to M diverse neighbors from the candidate pool, then
		// create bi-directional connections between the new node and each.
		// The new node's own outgoing connections are always capped at M,
		// uniformly across every layer (matches hnswlib: getNeighborsByHeuristic2
		// is always called with M_, never Mcurmax). But an existing neighbor's
		// own capacity (mCurMax below) is layer-dependent: hnswlib caps the base
		// layer at 2*M (maxM0_ = M_*2) and every other layer at M (maxM_ = M_) --
		// the base layer holds ~every node and does the most work at query time,
		// so it's allowed to grow denser. Using a uniform M everywhere (the
		// original bug here) silently starved the base layer to half the
		// connectivity a correctly-configured graph should have.
		mCurMax := g.M
		if i == 0 {
			mCurMax = 2 * g.M
		}
		selectedNeighbors := selectNeighborsHeuristic(candidatePool, g.M, distFn)
		for _, neighborNode := range selectedNeighbors {
			neighborNode.addNeighbor(newNode, mCurMax, distFn)
			newNode.addNeighbor(neighborNode, g.M, distFn)
		}
		return newElevator
	}

	// Pass-through layer: this node's insertLevel is below i, so it
	// never connects here -- this layer only exists to hand off a good
	// entry point to the layer below. A true ef=1 greedy descent (the
	// paper's INSERT algorithm, and what hnswlib's addPoint does for
	// exactly this case) is enough: no wider search here ever changes
	// recall, since nothing found gets kept or connected to. The old
	// code ran the full EfConstruction-breadth candidate search on
	// every pass-through layer too -- pure wasted insert cost, worse
	// the higher EfConstruction or the taller the graph.
	found := searchPoint.search(1, 1, vec, distFn, vecCache, nil, g.visitedScratch())
	if len(found) == 0 {
		return ptr(searchPoint.Key)
	}
	return ptr(found[0].node.Key)
}

// planStepKind identifies which of applyLive's three original branches a
// planStep represents -- see insertPlan's doc comment.
type planStepKind uint8

const (
	// stepBecomeEntry means the layer was empty when prepareInsert observed
	// it. Applying this step is only safe as planned if the layer is *still*
	// empty at Apply time -- see applyInsert's doc comment for the hazard
	// this guards against.
	stepBecomeEntry planStepKind = iota
	// stepConnect means insertLevel >= this layer's index: the node actually
	// connects here, with neighbors already selected during Prepare.
	stepConnect
	// stepPassThrough means insertLevel < this layer's index: no connection
	// is made, this step only hands an elevator key to the layer below.
	stepPassThrough
)

// planStep is one layer's worth of prepareInsert's read-only work, replayed
// by applyInsert. newNode (stepBecomeEntry/stepConnect only) is a freshly
// allocated *layerNode, private to this plan until applyInsert publishes it
// into the graph -- no other goroutine can observe or mutate it before then.
type planStep[K cmp.Ordered, V VectorType] struct {
	layerIdx          int
	kind              planStepKind
	newNode           *layerNode[K, V]   // stepBecomeEntry, stepConnect
	selectedNeighbors []*layerNode[K, V] // stepConnect only
	nextElevator      K                  // stepConnect, stepPassThrough only
}

// insertPlan is one item's full top-down set of planStep, computed by
// prepareInsert against the graph state as it stood when the enclosing
// round started, and later replayed serially by applyInsert.
//
// The empty-layer hazard: prepareInsert observes each layer read-only, so
// two round-mates can independently see the same layer as empty and both
// plan a stepBecomeEntry there. Applying both literally would let the
// second overwrite the first (layer.nodes is a plain map keyed by K) --
// silently losing the first node's connections entirely, not just a quality
// loss. applyInsert re-checks emptiness at Apply time and falls back to a
// live, single-layer redo (via applyLive) for every claim after the first;
// see applyInsert's doc comment.
type insertPlan[K cmp.Ordered, V VectorType] struct {
	node        Node[K, V]
	vecCache    float64
	insertLevel int
	steps       []planStep[K, V] // top-down: highest layer first
}

// visitedScratchPool hands each concurrent Prepare goroutine within one
// addRound call its own scratch map for layerNode.search()'s visited
// tracking -- g.insertVisited (the single shared map addSerial/applyLive
// use) is only safe because Add() never previously ran concurrently with
// itself; multiple Prepare goroutines sharing one map would both race and
// corrupt each other's traversal. Sized once per Add() call (to
// resolveParallelism's worker count) and reused across every round within
// that call: safe because addRound's sync.WaitGroup barrier guarantees the
// previous round's goroutine using a given slot has fully returned before
// the next round's goroutine for that same slot starts.
type visitedScratchPool[K cmp.Ordered] struct {
	maps []map[K]bool
}

// newVisitedScratchPool allocates size empty maps, one per concurrent slot.
func newVisitedScratchPool[K cmp.Ordered](size int) *visitedScratchPool[K] {
	maps := make([]map[K]bool, size)
	for i := range maps {
		maps[i] = make(map[K]bool)
	}
	return &visitedScratchPool[K]{maps: maps}
}

// forSlot clears and returns the map for the given slot -- callers must not
// retain it past the current prepareInsert call, and no two goroutines may
// ever call forSlot with the same slot concurrently (see the pool's own doc
// comment for why that's guaranteed).
func (p *visitedScratchPool[K]) forSlot(slot int) map[K]bool {
	clear(p.maps[slot])
	return p.maps[slot]
}

// prepareInsert computes node's full top-down insert plan by walking the
// graph exactly as applyLive does, but read-only: every mutation applyLive
// would perform (layer.nodes writes, addNeighbor calls, Delete) is replaced
// here with plan-step construction instead. Safe to call concurrently with
// other prepareInsert calls for the same round -- each only reads
// g.layers/layer.nodes/layerNode.neighbors/.Value/.precomputed/.Key, using
// its own caller-supplied visited scratch map (never g.insertVisited, which
// is reserved for the always-serial addSerial/applyLive paths).
//
// visited is cleared before each layer's search call, not just once at the
// top -- matching g.visitedScratch()'s per-call-fresh semantics exactly
// (see that method's doc comment): a node key can legitimately appear in
// more than one layer (higher layers are subsets of lower ones), so
// carrying visited-marks across layers would incorrectly suppress
// exploration in a later layer's unrelated traversal.
func (g *Graph[K, V]) prepareInsert(
	node Node[K, V], insertLevel int, vecCache float64,
	efConstruction int, distFn distFunc[V], visited map[K]bool,
) *insertPlan[K, V] {
	key := node.Key
	vec := node.Value

	plan := &insertPlan[K, V]{
		node:        node,
		vecCache:    vecCache,
		insertLevel: insertLevel,
		steps:       make([]planStep[K, V], 0, len(g.layers)),
	}

	var elevator *K

	for i := len(g.layers) - 1; i >= 0; i-- {
		currentLayer := g.layers[i]

		if currentLayer.entry() == nil {
			// Only legitimate if this item's own insertLevel actually
			// reaches this layer. A round's serial preamble grows g.layers
			// collectively to the round's *max* insertLevel across every
			// item in it -- so an item whose own insertLevel is lower than
			// that max will still walk through these newly-created empty
			// layers (the top-down loop always starts at len(g.layers)-1,
			// unconditionally). In serial addSerial/applyLive this
			// combination (entry()==nil but insertLevel<i) can never occur
			// -- a layer only ever exists because some single item's own
			// insertLevel warranted creating it, immediately followed by
			// that same item's own entry claim in the same synchronous
			// call. Under round-based batching that tight coupling breaks:
			// without this guard, a low-level item would wrongly become a
			// permanent (if isolated) entry at a layer far above where it
			// belongs, silently violating "higher layers are a subset of
			// lower layers" for every later search that starts from there
			// -- not a recall trade-off, real graph corruption. So: skip
			// entirely, no plan step, elevator unchanged -- deferring the
			// real claim to whichever round-mate (guaranteed to exist,
			// since it's the reason this layer was grown at all) actually
			// has insertLevel >= i.
			if insertLevel >= i {
				newNode := &layerNode[K, V]{
					Node:        Node[K, V]{Key: key, Value: vec},
					precomputed: vecCache,
				}
				plan.steps = append(plan.steps, planStep[K, V]{
					layerIdx: i,
					kind:     stepBecomeEntry,
					newNode:  newNode,
				})
			}
			// elevator deliberately left unchanged -- matches applyLive's
			// original "continue" here exactly: a stepBecomeEntry layer
			// never hands a new elevator to the layer below.
			continue
		}

		searchPoint := currentLayer.entry()
		if elevator != nil {
			searchPoint = currentLayer.nodes[*elevator]
		}

		if g.Distance == nil {
			panic("(*Graph).Distance must be set before adding nodes")
		}

		if insertLevel >= i {
			clear(visited)
			candidatePool := searchPoint.search(efConstruction, efConstruction, vec, distFn, vecCache, nil, visited)
			if len(candidatePool) == 0 {
				panic("search returned no nodes")
			}

			newElevator := candidatePool[0].node.Key
			newNode := &layerNode[K, V]{
				Node:        Node[K, V]{Key: key, Value: vec},
				precomputed: vecCache,
			}
			selectedNeighbors := selectNeighborsHeuristic(candidatePool, g.M, distFn)

			plan.steps = append(plan.steps, planStep[K, V]{
				layerIdx:          i,
				kind:              stepConnect,
				newNode:           newNode,
				selectedNeighbors: selectedNeighbors,
				nextElevator:      newElevator,
			})
			elevator = &newElevator
		} else {
			clear(visited)
			found := searchPoint.search(1, 1, vec, distFn, vecCache, nil, visited)
			var newElevator K
			if len(found) == 0 {
				newElevator = searchPoint.Key
			} else {
				newElevator = found[0].node.Key
			}

			plan.steps = append(plan.steps, planStep[K, V]{
				layerIdx:     i,
				kind:         stepPassThrough,
				nextElevator: newElevator,
			})
			elevator = &newElevator
		}
	}

	return plan
}

// applyInsert replays plan's steps serially, in top-down order, performing
// the mutations prepareInsert decided but never carried out itself.
//
// stepConnect/stepPassThrough need no re-validation against the graph's
// current state: with upserts excluded from the round-parallel path (see
// addRound), no round-mate ever calls Delete() mid-round, so
// selectedNeighbors' pointers stay valid for the whole round. A "stale"
// stepConnect plan -- frozen against the pre-round snapshot -- just produces
// the already-accepted lower-quality connection (the documented trade-off
// of round-based parallelism), never corruption.
//
// stepBecomeEntry does need re-validation: it re-checks
// currentLayer.entry() == nil at Apply time, not trusting the Prepare-time
// observation (see insertPlan's doc comment on the empty-layer hazard). If
// still nil, this item legitimately becomes the entry -- apply as planned.
// If another round-mate already claimed it, fall back to a live,
// single-item, single-layer redo via applyLive, against the graph's current
// (already-partially-mutated-by-this-round) state -- exactly what
// addSerialPrepared would do for this one layer if this item were being
// inserted right now, one at a time. This fallback is local to the one
// stale layer: it does not cascade into this item's remaining plan steps.
// In the original algorithm a stepBecomeEntry layer never sets elevator (it
// just continues), so the layer below always uses currentLayer.entry() as
// its search start point regardless of who the real entry turns out to be
// -- exactly what this item's already-planned stepConnect/stepPassThrough
// steps below it assumed too, so nothing here needs to propagate downward.
//
// Also note: Add()'s end-of-loop "trim top layer if still empty" invariant
// guard is deliberately not replicated here -- trimming g.layers mid-round
// would desync every other round-mate's already-computed step.layerIdx
// references. That guard stays only in addSerialPrepared.
func (g *Graph[K, V]) applyInsert(plan *insertPlan[K, V], efConstruction int, distFn distFunc[V]) {
	key := plan.node.Key

	for _, step := range plan.steps {
		currentLayer := g.layers[step.layerIdx]

		switch step.kind {
		case stepBecomeEntry:
			if currentLayer.entry() == nil {
				currentLayer.nodes = map[K]*layerNode[K, V]{key: step.newNode}
				continue
			}
			// Hazard: a round-mate applied earlier in this same loop and
			// already claimed this layer. Redo this one layer live, elevator
			// intentionally nil -- see this method's doc comment.
			g.applyLive(step.layerIdx, key, plan.node.Value, plan.vecCache, plan.insertLevel, nil, efConstruction, distFn)

		case stepConnect:
			currentLayer.nodes[key] = step.newNode
			mCurMax := g.M
			if step.layerIdx == 0 {
				mCurMax = 2 * g.M
			}
			for _, neighborNode := range step.selectedNeighbors {
				neighborNode.addNeighbor(step.newNode, mCurMax, distFn)
				step.newNode.addNeighbor(neighborNode, g.M, distFn)
			}

		case stepPassThrough:
			// No mutation -- this layer only handed off an elevator during
			// Prepare, already baked into the layer below's own plan step.
		}
	}
}

// addRound processes one round of a batch: a serial preamble (level
// assignment, layer growth, precompute -- the only touch points for
// g.Rng/g.layers, so they must stay serial), then every item's Prepare
// running concurrently, a hard barrier, then every item's Apply running
// serially. No read ever overlaps in time with a write to the same data:
// Prepare only reads, and nothing in Apply starts until every Prepare
// goroutine has returned.
//
// Upserts (node.Key already present in the graph's base layer) and
// intra-round duplicate keys are excluded from the parallel path and
// instead run through addSerialPrepared immediately, in the preamble, in
// their original order -- mixing them into a parallel round would let one
// item's Delete() (a real mutation touching other nodes' neighbor lists)
// invalidate another round-mate's already-computed plan. See
// insertPlan's doc comment for why plain new inserts don't have this
// problem.
func (g *Graph[K, V]) addRound(nodes []Node[K, V], efConstruction int, distFn distFunc[V], pool *visitedScratchPool[K]) {
	n := len(nodes)
	insertLevels := make([]int, n)
	vecCaches := make([]float64, n)
	parallelIdx := make([]int, 0, n)
	seen := make(map[K]bool, n)

	for i, node := range nodes {
		g.assertDims(node.Value)
		insertLevel := g.randomLevel()
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K, V]{})
		}
		if insertLevel < 0 {
			panic("invalid level generated")
		}
		insertLevels[i] = insertLevel
		vecCaches[i] = g.precompute(node.Value)

		alreadyExists := len(g.layers) > 0 && g.layers[0].nodes[node.Key] != nil
		if alreadyExists || seen[node.Key] {
			g.addSerialPrepared(node, insertLevels[i], vecCaches[i], efConstruction, distFn)
			continue
		}
		seen[node.Key] = true
		parallelIdx = append(parallelIdx, i)
	}

	if len(parallelIdx) == 0 {
		return
	}

	plans := make([]*insertPlan[K, V], len(parallelIdx))
	var wg sync.WaitGroup
	wg.Add(len(parallelIdx))
	for slot, idx := range parallelIdx {
		go func(slot, idx int) {
			defer wg.Done()
			plans[slot] = g.prepareInsert(nodes[idx], insertLevels[idx], vecCaches[idx], efConstruction, distFn, pool.forSlot(slot))
		}(slot, idx)
	}
	wg.Wait() // The barrier: nothing below runs until every Prepare goroutine above has returned.

	for _, plan := range plans {
		g.applyInsert(plan, efConstruction, distFn)
	}
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
type SearchResult[K cmp.Ordered, V VectorType] struct {
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
			nodes := searchPoint.search(1, efSearch, near, distFn, nearCache, nil, nil)
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
		nodes := searchPoint.search(k, efSearch, near, distFn, nearCache, allowlist, nil)
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

// Len returns the total number of nodes (vectors) currently stored in the HNSW graph.
// This count is derived from the number of nodes in the base layer (layer 0).
func (h *Graph[K, V]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

// Delete removes a node from the graph specified by its key.
// It attempts to preserve the graph's clustering properties by reconnecting
// affected neighbors, ensuring graph integrity after removal.
// Returns true if the node was found and deleted, false otherwise.
func (h *Graph[K, V]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false // Cannot delete from an empty graph.
	}

	deleteLayer := map[int]struct{}{} // Keep track of layers that become empty.
	var deleted bool
	dist := h.rankDistFuncFor() // built once, reused across every layer this key appears in; Delete never returns a distance value
	// Iterate through all layers, from highest to lowest, to remove the node.
	for i, layer := range h.layers {
		node, ok := layer.nodes[key]
		if !ok {
			continue // Node not found in this layer, continue to next.
		}
		delete(layer.nodes, key) // Remove the node from the current layer.
		if len(layer.nodes) == 0 {
			deleteLayer[i] = struct{}{} // Mark layer as empty.
		}
		mCurMax := h.M
		if i == 0 {
			mCurMax = 2 * h.M // base layer allows up to 2*M connections -- see Add's comment
		}
		node.isolate(mCurMax, dist) // Disconnect the node and replenish its former neighbors.
		deleted = true
	}

	// If any layers became empty, reconstruct the layers slice to remove them.
	if len(deleteLayer) > 0 {
		newLayers := make([]*layer[K, V], 0, len(h.layers)-len(deleteLayer))
		for i, layer := range h.layers {
			if _, ok := deleteLayer[i]; ok {
				continue // Skip empty layers.
			}
			newLayers = append(newLayers, layer)
		}

		h.layers = newLayers
	}

	return deleted
}

// Lookup retrieves the vector associated with the given key from the graph's base layer.
// Returns the vector and true if found, or a zero-value vector and false if not found or the graph is empty.
func (h *Graph[K, V]) Lookup(key K) (V, bool) {
	if len(h.layers) == 0 {
		var zero V // Return zero-value for the generic vector type.
		return zero, false
	}

	// Look up in the base layer (layer 0).
	node, ok := h.layers[0].nodes[key]
	if !ok {
		var zero V
		return zero, false
	}
	return node.Value, ok
}
