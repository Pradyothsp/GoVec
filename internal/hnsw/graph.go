package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"time"

	"golang.org/x/exp/maps"

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
// It embeds a Node, adding layer-specific details like a map of neighbors.
type layerNode[K cmp.Ordered, V VectorType] struct {
	Node[K, V]

	// neighbors stores the direct neighbors of this node within the same layer.
	// It's a map for efficient neighbor lookup and deletion, especially useful
	// when the maximum number of neighbors (M) is high.
	neighbors map[K]*layerNode[K, V]
}

// addNeighbor attempts to add a new node as a neighbor to the current layerNode.
// If the current node already has 'm' neighbors (its maximum capacity), it identifies
// the existing neighbor with the "worst" (largest) distance to the current node
// and replaces it with the newNode. This ensures the neighborhood always contains
// the 'm' closest nodes in that layer.
// 'dist' is the distance function used to compare vectors.
func (n *layerNode[K, V]) addNeighbor(newNode *layerNode[K, V], m int, dist DistanceFunc[V]) {
	if n.neighbors == nil {
		n.neighbors = make(map[K]*layerNode[K, V], m)
	}

	n.neighbors[newNode.Key] = newNode
	if len(n.neighbors) <= m {
		return
	}

	// Find the neighbor with the worst distance to replace.
	var (
		worstDist = float32(math.Inf(-1)) // Initialize with negative infinity to find the true max distance
		worst     *layerNode[K, V]
	)
	for _, neighbor := range n.neighbors {
		d := dist(neighbor.Value, n.Value)
		// Compare current neighbor's distance to 'worstDist'. Also handle cases
		// where 'worst' is nil (first iteration) or NaN results from distance function.
		if d > worstDist || worst == nil {
			worstDist = d
			worst = neighbor
		}
	}

	// Remove the worst neighbor and its backlink.
	delete(n.neighbors, worst.Key)
	delete(worst.neighbors, n.Key)
	// Attempt to find a new best neighbor for the disconnected 'worst' node,
	// though this specific call usually leads to no-op if 'worst' was truly the furthest.
	worst.replenish(m, dist)
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
	distance DistanceFunc[V],
	// allowlist restricts which nodes may appear in the result set.
	// nil means no filter (all nodes qualify). An empty map means no nodes qualify.
	// Non-matching nodes are still explored for graph connectivity.
	allowlist map[K]struct{},
) []searchCandidate[K, V] {
	// candidates is a min-heap storing nodes to visit, prioritized by their distance to the target.
	candidates := heap.Heap[searchCandidate[K, V]]{}
	candidates.Init(make([]searchCandidate[K, V], 0, efSearch))
	candidates.Push(
		searchCandidate[K, V]{
			node: n,
			dist: distance(n.Value, target),
		},
	)
	var (
		// result is a max-heap storing the 'k' best (closest) found nodes so far.
		result = heap.Heap[searchCandidate[K, V]]{}
		// visited tracks nodes already processed to avoid redundant work and loops.
		visited = make(map[K]bool)
	)
	result.Init(make([]searchCandidate[K, V], 0, k))

	// Start with the initial node in the result set only if it passes the allowlist.
	// It is always in candidates so its neighbors are explored regardless.
	if allowlist == nil {
		result.Push(candidates.Min())
	} else if _, ok := allowlist[n.Key]; ok {
		result.Push(candidates.Min())
	}
	visited[n.Key] = true

	for candidates.Len() > 0 {
		var (
			current  = candidates.Pop().node // Get the closest node from candidates to explore
			improved = false
		)

		// Iterate through neighbors in a sorted, deterministic fashion for test consistency.
		neighborKeys := maps.Keys(current.neighbors)
		slices.Sort(neighborKeys)
		for _, neighborID := range neighborKeys {
			neighbor := current.neighbors[neighborID]
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.Value, target)

			// Always push to candidates — non-matching nodes still serve as stepping
			// stones for graph connectivity, ensuring reachability of distant matching nodes.
			candidates.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			// Maintain 'candidates' size up to 'efSearch'. If it exceeds, remove the furthest.
			if candidates.Len() > efSearch {
				candidates.PopLast()
			}

			// Only add to result if this node passes the allowlist filter.
			if allowlist != nil {
				if _, ok := allowlist[neighborID]; !ok {
					continue
				}
			}

			// Check if this new neighbor improves the current best result.
			// Guard result.Min() — result may be empty if all nodes so far were filtered.
			improved = improved || result.Len() == 0 || dist < result.Min().dist
			if result.Len() < k {
				result.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist { // If new node is better than the worst in result set
				result.PopLast()                                               // Remove worst
				result.Push(searchCandidate[K, V]{node: neighbor, dist: dist}) // Add new best
			}
		}

		// Termination condition: if no improvement was made and 'k' results are already found,
		// further exploration might not yield significantly better results.
		if !improved && result.Len() >= k {
			break
		}
	}

	return result.Slice()
}

// replenish attempts to restore connectivity for a node that might have lost neighbors,
// for instance, after a neighbor deletion. It looks at the node's existing neighbors
// and their neighbors to find suitable candidates to fill empty neighbor slots up to 'm'.
// This is a naive implementation and could be optimized.
func (n *layerNode[K, V]) replenish(m int, dist DistanceFunc[V]) {
	if len(n.neighbors) >= m {
		return
	}

	// Iterate through current neighbors' neighbors to find new candidates.
	for _, neighbor := range n.neighbors {
		for key, candidate := range neighbor.neighbors {
			if _, ok := n.neighbors[key]; ok {
				// Avoid adding duplicates
				continue
			}
			if candidate == n {
				// Avoid adding itself as a neighbor
				continue
			}
			n.addNeighbor(candidate, m, dist)
			if len(n.neighbors) >= m {
				return // Stop once 'm' neighbors are restored
			}
		}
	}
}

// isolate removes all connections to and from this node within its layer.
// This is a prerequisite for deleting a node to ensure it's fully disconnected.
func (n *layerNode[K, V]) isolate(m int, dist DistanceFunc[V]) {
	// Remove backlinks from its current neighbors.
	for _, neighbor := range n.neighbors {
		delete(neighbor.neighbors, n.Key)
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

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// M is the maximum number of neighbors to keep for each node.
	// A good default for OpenAI embeddings is 16.
	M int

	// Ml is the level generation factor.
	// E.g., for Ml = 0.25, each layer is 1/4 the size of the previous layer.
	Ml float64

	// EfSearch is the number of nodes to consider in the search phase.
	// 20 is a reasonable default. Higher values improve search accuracy at
	// the expense of memory.
	EfSearch int

	// layers is a slice of layers in the graph.
	layers []*layer[K, V]
}

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
		M:        16,
		Ml:       0.25,
		EfSearch: 20,
		Rng:      defaultRand(),
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
	if len(g.layers) == 0 {
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
	for _, node := range nodes {
		key := node.Key
		vec := node.Value

		g.assertDims(vec)              // Ensure dimensional consistency.
		insertLevel := g.randomLevel() // Determine the highest layer this node will be added to.
		// Create layers that don't exist yet, up to the insertLevel.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K, V]{})
		}

		if insertLevel < 0 {
			panic("invalid level generated") // Should not happen with current randomLevel logic.
		}

		var elevator *K // Tracks the entry point into lower layers from higher ones.

		preLen := g.Len() // Store graph length before insertion for invariant check.

		// Insert the node at each layer from the highest (insertLevel) down to the base layer (0).
		for i := len(g.layers) - 1; i >= 0; i-- {
			currentLayer := g.layers[i]
			newNode := &layerNode[K, V]{
				Node: Node[K, V]{
					Key:   key,
					Value: vec,
				},
			}

			// If the current layer is empty, the new node becomes the entry point.
			if currentLayer.entry() == nil {
				currentLayer.nodes = map[K]*layerNode[K, V]{key: newNode}
				continue
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

			// Find the 'M' nearest neighbors in the current layer's local neighborhood.
			neighborhood := searchPoint.search(g.M, g.EfSearch, vec, g.Distance, nil)
			if len(neighborhood) == 0 {
				// This should ideally not happen as the searchPoint itself should be in the result set.
				panic("search returned no nodes")
			}

			// Update the 'elevator' node for the next lower layer. It will be the closest node found.
			elevator = ptr(neighborhood[0].node.Key)

			// If the current layer is at or below the node's insertLevel, add the node.
			if insertLevel >= i {
				// If the node already exists at this key, delete it first to update its position and connections.
				if _, ok := currentLayer.nodes[key]; ok {
					g.Delete(key) // This handles isolating the old node.
				}

				currentLayer.nodes[key] = newNode
				// Create bi-directional connections between the new node and its neighbors in this layer.
				for _, neighborResult := range neighborhood {
					neighborNode := neighborResult.node
					neighborNode.addNeighbor(newNode, g.M, g.Distance)
					newNode.addNeighbor(neighborNode, g.M, g.Distance)
				}
			}
		}

		// Invariant check: ensure the node was successfully added to the graph's base layer.
		if g.Len() != preLen+1 {
			// If adding failed for some reason, and the highest layer is now empty, trim it.
			if len(g.layers) > 0 && g.layers[len(g.layers)-1].entry() == nil {
				g.layers = g.layers[:len(g.layers)-1]
			}
		}
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
			nodes := searchPoint.search(1, efSearch, near, h.Distance, nil)
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
		nodes := searchPoint.search(k, efSearch, near, h.Distance, allowlist)
		out := make([]SearchResult[K, V], 0, len(nodes))

		for _, node := range nodes {
			out = append(out, SearchResult[K, V]{
				Node:     node.node.Node,
				Distance: node.dist,
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
		node.isolate(h.M, h.Distance) // Disconnect the node and replenish its former neighbors.
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
