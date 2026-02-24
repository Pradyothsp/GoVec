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

// layerNode is a node in a layer of the graph.
type layerNode[K cmp.Ordered, V VectorType] struct {
	Node[K, V]

	// neighbors is map of neighbor keys to neighbor nodes.
	// It is a map and not a slice to allow for efficient deletes, esp.
	// when M is high.
	neighbors map[K]*layerNode[K, V]
}

// addNeighbor adds a o neighbor to the node, replacing the neighbor
// with the worst distance if the neighbor set is full.
func (n *layerNode[K, V]) addNeighbor(newNode *layerNode[K, V], m int, dist DistanceFunc[V]) {
	if n.neighbors == nil {
		n.neighbors = make(map[K]*layerNode[K, V], m)
	}

	n.neighbors[newNode.Key] = newNode
	if len(n.neighbors) <= m {
		return
	}

	// Find the neighbor with the worst distance.
	var (
		worstDist = float32(math.Inf(-1))
		worst     *layerNode[K, V]
	)
	for _, neighbor := range n.neighbors {
		d := dist(neighbor.Value, n.Value)
		// d > worstDist may always be false if the distance function
		// returns NaN, e.g., when the embeddings are zero.
		if d > worstDist || worst == nil {
			worstDist = d
			worst = neighbor
		}
	}

	delete(n.neighbors, worst.Key)
	// Delete backlink from the worst neighbor.
	delete(worst.neighbors, n.Key)
	worst.replenish(m, dist)
}

type searchCandidate[K cmp.Ordered, V VectorType] struct {
	node *layerNode[K, V]
	dist float32
}

func (s searchCandidate[K, V]) Less(o searchCandidate[K, V]) bool {
	return s.dist < o.dist
}

// search returns the layer node closest to the target node
// within the same layer.
func (n *layerNode[K, V]) search(
	// k is the number of candidates in the result set.
	k int,
	efSearch int,
	target V,
	distance DistanceFunc[V],
) []searchCandidate[K, V] {
	// This is a basic greedy algorithm to find the entry point at the given level
	// that is closest to the target node.
	candidates := heap.Heap[searchCandidate[K, V]]{}
	candidates.Init(make([]searchCandidate[K, V], 0, efSearch))
	candidates.Push(
		searchCandidate[K, V]{
			node: n,
			dist: distance(n.Value, target),
		},
	)
	var (
		result  = heap.Heap[searchCandidate[K, V]]{}
		visited = make(map[K]bool)
	)
	result.Init(make([]searchCandidate[K, V], 0, k))

	// Begin with the entry node in the result set.
	result.Push(candidates.Min())
	visited[n.Key] = true

	for candidates.Len() > 0 {
		var (
			current  = candidates.Pop().node
			improved = false
		)

		// We iterate the map in a sorted, deterministic fashion for
		// tests.
		neighborKeys := maps.Keys(current.neighbors)
		slices.Sort(neighborKeys)
		for _, neighborID := range neighborKeys {
			neighbor := current.neighbors[neighborID]
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.Value, target)
			improved = improved || dist < result.Min().dist
			if result.Len() < k {
				result.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist {
				result.PopLast()
				result.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			}

			candidates.Push(searchCandidate[K, V]{node: neighbor, dist: dist})
			// Always store candidates if we haven't reached the limit.
			if candidates.Len() > efSearch {
				candidates.PopLast()
			}
		}

		// Termination condition: no improvement in distance and at least
		// kMin candidates in the result set.
		if !improved && result.Len() >= k {
			break
		}
	}

	return result.Slice()
}

func (n *layerNode[K, V]) replenish(m int, dist DistanceFunc[V]) {
	if len(n.neighbors) >= m {
		return
	}

	// Restore connectivity by adding new neighbors.
	// This is a naive implementation that could be improved by
	// using a priority queue to find the best candidates.
	for _, neighbor := range n.neighbors {
		for key, candidate := range neighbor.neighbors {
			if _, ok := n.neighbors[key]; ok {
				// do not add duplicates
				continue
			}
			if candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, dist)
			if len(n.neighbors) >= m {
				return
			}
		}
	}
}

// isolates remove the node from the graph by removing all connections
// to neighbors.
func (n *layerNode[K, V]) isolate(m int, dist DistanceFunc[V]) {
	for _, neighbor := range n.neighbors {
		delete(neighbor.neighbors, n.Key)
	}

	for _, neighbor := range n.neighbors {
		neighbor.replenish(m, dist)
	}
}

type layer[K cmp.Ordered, V VectorType] struct {
	// nodes is a map of nodes IDs to nodes.
	// All nodes in a higher layer are also in the lower layers, an essential
	// property of the graph.
	//
	// nodes is exported for interop with encoding/gob.
	nodes map[K]*layerNode[K, V]
}

// entry returns the entry node of the layer.
// It doesn't matter which node is returned, even that the
// entry node is consistent, so we just return the first node
// in the map to avoid tracking extra state.
func (l *layer[K, V]) entry() *layerNode[K, V] {
	if l == nil {
		return nil
	}
	for _, node := range l.nodes {
		return node
	}
	return nil
}

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

// maxLevel returns an upper-bound on the number of levels in the graph
// based on the size of the base layer.
func maxLevel(ml float64, numNodes int) int {
	if ml == 0 {
		panic("ml must be greater than 0")
	}

	if numNodes == 0 {
		return 1
	}

	l := math.Log(float64(numNodes))
	l /= math.Log(1 / ml)

	m := int(math.Round(l)) + 1

	return m
}

// randomLevel generates a random level for a new node.
func (h *Graph[K, V]) randomLevel() int {
	// maxLvl avoids having to accept an additional parameter for the maximum level
	// by calculating a probably good one from the size of the base layer.
	maxLvl := 1
	if len(h.layers) > 0 {
		if h.Ml == 0 {
			panic("(*Graph).Ml must be greater than 0")
		}
		maxLvl = maxLevel(h.Ml, h.layers[0].size())
	}

	for level := 0; level < maxLvl; level++ {
		if h.Rng == nil {
			h.Rng = defaultRand()
		}
		r := h.Rng.Float64()
		if r > h.Ml {
			return level
		}
	}

	return maxLvl
}

func (g *Graph[K, V]) assertDims(n V) {
	if len(g.layers) == 0 {
		return
	}
	hasDims := g.Dims()
	// Get length by type assertion - both []float32 and []int8 support len()
	var nLen int
	switch v := any(n).(type) {
	case []float32:
		nLen = len(v)
	case []int8:
		nLen = len(v)
	}
	if hasDims != nLen {
		panic(fmt.Sprint("embedding dimension mismatch: ", hasDims, " != ", nLen))
	}
}

// Dims returns the number of dimensions in the graph, or
// 0 if the graph is empty.
func (g *Graph[K, V]) Dims() int {
	if len(g.layers) == 0 {
		return 0
	}
	val := g.layers[0].entry().Value
	// Get length by type assertion - both []float32 and []int8 support len()
	switch v := any(val).(type) {
	case []float32:
		return len(v)
	case []int8:
		return len(v)
	}
	return 0
}

func ptr[T any](v T) *T {
	return &v
}

// Add inserts nodes into the graph.
// If another node with the same ID exists, it is replaced.
func (g *Graph[K, V]) Add(nodes ...Node[K, V]) {
	for _, node := range nodes {
		key := node.Key
		vec := node.Value

		g.assertDims(vec)
		insertLevel := g.randomLevel()
		// Create layers that don't exist yet.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K, V]{})
		}

		if insertLevel < 0 {
			panic("invalid level")
		}

		var elevator *K

		preLen := g.Len()

		// Insert node at each layer, beginning with the highest.
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]
			newNode := &layerNode[K, V]{
				Node: Node[K, V]{
					Key:   key,
					Value: vec,
				},
			}

			// Insert the new node into the layer.
			if layer.entry() == nil {
				layer.nodes = map[K]*layerNode[K, V]{key: newNode}
				continue
			}

			// Now at the highest layer with more than one node, so we can begin
			// searching for the best way to enter the graph.
			searchPoint := layer.entry()

			// On subsequent layers, we use the elevator node to enter the graph
			// at the best point.
			if elevator != nil {
				searchPoint = layer.nodes[*elevator]
			}

			if g.Distance == nil {
				panic("(*Graph).Distance must be set")
			}

			neighborhood := searchPoint.search(g.M, g.EfSearch, vec, g.Distance)
			if len(neighborhood) == 0 {
				// This should never happen because the searchPoint itself
				// should be in the result set.
				panic("no nodes found")
			}

			// Re-set the elevator node for the next layer.
			elevator = ptr(neighborhood[0].node.Key)

			if insertLevel >= i {
				if _, ok := layer.nodes[key]; ok {
					g.Delete(key)
				}
				// Insert the new node into the layer.
				layer.nodes[key] = newNode
				for _, node := range neighborhood {
					// Create a bi-directional edge between the new node and the best node.
					node.node.addNeighbor(newNode, g.M, g.Distance)
					newNode.addNeighbor(node.node, g.M, g.Distance)
				}
			}
		}

		// Invariant check: the node should have been added to the graph.
		if g.Len() != preLen+1 {
			if len(g.layers) > 0 && g.layers[len(g.layers)-1].entry() == nil {
				g.layers = g.layers[:len(g.layers)-1]
			}
		}
	}
}

// Search finds the k nearest neighbors from the target node.
func (h *Graph[K, V]) Search(near V, k int) []Node[K, V] {
	sr := h.search(near, k)
	out := make([]Node[K, V], len(sr))
	for i, node := range sr {
		out[i] = node.Node
	}
	return out
}

// SearchWithDistance finds the k nearest neighbors from the target node
// and returns the distance.
func (h *Graph[K, V]) SearchWithDistance(near V, k int) []SearchResult[K, V] {
	return h.search(near, k)
}

// SearchResult represents a single result from a nearest neighbor search.
type SearchResult[K cmp.Ordered, V VectorType] struct {
	Node[K, V]
	Distance float32
}

func (h *Graph[K, V]) search(near V, k int) []SearchResult[K, V] {
	h.assertDims(near)
	if len(h.layers) == 0 {
		return nil
	}

	var (
		efSearch = h.EfSearch

		elevator *K
	)

	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		searchPoint := h.layers[layer].entry()
		if elevator != nil {
			searchPoint = h.layers[layer].nodes[*elevator]
		}

		// Descending hierarchies
		if layer > 0 {
			nodes := searchPoint.search(1, efSearch, near, h.Distance)
			elevator = ptr(nodes[0].node.Key)
			continue
		}

		nodes := searchPoint.search(k, efSearch, near, h.Distance)
		out := make([]SearchResult[K, V], 0, len(nodes))

		for _, node := range nodes {
			out = append(out, SearchResult[K, V]{
				Node:     node.node.Node,
				Distance: node.dist,
			})
		}

		return out
	}

	panic("unreachable")
}

// Len returns the number of nodes in the graph.
func (h *Graph[K, V]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

// Delete removes a node from the graph by key.
// It tries to preserve the clustering properties of the graph by
// replenishing connectivity in the affected neighborhoods.
func (h *Graph[K, V]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false
	}

	deleteLayer := map[int]struct{}{}
	var deleted bool
	for i, layer := range h.layers {
		node, ok := layer.nodes[key]
		if !ok {
			continue
		}
		delete(layer.nodes, key)
		if len(layer.nodes) == 0 {
			deleteLayer[i] = struct{}{}
		}
		node.isolate(h.M, h.Distance)
		deleted = true
	}

	if len(deleteLayer) > 0 {
		newLayers := make([]*layer[K, V], 0, len(h.layers)-len(deleteLayer))
		for i, layer := range h.layers {
			if _, ok := deleteLayer[i]; ok {
				continue
			}
			newLayers = append(newLayers, layer)
		}

		h.layers = newLayers
	}

	return deleted
}

// Lookup returns the vector with the given key.
func (h *Graph[K, V]) Lookup(key K) (V, bool) {
	if len(h.layers) == 0 {
		var zero V
		return zero, false
	}

	node, ok := h.layers[0].nodes[key]
	if !ok {
		var zero V
		return zero, false
	}
	return node.Value, ok
}
