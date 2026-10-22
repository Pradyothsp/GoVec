package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/Pradyothsp/govec/internal/metric"
)

// VectorType is the vector types a graph can hold: the metric package's, so a
// graph and the metric it runs on always agree on them.
type VectorType = metric.VectorType

// Node is a node in the graph.
// K is the key type, a non-negative integer (see Key).
// V is the vector type (must satisfy VectorType constraint: []float32 or []int8).
type Node[K Key, V VectorType] struct {
	Key   K
	Value V // Generic vector type
}

// MakeNode creates a new Node with the given key and vector.
func MakeNode[K Key, V VectorType](key K, vec V) Node[K, V] {
	return Node[K, V]{Key: key, Value: vec}
}

// layer represents a single layer within the HNSW graph, holding a collection of nodes.
// Nodes in higher layers are always a subset of nodes in lower layers.
type layer[K Key, V VectorType] struct {
	// nodes is a map of node IDs to the actual layerNode instances within this layer.
	// Write it only through put and remove, which keep first in step.
	nodes map[K]*layerNode[K, V]

	// first is the node with the smallest key, the layer's entry point. It
	// used to be whichever node a map iteration returned first, which Go
	// randomizes: the same nodes and seed built a different graph every time,
	// so benchmarks compared two random graphs. The smallest key depends only
	// on what the layer holds, so a graph reloaded from a snapshot gets the
	// same entry points without the snapshot storing them.
	first *layerNode[K, V]
}

// entry returns the layer's entry point, or nil if the layer is empty.
func (l *layer[K, V]) entry() *layerNode[K, V] {
	if l == nil {
		return nil
	}

	return l.first
}

// put adds n to the layer, replacing any node with the same key.
func (l *layer[K, V]) put(n *layerNode[K, V]) {
	if l.nodes == nil {
		l.nodes = make(map[K]*layerNode[K, V])
	}

	l.nodes[n.Key] = n

	if l.first == nil || n.Key <= l.first.Key {
		l.first = n
	}
}

// remove deletes key from the layer. Removing the entry point scans the layer
// for the next smallest key: a full pass, but only when that one node goes.
func (l *layer[K, V]) remove(key K) {
	delete(l.nodes, key)

	if l.first != nil && l.first.Key == key {
		l.findFirst()
	}
}

// findFirst sets first to the node with the smallest key, or nil if the
// layer is empty.
func (l *layer[K, V]) findFirst() {
	l.first = nil

	for _, n := range l.nodes {
		if l.first == nil || n.Key < l.first.Key {
			l.first = n
		}
	}
}

// size returns the number of nodes currently in this layer.
func (l *layer[K, V]) size() int {
	if l == nil {
		return 0
	}

	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World graph.
// Build one with NewGraph; set Rng before adding nodes, if at all.
// K is an integer key; see Key.
// V is the vector type (must satisfy VectorType constraint: []float32 or []int8).
type Graph[K Key, V VectorType] struct {
	// Metric is how the graph compares vectors. Distance is required. The
	// optional pairs are used when set: Precompute/CachedDistance caches a
	// value per node (layerNode.precomputed) and compares with it, and
	// SquaredDistance ranks every comparison inside the graph, with
	// FromSquaredDistance converting just the returned results back (see
	// rankDistFuncFor). Similarity and Score are the caller's; the graph
	// doesn't use them.
	Metric metric.Metric[V]

	// Config is how the graph is built and searched.
	Config

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// layers is a slice of layers in the graph.
	layers []*layer[K, V]

	// visitedPool holds *visitedSet for searches and inserts to reuse, so a
	// traversal allocates nothing once the pool has warmed up. See
	// acquireVisited.
	visitedPool sync.Pool
}

// defaultRand returns a new pseudo-random number generator initialized with the current time.
// It is used for probabilistic layer assignment during graph construction.
//
// Reproducibility can be achieved by seeding the Rng field of the Graph with a fixed source.
func defaultRand() *rand.Rand {
	return seededRand(time.Now().UnixNano())
}

// seededRand returns the generator for Config.Seed: seeded with it, or from
// the clock when it is 0.
func seededRand(seed int64) *rand.Rand {
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	//nolint:gosec // G404: Non-crypto RNG acceptable for HNSW layer generation
	return rand.New(rand.NewSource(seed))
}

// NewGraph returns an empty graph that compares vectors with m and is built
// and searched as cfg says, with this package's defaults for its zero fields.
// K is the key type (e.g., uint32).
// V is the vector type (e.g., []float32, []int8).
func NewGraph[K Key, V VectorType](m metric.Metric[V], cfg Config) *Graph[K, V] {
	return &Graph[K, V]{
		Metric: m,
		Config: cfg.withDefaults(),
		Rng:    seededRand(cfg.Seed),
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

// Len returns the total number of nodes (vectors) currently stored in the HNSW graph.
// This count is derived from the number of nodes in the base layer (layer 0).
func (h *Graph[K, V]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}

	return h.layers[0].size()
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
