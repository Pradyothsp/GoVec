package hnsw

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

	for start := 0; start < len(nodes); start += workers {
		end := start + workers
		if end > len(nodes) {
			end = len(nodes)
		}

		g.addRound(nodes[start:end], efConstruction, distFn)
	}
}

// addSerial inserts nodes one at a time, each running fully through
// applyLive against the graph's live current state before the next node
// starts -- this is exactly Add()'s original, single-goroutine behavior,
// used both as the below-threshold path (see resolveParallelism) and,
// within addRound, for any item excluded from that round's parallel path
// (an upsert or an intra-round duplicate key -- see addRound's doc comment).
func (g *Graph[K, V]) addSerial(nodes []Node[K, V], efConstruction int, distFn distFunc[V]) {
	for _, node := range nodes {
		// vecCache is this node's precomputed value (see metric.Metric's
		// Precompute) -- computed once here and reused for every layer this
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

	if g.Metric.Distance == nil {
		panic("(*Graph).Metric.Distance must be set before adding nodes")
	}

	visited := g.acquireVisited()
	defer g.releaseVisited(visited)

	// If the current layer is at or below the node's insertLevel, the node
	// actually connects here: find a real candidate pool, bounded by
	// EfConstruction (not M) -- selectNeighborsHeuristic below needs a wide
	// pool to choose diverse connections from, not just the M closest.
	if insertLevel >= i {
		candidatePool := searchPoint.search(efConstruction, efConstruction, vec, distFn, vecCache, nil, visited)
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
	found := searchPoint.search(1, 1, vec, distFn, vecCache, nil, visited)
	if len(found) == 0 {
		return ptr(searchPoint.Key)
	}
	return ptr(found[0].node.Key)
}
