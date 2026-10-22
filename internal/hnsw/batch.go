package hnsw

import (
	"runtime"
	"sync"
)

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
type planStep[K Key, V VectorType] struct {
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
type insertPlan[K Key, V VectorType] struct {
	node        Node[K, V]
	vecCache    float64
	insertLevel int
	steps       []planStep[K, V] // top-down: highest layer first
}

// prepareInsert computes node's full top-down insert plan by walking the
// graph exactly as applyLive does, but read-only: every mutation applyLive
// would perform (layer.nodes writes, addNeighbor calls, Delete) is replaced
// here with plan-step construction instead. Safe to call concurrently with
// other prepareInsert calls for the same round -- each only reads
// g.layers/layer.nodes/layerNode.neighbors/.Value/.precomputed/.Key, and
// takes its own visited set from the pool.
func (g *Graph[K, V]) prepareInsert(
	node Node[K, V], insertLevel int, vecCache float64,
	efConstruction int, distFn distFunc[V],
) *insertPlan[K, V] {
	key := node.Key
	vec := node.Value

	visited := g.acquireVisited()
	defer g.releaseVisited(visited)

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

		if g.Metric.Distance == nil {
			panic("(*Graph).Metric.Distance must be set before adding nodes")
		}

		if insertLevel >= i {
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
func (g *Graph[K, V]) addRound(nodes []Node[K, V], efConstruction int, distFn distFunc[V]) {
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
			plans[slot] = g.prepareInsert(nodes[idx], insertLevels[idx], vecCaches[idx], efConstruction, distFn)
		}(slot, idx)
	}

	wg.Wait() // The barrier: nothing below runs until every Prepare goroutine above has returned.

	for _, plan := range plans {
		g.applyInsert(plan, efConstruction, distFn)
	}
}
