package hnsw

import (
	"github.com/Pradyothsp/govec/internal/metric"
)

// distFunc computes the distance between two vectors, given each one's
// precomputed cache value (0 and ignored when the configured metric has no
// cache -- see metric.Metric's Precompute). It replaces passing a bare
// metric.DistanceFunc[V] to search()/addNeighbor()/replenish()/
// selectNeighborsHeuristic() below: those need each side's cache, not just
// its vector. It is purely internal plumbing built fresh (via distFuncFor)
// for each top-level Add()/search() call.
type distFunc[V VectorType] func(aVec V, aCache float64, bVec V, bCache float64) float32

// distFuncFor builds this graph's distFunc[V]: CachedDistance when the
// configured metric opted in, or a thin wrapper around Distance (ignoring
// both cache arguments) otherwise. Call once per top-level Add()/search()
// call and thread the result down, rather than re-branching on every
// comparison.
func (g *Graph[K, V]) distFuncFor() distFunc[V] {
	if g.Metric.CachedDistance != nil {
		return distFunc[V](g.Metric.CachedDistance)
	}

	return noCacheDist(g.Metric.Distance)
}

// noCacheDist adapts a plain DistanceFunc[V] into a distFunc[V] that ignores
// both cache arguments -- the shape every internal call site needs, for a
// metric that never populates a cache. Shared by distFuncFor's fallback
// branch and any caller (e.g. tests) that wants to drive search()/
// addNeighbor()/etc. directly with a bare DistanceFunc[V] and no caching.
func noCacheDist[V VectorType](dist metric.DistanceFunc[V]) distFunc[V] {
	return func(aVec V, _ float64, bVec V, _ float64) float32 {
		return dist(aVec, bVec)
	}
}

// precompute returns g.Metric.Precompute(v), or 0 if the configured metric has no
// cache (Precompute == nil) -- safe because distFuncFor's non-cached branch
// ignores the cache arguments entirely in that case.
func (g *Graph[K, V]) precompute(v V) float64 {
	if g.Metric.Precompute == nil {
		return 0
	}

	return g.Metric.Precompute(v)
}

// rankDistFuncFor builds a distFunc[V] for internal ranking/comparison use
// only -- values it produces must never reach a caller as-is; see
// metric.Metric's SquaredDistance. Prefers SquaredDistance when the
// metric opted in (cheaper: skips a monotonic transform every comparison
// would otherwise pay for); falls back to distFuncFor's existing behavior
// (CachedDistance, or plain Distance) when it hasn't. Callers whose
// distance values never leave the graph (Add, Delete, and everything they
// call) should always use this instead of distFuncFor -- only
// Graph.search()'s base-layer result needs the real distance, converted
// back via fromRank at that single boundary.
func (g *Graph[K, V]) rankDistFuncFor() distFunc[V] {
	if g.Metric.SquaredDistance != nil {
		return noCacheDist(g.Metric.SquaredDistance)
	}

	return g.distFuncFor()
}

// fromRank converts a rankDistFuncFor()-produced value back into the real,
// externally-correct distance -- the identity function when the configured
// metric has no cheaper ranking equivalent (SquaredDistance == nil, so
// rankDistFuncFor never substituted anything in the first place).
func (g *Graph[K, V]) fromRank(v float32) float32 {
	if g.Metric.FromSquaredDistance == nil {
		return v
	}

	return g.Metric.FromSquaredDistance(v)
}
