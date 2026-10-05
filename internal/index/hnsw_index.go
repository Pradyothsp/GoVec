package index

import (
	"bufio"
	"context"
	"encoding/gob"
	"fmt"
	"time"

	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/storage"
)

// HNSWIndex is a thread-safe vector index backed by an HNSW graph for
// approximate nearest neighbor search. T is the vector storage type ([]float32
// or []int8). Everything about storing a record is in the embedded records;
// HNSWIndex adds the graph, kept in step with the records on every insert and
// delete, and searches it.
type HNSWIndex[T hnsw.VectorType] struct {
	records[T]
	graph *hnsw.Graph[uint32, T]

	distanceFunc               func(T, T) (float32, error)
	hnswDistFunc               hnsw.DistanceFunc[T]                                         // stored for Clear() graph reset
	hnswPrecompute             func(v T) float64                                            // stored for Clear() graph reset; nil for metrics with nothing to cache
	hnswCachedDistance         func(aVec T, aCache float64, bVec T, bCache float64) float32 // stored for Clear() graph reset; nil alongside hnswPrecompute
	hnswSquaredDistance        hnsw.DistanceFunc[T]                                         // stored for Clear() graph reset; nil for metrics with no cheaper ranking equivalent
	hnswFromSquared            func(rank float32) float32                                   // stored for Clear() graph reset; nil alongside hnswSquaredDistance
	hnswM                      int                                                          // stored for Clear() graph reset
	hnswEfSearch               int                                                          // stored for Clear() graph reset
	hnswEfConstruction         int                                                          // stored for Clear() graph reset
	hnswBatchParallelism       int                                                          // stored for Clear() graph reset
	hnswBatchParallelThreshold int                                                          // stored for Clear() graph reset
}

// NewHNSWIndex creates an empty HNSWIndex ready for use.
// Pass a non-nil metaIndex to enable metadata-accelerated filtered search.
// Pass a non-nil vectorStore to enable mmap-backed vector storage.
func NewHNSWIndex[T hnsw.VectorType](
	wal *WAL,
	invertedIndex map[uint32][]core.Posting,
	idMapper *core.IDMapper,
	encodeFunc func([]float32) T,
	distanceFunc func(T, T) (float32, error),
	hnswDistFunc hnsw.DistanceFunc[T],
	// hnswPrecompute/hnswCachedDistance are the optional cache-aware pair
	// (see hnsw.Graph.Precompute's doc comment) -- nil together for a
	// metric with nothing worth caching.
	hnswPrecompute func(v T) float64,
	hnswCachedDistance func(aVec T, aCache float64, bVec T, bCache float64) float32,
	// hnswSquaredDistance/hnswFromSquared are the optional ranking-only
	// pair (see hnsw.Graph.SquaredDistance's doc comment) -- nil together
	// for a metric with no cheaper ranking equivalent.
	hnswSquaredDistance hnsw.DistanceFunc[T],
	hnswFromSquared func(rank float32) float32,
	m int,
	efSearch int,
	efConstruction int,
	metaIndex *core.MetadataIndex,
	vectorStore *storage.MmapStore,
) *HNSWIndex[T] {
	idx := &HNSWIndex[T]{
		records: records[T]{
			Store:         make(map[uint32]*core.VectorNode[T]),
			InvertedIndex: invertedIndex,
			IDMapper:      idMapper,
			metaIndex:     metaIndex,
			indexType:     "hnsw",
			wal:           wal,
			encodeFunc:    encodeFunc,
			vectorStore:   vectorStore,
		},
		distanceFunc:        distanceFunc,
		hnswDistFunc:        hnswDistFunc,
		hnswPrecompute:      hnswPrecompute,
		hnswCachedDistance:  hnswCachedDistance,
		hnswSquaredDistance: hnswSquaredDistance,
		hnswFromSquared:     hnswFromSquared,
		hnswM:               m,
		hnswEfSearch:        efSearch,
		hnswEfConstruction:  efConstruction,
	}
	idx.graph = idx.newGraph()
	return idx
}

// newGraph builds an empty graph from the index's settings.
func (idx *HNSWIndex[T]) newGraph() *hnsw.Graph[uint32, T] {
	g := hnsw.NewGraph[uint32, T]()
	g.M = idx.hnswM
	g.EfSearch = idx.hnswEfSearch
	g.EfConstruction = idx.hnswEfConstruction
	g.Distance = idx.hnswDistFunc
	g.Precompute = idx.hnswPrecompute
	g.CachedDistance = idx.hnswCachedDistance
	g.SquaredDistance = idx.hnswSquaredDistance
	g.FromSquaredDistance = idx.hnswFromSquared
	g.BatchParallelism = idx.hnswBatchParallelism
	g.BatchParallelThreshold = idx.hnswBatchParallelThreshold
	return g
}

// setBatchParallelism sets how batch inserts parallelise graph construction,
// now and after a Clear.
func (idx *HNSWIndex[T]) setBatchParallelism(parallelism, threshold int) {
	idx.hnswBatchParallelism = parallelism
	idx.hnswBatchParallelThreshold = threshold
	idx.graph.BatchParallelism = parallelism
	idx.graph.BatchParallelThreshold = threshold
}

// Insert adds or updates a vector in the index with thread-safety.
func (idx *HNSWIndex[T]) Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	return idx.insert(ctx, id, vec, sparse, meta, idx.insertInternal)
}

// BatchInsert adds or updates multiple vectors with a single WAL fsync for the
// whole batch, and a single batched idx.graph.Add call for the whole batch
// (instead of one call per item) -- this is what lets Graph.Add's
// round-based parallel Prepare (internal/hnsw's addRound) actually kick in;
// a per-item Add call, one at a time, would never give it a batch to work
// with. If writing the batch to the WAL fails, no item is applied and the
// batch aborts entirely — see WAL.WriteEntries for the atomicity this relies on.
func (idx *HNSWIndex[T]) BatchInsert(ctx context.Context, items []BatchInsertItem) ([]BatchInsertError, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if err := idx.logBatch(ctx, items); err != nil {
		return nil, err
	}

	var failures []BatchInsertError

	// pendingByID collects every stored item's graph node, keyed by
	// internalID: an intra-batch duplicate external ID always resolves to the
	// same internalID, so a repeat overwrites to last-wins. order preserves
	// first-seen position so graph.Add sees a deterministic node order.
	//
	// Each item is put in the records immediately, not deferred: a duplicate
	// ID's second occurrence must see the first's record to replace its index
	// entries. Only the graph mutation is deferred, into one graph.Add below.
	pendingByID := make(map[uint32]hnsw.Node[uint32, T], len(items))
	order := make([]uint32, 0, len(items))

	for _, item := range items {
		node, err := idx.putAndDetach(item.ID, item.Vector, item.Sparse, item.Meta)
		if err != nil {
			zerolog.Ctx(ctx).Error().Err(err).Str("id", item.ID).Msg("failed to apply batch insert item")
			failures = append(failures, BatchInsertError{ID: item.ID, Err: err})
			continue
		}

		if _, dup := pendingByID[node.InternalID]; !dup {
			order = append(order, node.InternalID)
		}
		pendingByID[node.InternalID] = hnsw.MakeNode[uint32, T](node.InternalID, node.Vector)
	}

	if len(order) == 0 {
		return failures, nil
	}

	allNodes := make([]hnsw.Node[uint32, T], len(order))
	for i, internalID := range order {
		allNodes[i] = pendingByID[internalID]
	}
	idx.graph.Add(allNodes...)

	return failures, nil
}

// insertInternal applies an insert without writing it to the WAL, used by
// Insert and by ReplayWAL.
// PRECONDITION: idx.mu.Lock() must be held by caller.
func (idx *HNSWIndex[T]) insertInternal(id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	node, err := idx.putAndDetach(id, vec, sparse, meta)
	if err != nil {
		return err
	}
	idx.graph.Add(hnsw.MakeNode[uint32, T](node.InternalID, node.Vector))
	return nil
}

// putAndDetach stores a record (records.put) and, when it replaces one, takes
// the old node out of the graph, ready for the caller's graph.Add. graph.Add's
// built-in delete-on-update operates inside the layer-iteration loop and can
// leave layers in a stale state when the only node is deleted mid-loop.
// PRECONDITION: idx.mu.Lock() held.
func (idx *HNSWIndex[T]) putAndDetach(id string, vec []float32, sparse core.SparseVector, meta map[string]any) (*core.VectorNode[T], error) {
	node, replaced, err := idx.put(id, vec, sparse, meta)
	if err != nil {
		return nil, err
	}
	if replaced {
		idx.graph.Delete(node.InternalID)
	}
	return node, nil
}

// Search finds the k nearest neighbors to the query vector using HNSW approximate search.
// Filter strategy is chosen dynamically based on selectivity:
//   - Selective filters (≤50% match): allowlist pushed into graph traversal + scaled efSearch.
//   - Non-selective filters (>50% match): post-filter with mild over-fetch.
//
// With a sparse query and hybrid search on, scores blend dense and sparse
// similarity (see records.scorer).
func (idx *HNSWIndex[T]) Search(_ context.Context, query []float32, sparseQuery core.SparseVector, k int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encodedQuery, err := idx.encodeQuery(query)
	if err != nil {
		return nil, err
	}
	if idx.graph.Len() == 0 {
		return nil, nil
	}

	allowlist, indexed := idx.filterAllowlist(filters)
	if indexed && len(allowlist) == 0 {
		return nil, nil // no documents match the filter
	}

	// Choose search strategy based on filter selectivity.
	const selectivityThreshold = 0.5
	const maxEfScale = 10
	fetchCount := k
	var graphAllowlist map[uint32]struct{}
	scaledEf := 0 // 0 → graph uses its configured default

	if indexed {
		selectivity := float64(len(allowlist)) / float64(idx.graph.Len())
		if selectivity > selectivityThreshold {
			// Non-selective: most docs match, post-filter is cheap enough.
			fetchCount = k * 2
		} else {
			// Selective: push allowlist into graph so only matching nodes reach results.
			// Scale efSearch by inverse selectivity (capped) to maintain recall.
			graphAllowlist = allowlist
			scale := 1.0 / selectivity
			if scale > maxEfScale {
				scale = maxEfScale
			}
			scaledEf = int(float64(idx.hnswEfSearch) * scale)
		}
	}

	score := idx.scorer(sparseQuery)
	candidates := idx.graph.SearchWithDistance(encodedQuery, fetchCount, graphAllowlist, scaledEf)
	results := make([]SearchResult, 0, len(candidates))
	for _, candidate := range candidates {
		node, ok := idx.Store[candidate.Key]
		if !ok || !core.MatchFilter(node.Metadata, filters) {
			continue
		}
		results = append(results, SearchResult{ID: node.ExternalID, Score: score(candidate.Key, idx.distanceToScore(candidate.Distance)), Meta: node.Metadata})
	}

	return rankTop(results, k), nil
}

// distanceToScore converts a graph distance into a "higher = closer" score, matching
// the convention core.MathBlock funcs use for the brute-force engine (see
// core.EuclideanSimilarity) so hybrid-search alpha-blending behaves consistently
// across index types. The conversion is metric-specific: cosine distance is bounded
// (~[0,2]), so 1-distance works directly; Euclidean distance is unbounded, so it's
// converted the same way core.EuclideanSimilarity is: 1/(1+distance).
func (idx *HNSWIndex[T]) distanceToScore(distance float32) float32 {
	if idx.distanceMetric == "euclidean" {
		return 1 / (1 + distance)
	}
	return 1 - distance
}

// Delete removes a vector from the index by ID.
func (idx *HNSWIndex[T]) Delete(ctx context.Context, id string) (bool, error) {
	return idx.delete(ctx, id, idx.deleteInternal)
}

// deleteInternal applies a delete without writing it to the WAL, taking the
// node out of the graph too.
// PRECONDITION: idx.mu.Lock() must be held by caller.
func (idx *HNSWIndex[T]) deleteInternal(id string) error {
	return idx.remove(id, func(internalID uint32) { idx.graph.Delete(internalID) })
}

// SaveToFile serializes the HNSW index to a snapshot file. The graph goes in
// as topology only, after the node records: every vector is already in a
// record, or in the mmap store.
func (idx *HNSWIndex[T]) SaveToFile(ctx context.Context, path string) error {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	w, err := startSnapshot(path, idx.snapshotContents())
	if err != nil {
		return err
	}

	topology := newGobChunkWriter(w.enc)
	if err := idx.graph.ExportTopology(topology); err != nil {
		w.abort()
		return fmt.Errorf("failed to export HNSW topology: %w", err)
	}
	if err := topology.Close(); err != nil {
		w.abort()
		return err
	}
	if err := w.publish(idx.wal); err != nil {
		return err
	}

	zerolog.Ctx(ctx).Info().
		Str("path", path).
		Int("vectors", idx.graph.Len()).
		Dur("duration", time.Since(start)).
		Msg("HNSW snapshot saved to disk")
	return nil
}

// LoadFromFile reads the HNSW index from a snapshot file.
// Handles both heap-backed and mmap-backed snapshots.
// A missing file is not an error — the server starts with an empty index.
func (idx *HNSWIndex[T]) LoadFromFile(ctx context.Context, path string) error {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	found, err := readSnapshotFile(path, func(dec *gob.Decoder) error {
		dims, err := readSnapshotContents(dec, idx.snapshotContents(), idx.configuredDimensions)
		if err != nil {
			return err
		}
		idx.dimensions = dims

		lookupVec := func(key uint32) (T, bool) {
			node, ok := idx.Store[key]
			if !ok {
				var zero T
				return zero, false
			}
			return node.Vector, true
		}
		topology := bufio.NewReaderSize(newGobChunkReader(dec), snapshotChunkSize)
		if err := idx.graph.ImportTopology(topology, lookupVec); err != nil {
			return fmt.Errorf("failed to import HNSW topology: %w", err)
		}
		return nil
	})
	if err != nil || !found {
		return err
	}

	idx.rebuildIndexes()

	zerolog.Ctx(ctx).Info().
		Str("path", path).
		Int("vectors", idx.graph.Len()).
		Dur("duration", time.Since(start)).
		Msg("HNSW snapshot loaded from disk")
	return nil
}

// ReplayWAL reads the WAL file and applies changes to the HNSW index.
func (idx *HNSWIndex[T]) ReplayWAL(walPath string) error {
	return idx.replay(walPath, idx.insertInternal, idx.deleteInternal)
}

// Clear removes all vectors from the HNSW index, including ID mappings and the WAL,
// recreating the graph with the same parameters.
func (idx *HNSWIndex[T]) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.graph = idx.newGraph()
	idx.clear()
}
