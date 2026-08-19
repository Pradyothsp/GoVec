package index

import (
	"bufio"
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/storage"
)

// HNSWIndex is a thread-safe vector index backed by an HNSW graph for approximate nearest neighbor search.
// T is the vector storage type ([]float32 or []int8). Dual storage is used: the HNSW graph holds
// vectors for fast search; the metadata map holds full node data (sparse vectors, external ID, metadata).
type HNSWIndex[T hnsw.VectorType] struct {
	graph         *hnsw.Graph[uint32, T]
	metadata      map[uint32]*core.VectorNode[T]
	metaIndex     *core.MetadataIndex // inverted index over metadata fields for O(1) allowlist computation
	InvertedIndex map[uint32][]core.Posting
	IDMapper      *core.IDMapper

	// Engine config metadata — set by factory after construction
	quantization   string
	indexType      string
	distanceMetric string
	dimensions     int // 0 until first insert; set lazily under mu.Lock()

	mu                  sync.RWMutex
	wal                 *WAL
	encodeFunc          func([]float32) T
	distanceFunc        func(T, T) (float32, error)
	hnswDistFunc        hnsw.DistanceFunc[T]                                         // stored for Clear() graph reset
	hnswPrecompute      func(v T) float64                                            // stored for Clear() graph reset; nil for metrics with nothing to cache
	hnswCachedDistance  func(aVec T, aCache float64, bVec T, bCache float64) float32 // stored for Clear() graph reset; nil alongside hnswPrecompute
	hnswSquaredDistance hnsw.DistanceFunc[T]                                         // stored for Clear() graph reset; nil for metrics with no cheaper ranking equivalent
	hnswFromSquared     func(rank float32) float32                                   // stored for Clear() graph reset; nil alongside hnswSquaredDistance
	hnswM               int                                                          // stored for Clear() graph reset
	hnswEfSearch        int                                                          // stored for Clear() graph reset
	hnswEfConstruction  int                                                          // stored for Clear() graph reset
	vectorStore         *storage.MmapStore                                           // nil when mmap is disabled
}

// GetByID returns a vector by ID.
func (idx *HNSWIndex[T]) GetByID(_ context.Context, id string) (*VectorRecord, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	internalID, err := idx.IDMapper.ToUint32ID(id)
	if err != nil {
		return nil, core.ErrNotFound
	}

	node, ok := idx.metadata[internalID]
	if !ok {
		return nil, core.ErrNotFound
	}

	var vec []float32
	switch v := any(node.Vector).(type) {
	case []float32:
		vec = v
	case []int8:
		vec = core.DequantizeVector(v)
	default:
		return nil, fmt.Errorf("unsupported vector type %T", node.Vector)
	}

	return &VectorRecord{
		ID:           node.ExternalID,
		Vector:       vec,
		SparseVector: node.Sparse,
		Metadata:     node.Metadata,
	}, nil
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
	g := hnsw.NewGraph[uint32, T]()
	g.M = m
	g.EfSearch = efSearch
	g.EfConstruction = efConstruction
	g.Distance = hnswDistFunc
	g.Precompute = hnswPrecompute
	g.CachedDistance = hnswCachedDistance
	g.SquaredDistance = hnswSquaredDistance
	g.FromSquaredDistance = hnswFromSquared

	return &HNSWIndex[T]{
		graph:               g,
		metadata:            make(map[uint32]*core.VectorNode[T]),
		metaIndex:           metaIndex,
		InvertedIndex:       invertedIndex,
		IDMapper:            idMapper,
		wal:                 wal,
		encodeFunc:          encodeFunc,
		distanceFunc:        distanceFunc,
		hnswDistFunc:        hnswDistFunc,
		hnswPrecompute:      hnswPrecompute,
		hnswCachedDistance:  hnswCachedDistance,
		hnswSquaredDistance: hnswSquaredDistance,
		hnswFromSquared:     hnswFromSquared,
		hnswM:               m,
		hnswEfSearch:        efSearch,
		hnswEfConstruction:  efConstruction,
		vectorStore:         vectorStore,
	}
}

// Insert adds or updates a vector in the index with thread-safety.
func (idx *HNSWIndex[T]) Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	err := idx.wal.WriteEntry(ctx, &WALEntry{
		Action: WALActionInsert,
		ID:     id,
		Vector: vec,
		Sparse: sparse,
		Meta:   meta,
	})
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to write WAL entry")
		return err
	}

	return idx.insertInternal(id, vec, sparse, meta)
}

// BatchInsert adds or updates multiple vectors with a single WAL fsync for the
// whole batch. If writing the batch to the WAL fails, no item is applied and the
// batch aborts entirely — see WAL.WriteEntries for the atomicity this relies on.
func (idx *HNSWIndex[T]) BatchInsert(ctx context.Context, items []BatchInsertItem) ([]BatchInsertError, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	entries := make([]*WALEntry, len(items))
	for i, item := range items {
		entries[i] = &WALEntry{
			Action: WALActionInsert,
			ID:     item.ID,
			Vector: item.Vector,
			Sparse: item.Sparse,
			Meta:   item.Meta,
		}
	}

	if err := idx.wal.WriteEntries(ctx, entries); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Int("count", len(items)).Msg("failed to write WAL batch")
		return nil, err
	}

	var failures []BatchInsertError
	for _, item := range items {
		if err := idx.insertInternal(item.ID, item.Vector, item.Sparse, item.Meta); err != nil {
			zerolog.Ctx(ctx).Error().Err(err).Str("id", item.ID).Msg("failed to apply batch insert item")
			failures = append(failures, BatchInsertError{ID: item.ID, Err: err})
		}
	}

	return failures, nil
}

// insertInternal performs the core insert logic without WAL writes.
// PRECONDITION: idx.mu.Lock() must be held by caller.
func (idx *HNSWIndex[T]) insertInternal(id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	internalID, err := idx.IDMapper.GetOrCreate(id)
	if err != nil {
		return err
	}

	if idx.InvertedIndex != nil {
		if existingNode, exists := idx.metadata[internalID]; exists {
			idx.removeFromInvertedIndex(internalID, existingNode.Sparse)
		}
		idx.addToInvertedIndex(internalID, sparse)
	}

	// Update MetadataIndex when enabled.
	if idx.metaIndex != nil {
		if existingNode, exists := idx.metadata[internalID]; exists {
			idx.metaIndex.Remove(internalID, existingNode.Metadata)
		}
		idx.metaIndex.Add(internalID, meta)
	}

	// Track dimensions from the first vector seen
	if idx.dimensions == 0 && len(vec) > 0 {
		idx.dimensions = len(vec)
	}

	encodeInput, err := prepareForEncode(vec, idx.quantization, idx.distanceMetric)
	if err != nil {
		return err
	}
	encoded := idx.encodeFunc(encodeInput)

	// When mmap is enabled, persist the encoded bytes and replace encoded with
	// the mmap-backed slice.  Both the graph node (Node.Value) and the metadata
	// map (VectorNode.Vector) will alias the same mmap memory — no heap copy.
	if idx.vectorStore != nil {
		mmapVec, err := putToMmapStore(idx.vectorStore, internalID, encoded)
		if err != nil {
			return fmt.Errorf("mmap store put: %w", err)
		}
		encoded = mmapVec
	}

	// For updates (re-using the same internalID), explicitly remove the node
	// from the graph before re-adding. graph.Add's built-in delete-on-update
	// operates inside the layer-iteration loop and can leave layers in a
	// stale state when the only node is deleted mid-loop.
	if _, exists := idx.metadata[internalID]; exists {
		idx.graph.Delete(internalID)
	}
	idx.graph.Add(hnsw.MakeNode[uint32, T](internalID, encoded))

	idx.metadata[internalID] = &core.VectorNode[T]{
		InternalID: internalID,
		ExternalID: id,
		Vector:     encoded,
		Sparse:     sparse,
		Metadata:   meta,
	}

	return nil
}

// Search finds the k nearest neighbors to the query vector using HNSW approximate search.
// Filter strategy is chosen dynamically based on selectivity:
//   - Selective filters (≤50% match): allowlist pushed into graph traversal + scaled efSearch.
//   - Non-selective filters (>50% match): post-filter with mild over-fetch.
//
// When hybrid search is enabled, dense and sparse scores are combined with alpha=0.7.
func (idx *HNSWIndex[T]) Search(_ context.Context, query []float32, sparseQuery core.SparseVector, k int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) == 0 {
		return nil, errors.New("empty query vector")
	}

	if idx.graph.Len() == 0 {
		return nil, nil
	}

	encodeInput, err := prepareForEncode(query, idx.quantization, idx.distanceMetric)
	if err != nil {
		return nil, err
	}
	encodedQuery := idx.encodeFunc(encodeInput)

	useHybridSearch := idx.InvertedIndex != nil && !sparseQuery.IsEmpty()
	const alpha = 0.7

	var sparseScores map[uint32]float32
	if useHybridSearch {
		sparseScores = idx.computeSparseScores(sparseQuery)
	}

	// Compute allowlist via MetadataIndex — O(1) lookup, no scanning.
	// Falls back to post-filter-only when MetadataIndex is disabled (nil).
	var allowlist map[uint32]struct{}
	if filters != nil && idx.metaIndex != nil {
		allowlist = idx.metaIndex.Allowlist(filters)
		if len(allowlist) == 0 {
			return nil, nil // short-circuit: no documents match the filter
		}
	}

	// Choose search strategy based on filter selectivity.
	const selectivityThreshold = 0.5
	const maxEfScale = 10
	fetchCount := k
	var graphAllowlist map[uint32]struct{}
	scaledEf := 0 // 0 → graph uses its configured default

	if allowlist != nil {
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

	candidates := idx.graph.SearchWithDistance(encodedQuery, fetchCount, graphAllowlist, scaledEf)
	results := make([]SearchResult, 0, len(candidates))

	for _, candidate := range candidates {
		node, ok := idx.metadata[candidate.Key]
		if !ok {
			// Graph and metadata can be briefly inconsistent during a concurrent delete.
			continue
		}

		if !core.MatchFilter(node.Metadata, filters) {
			continue
		}

		denseScore := idx.distanceToScore(candidate.Distance)

		var finalScore float32
		if useHybridSearch {
			finalScore = alpha*denseScore + (1-alpha)*sparseScores[candidate.Key]
		} else {
			finalScore = denseScore
		}
		results = append(results, SearchResult{ID: node.ExternalID, Score: finalScore, Meta: node.Metadata})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if k > 0 && len(results) > k {
		results = results[:k]
	}

	return results, nil
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
	idx.mu.Lock()
	defer idx.mu.Unlock()

	internalID, err := idx.IDMapper.ToUint32ID(id)
	if err != nil {
		return false, nil
	}

	_, exists := idx.metadata[internalID]
	if !exists {
		return false, nil
	}

	err = idx.wal.WriteEntry(ctx, &WALEntry{
		Action: WALActionDelete,
		ID:     id,
	})
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to write WAL entry")
		return false, err
	}

	if err := idx.deleteInternal(id); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to delete vector")
		return false, err
	}

	return true, nil
}

// deleteInternal performs the core delete logic without WAL writes.
// PRECONDITION: idx.mu.Lock() must be held by caller.
func (idx *HNSWIndex[T]) deleteInternal(id string) error {
	internalID, err := idx.IDMapper.ToUint32ID(id)
	if err != nil {
		return core.ErrNotFound
	}

	node, exists := idx.metadata[internalID]
	if !exists {
		return core.ErrNotFound
	}

	if idx.InvertedIndex != nil {
		idx.removeFromInvertedIndex(internalID, node.Sparse)
	}

	idx.graph.Delete(internalID)
	delete(idx.metadata, internalID)
	if idx.metaIndex != nil {
		idx.metaIndex.Remove(internalID, node.Metadata)
	}

	return idx.IDMapper.Delete(id)
}

// SaveToFile serializes the HNSW index to a snapshot file.
// When mmap is enabled it writes version 6 (topology-only graph; no inline vectors).
// Otherwise it writes the standard version 4 format.
func (idx *HNSWIndex[T]) SaveToFile(ctx context.Context, path string) error {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	var quantType string
	switch any(*new(T)).(type) {
	case []float32:
		quantType = "none"
	case []int8:
		quantType = "scalar"
	default:
		return fmt.Errorf("unknown vector type")
	}

	var err error
	if idx.vectorStore != nil {
		err = idx.saveToFileMmap(path, quantType)
	} else {
		err = idx.saveToFileHeap(path, quantType)
	}

	if err == nil {
		zerolog.Ctx(ctx).Info().
			Str("path", path).
			Int("vectors", idx.graph.Len()).
			Dur("duration", time.Since(start)).
			Msg("HNSW snapshot saved to disk")
	}
	return err
}

// saveToFileHeap writes the standard v4 snapshot (inline vectors embedded in GOB).
// PRECONDITION: idx.mu.Lock() held.
func (idx *HNSWIndex[T]) saveToFileHeap(path, quantType string) error {
	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			return err
		}
	}

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path from operator config
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(f)

	header := SnapshotHeader{
		Version:             4,
		Quantization:        quantType,
		DistanceMetric:      "cosine",
		HybridSearchEnabled: idx.InvertedIndex != nil,
		IndexType:           "hnsw",
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := idx.IDMapper.EncodeGOB(encoder); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := encoder.Encode(idx.metadata); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if idx.InvertedIndex != nil {
		if err := encoder.Encode(idx.InvertedIndex); err != nil {
			_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			return err
		}
	}

	var graphBuf bytes.Buffer
	if err := idx.graph.Export(&graphBuf); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return fmt.Errorf("failed to export HNSW graph: %w", err)
	}
	if err := encoder.Encode(graphBuf.Bytes()); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}

	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	return nil
}

// saveToFileMmap writes a v6 snapshot: topology-only graph bytes + vectorNodeSnapshot list.
// PRECONDITION: idx.mu.Lock() held.
func (idx *HNSWIndex[T]) saveToFileMmap(path, quantType string) error {
	if err := idx.vectorStore.Sync(); err != nil {
		return fmt.Errorf("mmap sync: %w", err)
	}
	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			return err
		}
	}

	// Build vector-free snapshots.
	snapshots := make([]vectorNodeSnapshot, 0, len(idx.metadata))
	for _, node := range idx.metadata {
		snapshots = append(snapshots, vectorNodeSnapshot{
			InternalID: node.InternalID,
			ExternalID: node.ExternalID,
			Sparse:     node.Sparse,
			Metadata:   node.Metadata,
		})
	}

	// Topology-only graph bytes (no vector data).
	var topoBytes bytes.Buffer
	if err := idx.graph.ExportTopology(&topoBytes); err != nil {
		return fmt.Errorf("failed to export HNSW topology: %w", err)
	}

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path from operator config
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(f)
	header := SnapshotHeader{
		Version:             6,
		Quantization:        quantType,
		DistanceMetric:      "cosine",
		HybridSearchEnabled: idx.InvertedIndex != nil,
		IndexType:           "hnsw",
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := idx.IDMapper.EncodeGOB(encoder); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := encoder.Encode(snapshots); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if idx.InvertedIndex != nil {
		if err := encoder.Encode(idx.InvertedIndex); err != nil {
			_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			return err
		}
	}
	if err := encoder.Encode(topoBytes.Bytes()); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	return nil
}

// LoadFromFile reads the HNSW index from a snapshot file.
// Supports version 4 (heap-backed vectors) and version 6 (mmap-backed vectors).
// A missing file is not an error — the server starts with an empty index.
func (idx *HNSWIndex[T]) LoadFromFile(ctx context.Context, path string) (err error) {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path) //nolint:gosec // path from operator config
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	decoder := gob.NewDecoder(f)

	var header SnapshotHeader
	if err := decoder.Decode(&header); err != nil {
		return fmt.Errorf("failed to decode snapshot header: %w", err)
	}

	if header.IndexType != "hnsw" {
		return fmt.Errorf("snapshot was created with brute-force index (version %d). Delete data files or change index_type to 'brute' in config", header.Version)
	}
	if header.Version != 4 && header.Version != 6 {
		return fmt.Errorf("unsupported HNSW snapshot version %d (expected 4 or 6)", header.Version)
	}

	var expectedQuant string
	switch any(*new(T)).(type) {
	case []float32:
		expectedQuant = "none"
	case []int8:
		expectedQuant = "scalar"
	default:
		return fmt.Errorf("unknown vector type")
	}
	if header.Quantization != expectedQuant {
		return fmt.Errorf("snapshot quantization mismatch: config expects '%s' but snapshot is '%s'. Delete data files or change config", expectedQuant, header.Quantization)
	}

	if err := idx.IDMapper.DecodeGOB(decoder); err != nil {
		return fmt.Errorf("failed to decode IDMapper: %w", err)
	}

	if header.Version == 6 {
		err = idx.loadFromFileMmap(decoder, &header)
	} else {
		// Standard v4: metadata map contains inline vectors.
		if err := decoder.Decode(&idx.metadata); err != nil {
			return fmt.Errorf("failed to decode metadata: %w", err)
		}

		if header.HybridSearchEnabled {
			if idx.InvertedIndex != nil {
				var loadedIndex map[uint32][]core.Posting
				if err := decoder.Decode(&loadedIndex); err != nil {
					return fmt.Errorf("failed to decode inverted index: %w", err)
				}
				idx.InvertedIndex = loadedIndex
			} else {
				var discarded map[uint32][]core.Posting
				if err := decoder.Decode(&discarded); err != nil {
					return fmt.Errorf("failed to skip inverted index: %w", err)
				}
			}
		}

		var graphBytes []byte
		if err := decoder.Decode(&graphBytes); err != nil {
			return fmt.Errorf("failed to decode HNSW graph bytes: %w", err)
		}
		if err := idx.graph.Import(bufio.NewReader(bytes.NewReader(graphBytes))); err != nil {
			return fmt.Errorf("failed to import HNSW graph: %w", err)
		}
	}

	if idx.metaIndex != nil {
		idx.rebuildMetaIndex()
	}

	if err == nil {
		zerolog.Ctx(ctx).Info().
			Str("path", path).
			Int("vectors", idx.graph.Len()).
			Dur("duration", time.Since(start)).
			Msg("HNSW snapshot loaded from disk")
	}

	return err
}

// loadFromFileMmap reconstructs the index from a v6 mmap-backed snapshot.
// PRECONDITION: idx.mu.Lock() held; IDMapper already decoded.
func (idx *HNSWIndex[T]) loadFromFileMmap(decoder *gob.Decoder, header *SnapshotHeader) error {
	if idx.vectorStore == nil {
		return fmt.Errorf("snapshot is mmap format (v6) but mmap storage is not configured")
	}

	var snapshots []vectorNodeSnapshot
	if err := decoder.Decode(&snapshots); err != nil {
		return fmt.Errorf("failed to decode mmap snapshots: %w", err)
	}

	// Reconstruct metadata with mmap-backed vectors.
	for _, snap := range snapshots {
		vec, ok := getFromMmapStore[T](idx.vectorStore, snap.InternalID)
		if !ok {
			return fmt.Errorf("mmap slot missing for internalID %d (externalID %q)", snap.InternalID, snap.ExternalID)
		}
		idx.metadata[snap.InternalID] = &core.VectorNode[T]{
			InternalID: snap.InternalID,
			ExternalID: snap.ExternalID,
			Vector:     vec,
			Sparse:     snap.Sparse,
			Metadata:   snap.Metadata,
		}
	}

	if header.HybridSearchEnabled {
		if idx.InvertedIndex != nil {
			var loadedIndex map[uint32][]core.Posting
			if err := decoder.Decode(&loadedIndex); err != nil {
				return fmt.Errorf("failed to decode inverted index: %w", err)
			}
			idx.InvertedIndex = loadedIndex
		} else {
			var discarded map[uint32][]core.Posting
			if err := decoder.Decode(&discarded); err != nil {
				return fmt.Errorf("failed to skip inverted index: %w", err)
			}
		}
	}

	// Decode topology bytes and import graph with mmap-backed vectors.
	var topoBytes []byte
	if err := decoder.Decode(&topoBytes); err != nil {
		return fmt.Errorf("failed to decode HNSW topology bytes: %w", err)
	}

	lookupVec := func(key uint32) (T, bool) {
		node, ok := idx.metadata[key]
		if !ok {
			var zero T
			return zero, false
		}
		return node.Vector, true
	}
	if err := idx.graph.ImportTopology(bufio.NewReader(bytes.NewReader(topoBytes)), lookupVec); err != nil {
		return fmt.Errorf("failed to import HNSW topology: %w", err)
	}

	if idx.metaIndex != nil {
		idx.rebuildMetaIndex()
	}
	return nil
}

// rebuildMetaIndex reconstructs the MetadataIndex from the current metadata map.
// Called after LoadFromFile since MetadataIndex is a derived, in-memory structure.
// PRECONDITION: idx.mu.Lock() must be held by caller. idx.metaIndex must not be nil.
func (idx *HNSWIndex[T]) rebuildMetaIndex() {
	idx.metaIndex.Clear()
	for internalID, node := range idx.metadata {
		idx.metaIndex.Add(internalID, node.Metadata)
	}
}

// ReplayWAL reads the WAL file and applies changes to the HNSW index.
func (idx *HNSWIndex[T]) ReplayWAL(filepath string) error {
	f, err := os.Open(filepath) //nolint:gosec // filepath comes from config, not user input
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close() //nolint:errcheck // read-only, close error not critical

	scanner := bufio.NewScanner(f)
	// Increase buffer to 4MB to handle large vector WAL entries (default 64KB is too small)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	count := 0
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		var entry WALEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			log.Warn().Int("line", lineNum).Err(err).Msg("skipping malformed WAL entry")
			continue
		}

		switch entry.Action {
		case WALActionInsert:
			idx.mu.Lock()
			if err := idx.insertInternal(entry.ID, entry.Vector, entry.Sparse, entry.Meta); err != nil {
				log.Warn().Str("id", entry.ID).Int("line", lineNum).Err(err).Msg("failed to replay insert")
				idx.mu.Unlock()
				continue
			}
			idx.mu.Unlock()

		case WALActionDelete:
			idx.mu.Lock()
			if err := idx.deleteInternal(entry.ID); err != nil {
				if !errors.Is(err, core.ErrNotFound) {
					log.Warn().Str("id", entry.ID).Int("line", lineNum).Err(err).Msg("failed to replay delete")
				}
				idx.mu.Unlock()
				continue
			}
			idx.mu.Unlock()
		}
		count++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("WAL scanner error at line %d: %w", lineNum, err)
	}

	log.Info().Int("count", count).Msg("WAL replay complete")
	return nil
}

// Clear removes all vectors from the HNSW index, including ID mappings and the WAL,
// recreating the graph with the same parameters.
func (idx *HNSWIndex[T]) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	g := hnsw.NewGraph[uint32, T]()
	g.M = idx.hnswM
	g.EfSearch = idx.hnswEfSearch
	g.EfConstruction = idx.hnswEfConstruction
	g.Distance = idx.hnswDistFunc
	g.Precompute = idx.hnswPrecompute
	g.CachedDistance = idx.hnswCachedDistance
	g.SquaredDistance = idx.hnswSquaredDistance
	g.FromSquaredDistance = idx.hnswFromSquared
	idx.graph = g

	idx.metadata = make(map[uint32]*core.VectorNode[T])
	if idx.metaIndex != nil {
		idx.metaIndex.Clear()
	}
	if idx.InvertedIndex != nil {
		idx.InvertedIndex = make(map[uint32][]core.Posting)
	}
	if idx.vectorStore != nil {
		if err := idx.vectorStore.Reset(); err != nil {
			// Log but don't propagate — in-memory state is already reset.
			log.Error().Err(err).Msg("failed to reset mmap vector store")
		}
	}
	if idx.IDMapper != nil {
		idx.IDMapper.Clear()
	}
	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			log.Error().Err(err).Msg("failed to clear WAL")
		}
	}
}

// Len returns the number of vectors in the HNSW index.
func (idx *HNSWIndex[T]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Len()
}

// Info returns engine configuration and runtime statistics.
// NOTE: reads len(idx.metadata) directly to avoid re-acquiring mu (not re-entrant).
func (idx *HNSWIndex[T]) Info() EngineInfo {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return EngineInfo{
		Quantization:   idx.quantization,
		IndexType:      idx.indexType,
		DistanceMetric: idx.distanceMetric,
		Dimensions:     idx.dimensions,
		VectorCount:    len(idx.metadata),
	}
}

// computeSparseScores computes sparse similarity scores for all documents.
// MUST be called with idx.mu.RLock() held.
func (idx *HNSWIndex[T]) computeSparseScores(sparseQuery core.SparseVector) map[uint32]float32 {
	scores := make(map[uint32]float32)
	for id, node := range idx.metadata {
		if !node.Sparse.IsEmpty() {
			score := core.SparseDotProduct(sparseQuery, node.Sparse)
			if score > 0 {
				scores[id] = score
			}
		}
	}
	return scores
}

// addToInvertedIndex adds postings for a document's sparse vector.
// MUST be called with idx.mu.Lock() held.
func (idx *HNSWIndex[T]) addToInvertedIndex(docID uint32, sparse core.SparseVector) {
	if sparse.IsEmpty() {
		return
	}
	for i := 0; i < len(sparse.Indices); i++ {
		tokenID := sparse.Indices[i]
		weight := sparse.Values[i]
		idx.InvertedIndex[tokenID] = append(idx.InvertedIndex[tokenID], core.Posting{
			DocID:  docID,
			Weight: weight,
		})
	}
}

// removeFromInvertedIndex removes all postings for a document.
// MUST be called with idx.mu.Lock() held.
func (idx *HNSWIndex[T]) removeFromInvertedIndex(docID uint32, sparse core.SparseVector) {
	if sparse.IsEmpty() {
		return
	}
	for _, tokenID := range sparse.Indices {
		postings := idx.InvertedIndex[tokenID]
		filtered := make([]core.Posting, 0, len(postings))
		for _, posting := range postings {
			if posting.DocID != docID {
				filtered = append(filtered, posting)
			}
		}
		if len(filtered) == 0 {
			delete(idx.InvertedIndex, tokenID)
		} else {
			idx.InvertedIndex[tokenID] = filtered
		}
	}
}
