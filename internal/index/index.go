package index

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/storage"
)

// VectorIndex is a thread-safe in-memory store for vector embeddings.
// T is the vector storage type (e.g., []float32 for no quantization, []int8 for scalar quantization)
type VectorIndex[T any] struct {
	Store         map[uint32]*core.VectorNode[T] // Changed from map[string] to map[uint32]
	InvertedIndex map[uint32][]core.Posting
	IDMapper      *core.IDMapper      // Translates string ↔ uint32 IDs
	metaIndex     *core.MetadataIndex // nil when metadata index is disabled

	// Engine config metadata — set by factory after construction
	quantization   string
	indexType      string
	distanceMetric string
	dimensions     int // 0 until first insert; set lazily under mu.Lock()
	// configuredDimensions is engine.dimensions from config, 0 when unset.
	// Kept apart from dimensions so Clear can tell a width the operator chose
	// from one the index happened to learn, and release only the latter.
	configuredDimensions int

	mu           sync.RWMutex
	wal          *WAL
	encodeFunc   func([]float32) T           // Converts input vectors to storage format
	distanceFunc func(T, T) (float32, error) // Computes distance between stored vectors
	vectorStore  *storage.MmapStore          // nil when mmap is disabled
}

// GetByID retrieves a VectorRecord by its external ID, returning an error if the ID does not exist or lookup fails.
func (idx *VectorIndex[T]) GetByID(_ context.Context, id string) (*VectorRecord, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	internalID, err := idx.IDMapper.ToUint32ID(id)
	if err != nil {
		return nil, core.ErrNotFound
	}

	node, ok := idx.Store[internalID]
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
		SparseVector: sparseOrNil(node.Sparse),
		Metadata:     node.Metadata,
	}, nil
}

// NewVectorIndex creates an empty VectorIndex ready for use.
// Pass a non-nil vectorStore to enable mmap-backed vector storage.
func NewVectorIndex[T any](wal *WAL, invertedIndex map[uint32][]core.Posting, idMapper *core.IDMapper, encodeFunc func([]float32) T, distanceFunc func(T, T) (float32, error), vectorStore *storage.MmapStore) *VectorIndex[T] {
	return &VectorIndex[T]{
		Store:         make(map[uint32]*core.VectorNode[T]),
		InvertedIndex: invertedIndex,
		IDMapper:      idMapper,
		wal:           wal,
		encodeFunc:    encodeFunc,
		distanceFunc:  distanceFunc,
		vectorStore:   vectorStore,
	}
}

// Insert adds or updates a vector in the index with thread-safety
func (idx *VectorIndex[T]) Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// 1. Write to WAL FIRST (write-ahead guarantee)
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

	// 2. Execute core insert logic
	err = idx.insertInternal(id, vec, sparse, meta)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to insert vector")
		return err
	}

	return nil
}

// BatchInsert adds or updates multiple vectors with a single WAL fsync for the
// whole batch. If writing the batch to the WAL fails, no item is applied and the
// batch aborts entirely — see WAL.WriteEntries for the atomicity this relies on.
func (idx *VectorIndex[T]) BatchInsert(ctx context.Context, items []BatchInsertItem) ([]BatchInsertError, error) {
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
func (idx *VectorIndex[T]) insertInternal(id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	// 0. Reject a vector that can't be compared against the rest of the index.
	//    This runs before any mutation below -- GetOrCreate allocates a
	//    permanent internal ID, and the index updates that follow are not
	//    rolled back on a later error.
	if err := validateVectorDims(vec, idx.dimensions); err != nil {
		return err
	}

	// 1. Translate string ID → uint32 ID (create if doesn't exist)
	internalID, err := idx.IDMapper.GetOrCreate(id)
	if err != nil {
		return err
	}

	// 2. If hybrid search is enabled, update inverted index
	if idx.InvertedIndex != nil {
		// Remove old postings if document exists (update case)
		if existingNode, exists := idx.Store[internalID]; exists {
			idx.removeFromInvertedIndex(internalID, existingNode.Sparse)
		}
		idx.addToInvertedIndex(internalID, sparse)
	}

	// 3. If metadata index is enabled, keep it in sync
	if idx.metaIndex != nil {
		if existingNode, exists := idx.Store[internalID]; exists {
			idx.metaIndex.Remove(internalID, existingNode.Metadata)
		}
		idx.metaIndex.Add(internalID, meta)
	}

	// 4. Encode vector; if mmap is enabled, write to the mmap store and store the
	//    mmap-backed slice — no heap copy is retained after this point.
	encodeInput, err := prepareForEncode(vec, idx.quantization, idx.distanceMetric)
	if err != nil {
		return err
	}
	encoded := idx.encodeFunc(encodeInput)
	if idx.vectorStore != nil {
		mmapVec, err := putToMmapStore(idx.vectorStore, internalID, encoded)
		if err != nil {
			return fmt.Errorf("mmap store put: %w", err)
		}
		encoded = mmapVec
	}

	// 5. Track dimensions from the first vector seen
	if idx.dimensions == 0 && len(vec) > 0 {
		idx.dimensions = len(vec)
	}

	// 6. Update store
	idx.Store[internalID] = &core.VectorNode[T]{
		InternalID: internalID,
		ExternalID: id,
		Vector:     encoded,
		Sparse:     sparse,
		Metadata:   meta,
	}

	return nil
}

// Search returns the top-limit nearest neighbours to query using the configured distance metric.
// If sparseQuery is provided and hybrid search is enabled, it performs hybrid search.
// Hybrid search combines dense (vector) and sparse (keyword) scores using the formula:
//
//	final_score = alpha * dense_score + (1 - alpha) * sparse_score
//
// where alpha = 0.7 (default weighting: 70% semantic, 30% keyword)
func (idx *VectorIndex[T]) Search(_ context.Context, query []float32, sparseQuery core.SparseVector, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// A query of the wrong width can't be compared against anything stored,
	// and the comparison functions report that as a bare "vector dimensions
	// mismatch" from deep inside the scan. Checking here names the required
	// width, classifies the failure as the caller's, and skips the scan.
	if err := validateVectorDims(query, idx.dimensions); err != nil {
		return nil, err
	}

	// Encode query to storage format
	encodeInput, err := prepareForEncode(query, idx.quantization, idx.distanceMetric)
	if err != nil {
		return nil, err
	}
	encodedQuery := idx.encodeFunc(encodeInput)

	// Determine if we're doing hybrid search
	useHybridSearch := idx.InvertedIndex != nil && !sparseQuery.IsEmpty()

	// Alpha parameter for hybrid search (0.7 = 70% dense, 30% sparse)
	const alpha = 0.7

	// Pre-compute sparse scores if doing hybrid search
	var sparseScores map[uint32]float32 // Changed to uint32 keys
	if useHybridSearch {
		sparseScores = idx.computeSparseScores(sparseQuery)
	}

	results := make([]SearchResult, 0, limit)

	var candidates []*core.VectorNode[T]
	if idx.metaIndex != nil && filters != nil {
		allowlist := idx.metaIndex.Allowlist(filters)
		if len(allowlist) == 0 {
			return nil, nil
		}
		candidates = make([]*core.VectorNode[T], 0, len(allowlist))
		for id := range allowlist {
			if node, ok := idx.Store[id]; ok {
				candidates = append(candidates, node)
			}
		}
	} else {
		candidates = make([]*core.VectorNode[T], 0, len(idx.Store))
		for _, node := range idx.Store {
			if core.MatchFilter(node.Metadata, filters) {
				candidates = append(candidates, node)
			}
		}
	}

	for _, node := range candidates {
		denseScore, err := idx.distanceFunc(encodedQuery, node.Vector)
		if err != nil {
			return nil, err
		}

		var finalScore float32
		if useHybridSearch {
			finalScore = alpha*denseScore + (1-alpha)*sparseScores[node.InternalID]
		} else {
			finalScore = denseScore
		}

		results = append(results, SearchResult{
			ID:    node.ExternalID,
			Score: finalScore,
			Meta:  node.Metadata,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// computeSparseScores computes sparse similarity scores for all documents using the inverted index.
// Uses the stored sparse vectors and computes dot product with the query.
// MUST be called with idx.mu.RLock() held.
func (idx *VectorIndex[T]) computeSparseScores(sparseQuery core.SparseVector) map[uint32]float32 {
	scores := make(map[uint32]float32) // Changed to uint32 keys

	// For each document in the store, compute sparse dot product
	for id, node := range idx.Store {
		if !node.Sparse.IsEmpty() {
			score := core.SparseDotProduct(sparseQuery, node.Sparse)
			if score > 0 {
				scores[id] = score // id is now uint32
			}
		}
	}

	return scores
}

// Delete removes a vector from the index by ID
func (idx *VectorIndex[T]) Delete(ctx context.Context, id string) (bool, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// 1. Check if vector exists (needed for bool return value)
	internalID, err := idx.IDMapper.ToUint32ID(id)
	if err != nil {
		return false, nil
	}

	_, exists := idx.Store[internalID]
	if !exists {
		return false, nil
	}

	// 2. Write to WAL
	err = idx.wal.WriteEntry(ctx, &WALEntry{
		Action: WALActionDelete,
		ID:     id,
	})
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to write WAL entry")
		return false, err
	}

	// 3. Execute core delete logic
	err = idx.deleteInternal(id)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to delete vector")
		return false, err
	}

	return true, nil
}

// deleteInternal performs the core delete logic without WAL writes.
// PRECONDITION: idx.mu.Lock() must be held by caller.
// Returns core.ErrNotFound if ID doesn't exist.
func (idx *VectorIndex[T]) deleteInternal(id string) error {
	// 1. Translate string ID → uint32 ID
	internalID, err := idx.IDMapper.ToUint32ID(id)
	if err != nil {
		return core.ErrNotFound
	}

	// 2. Check if vector exists in store
	node, exists := idx.Store[internalID]
	if !exists {
		return core.ErrNotFound
	}

	// 3. If hybrid search enabled, remove from inverted index
	if idx.InvertedIndex != nil {
		idx.removeFromInvertedIndex(internalID, node.Sparse)
	}

	// 4. If metadata index is enabled, remove from it
	if idx.metaIndex != nil {
		idx.metaIndex.Remove(internalID, node.Metadata)
	}

	// 5. Delete from store
	delete(idx.Store, internalID)

	// 5. Mark as tombstone in IDMapper
	err = idx.IDMapper.Delete(id)
	if err != nil {
		return err
	}

	return nil
}

// Clear removes all vectors from the index, including ID mappings and the WAL.
func (idx *VectorIndex[T]) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.Store = make(map[uint32]*core.VectorNode[T])

	if idx.InvertedIndex != nil {
		idx.InvertedIndex = make(map[uint32][]core.Posting)
	}

	if idx.metaIndex != nil {
		idx.metaIndex.Clear()
	}

	if idx.vectorStore != nil {
		if err := idx.vectorStore.Reset(); err != nil {
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

	// An empty index has no width to enforce. Keeping a learned one here left
	// a reset index rejecting every vector of a different size -- so clearing
	// to start over with different embeddings needed a server restart. A
	// configured width is the operator's choice and survives.
	idx.dimensions = idx.configuredDimensions
}

// Len returns the number of vectors in the index
func (idx *VectorIndex[T]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.Store)
}

// Info returns engine configuration and runtime statistics.
// NOTE: reads len(idx.Store) directly to avoid re-acquiring mu (not re-entrant).
func (idx *VectorIndex[T]) Info() EngineInfo {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return EngineInfo{
		Quantization:   idx.quantization,
		IndexType:      idx.indexType,
		DistanceMetric: idx.distanceMetric,
		Dimensions:     idx.dimensions,
		VectorCount:    len(idx.Store),
		EnableMmap:     idx.vectorStore != nil,
	}
}

// addToInvertedIndex adds postings for a document's sparse vector to the inverted index.
// MUST be called with idx.mu.Lock() held.
func (idx *VectorIndex[T]) addToInvertedIndex(docID uint32, sparse core.SparseVector) {
	if sparse.IsEmpty() {
		return // No sparse vector, nothing to add
	}

	// For each token in the sparse vector, add a posting
	for i := 0; i < len(sparse.Indices); i++ {
		tokenID := sparse.Indices[i]
		weight := sparse.Values[i]

		// Append posting to the token's posting list
		idx.InvertedIndex[tokenID] = append(idx.InvertedIndex[tokenID], core.Posting{
			DocID:  docID, // Now uses uint32 DocID
			Weight: weight,
		})
	}
}

// rebuildInvertedIndex recomputes every posting list from the sparse vectors
// in idx.Store, discarding whatever the inverted index held before.
// MUST be called with idx.mu.Lock() held.
func (idx *VectorIndex[T]) rebuildInvertedIndex() {
	if idx.InvertedIndex == nil {
		return // Hybrid search disabled
	}

	// A fresh map rather than clear(): nothing outside the index should hold
	// the old one, but a new map makes that not matter.
	idx.InvertedIndex = make(map[uint32][]core.Posting)
	for internalID, node := range idx.Store {
		idx.addToInvertedIndex(internalID, node.Sparse)
	}
}

// removeFromInvertedIndex removes all postings for a document from the inverted index.
// MUST be called with idx.mu.Lock() held.
func (idx *VectorIndex[T]) removeFromInvertedIndex(docID uint32, sparse core.SparseVector) {
	if sparse.IsEmpty() {
		return // No sparse vector, nothing to remove
	}

	// For each token in the old sparse vector, remove the posting
	for _, tokenID := range sparse.Indices {
		postings := idx.InvertedIndex[tokenID]

		// Filter out the posting for this document
		filtered := make([]core.Posting, 0, len(postings))
		for _, posting := range postings {
			if posting.DocID != docID { // Now compares uint32 IDs
				filtered = append(filtered, posting)
			}
		}

		// Update the posting list (or delete if empty)
		if len(filtered) == 0 {
			delete(idx.InvertedIndex, tokenID)
		} else {
			idx.InvertedIndex[tokenID] = filtered
		}
	}
}
