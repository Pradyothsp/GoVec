package index

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/storage"
)

// records is what both engines keep about the vectors they hold, and every
// rule for keeping it consistent: the ID mapping, the stored nodes, the
// metadata and inverted indexes, the index's width, encoding and mmap storage,
// the WAL-first write order, and how search results are scored and ranked.
// Each engine embeds it and adds only how it searches: VectorIndex scans every
// node, HNSWIndex walks a graph it keeps in step through put and remove.
//
// These rules used to be written out in both engines, and the copies drifted:
// HNSW learned the index's width from a vector it then rejected, and its Info
// never reported mmap.
type records[T any] struct {
	Store         map[uint32]*core.VectorNode[T]
	InvertedIndex map[uint32][]core.Posting // nil when hybrid search is off
	IDMapper      *core.IDMapper
	metaIndex     *core.MetadataIndex // nil when the metadata index is off

	quantization   string
	indexType      string
	distanceMetric string
	dimensions     int // 0 until set by config or the first insert
	// configuredDimensions is engine.dimensions from config, 0 when unset.
	// Kept apart from dimensions so clear can tell a width the operator chose
	// from one the index happened to learn, and release only the latter.
	configuredDimensions int

	mu          sync.RWMutex
	wal         *WAL
	encodeFunc  func([]float32) T  // converts input vectors to the stored type
	vectorStore *storage.MmapStore // nil when mmap is disabled
}

// configure applies the engine settings from config.
func (r *records[T]) configure(cfg *config.EngineConfig) {
	r.quantization = string(cfg.Quantization)
	r.distanceMetric = string(cfg.DistanceMetric)
	r.dimensions = cfg.Dimensions
	r.configuredDimensions = cfg.Dimensions
}

// GetByID retrieves a VectorRecord by its external ID, returning
// core.ErrNotFound if there is none.
func (r *records[T]) GetByID(_ context.Context, id string) (*VectorRecord, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	internalID, err := r.IDMapper.ToUint32ID(id)
	if err != nil {
		return nil, core.ErrNotFound
	}
	node, ok := r.Store[internalID]
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

// Len returns the number of vectors in the index.
func (r *records[T]) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.Store)
}

// Info returns engine configuration and runtime statistics.
func (r *records[T]) Info() EngineInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return EngineInfo{
		Quantization:   r.quantization,
		IndexType:      r.indexType,
		DistanceMetric: r.distanceMetric,
		Dimensions:     r.dimensions,
		VectorCount:    len(r.Store),
		EnableMmap:     r.vectorStore != nil,
	}
}

// check rejects a write the index would refuse, before it reaches the WAL.
// Replay treats an intact record it can't apply as corruption, so a rejected
// write that was logged anyway stopped the next restart. width is the index's
// width as of this write: a batch passes the width its earlier items set.
func (r *records[T]) check(vec []float32, sparse core.SparseVector, width int) error {
	if err := validateVectorDims(vec, width); err != nil {
		return err
	}
	if err := validateSparse(sparse); err != nil {
		return err
	}
	return checkEncodable(vec, r.quantization, r.distanceMetric)
}

// insert is Engine.Insert for both engines: the write is checked, goes to the
// WAL, then apply puts it in the index. Nothing is logged or applied if the
// check fails, and nothing is applied if the WAL write fails.
func (r *records[T]) insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]any, apply func(string, []float32, core.SparseVector, map[string]any) error) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.check(vec, sparse, r.dimensions); err != nil {
		return err
	}
	if err := r.wal.WriteEntry(ctx, &WALEntry{Action: WALActionInsert, ID: id, Vector: vec, Sparse: sparse, Meta: meta}); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to write WAL entry")
		return err
	}
	if err := apply(id, vec, sparse, meta); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to insert vector")
		return err
	}
	return nil
}

// logBatch checks each item of a batch insert and writes the ones that pass
// to the WAL with one fsync, before any is applied. It returns the items to
// apply and the ones rejected. If the WAL write fails, the caller applies
// nothing: see WAL.WriteEntries for the atomicity this relies on.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) logBatch(ctx context.Context, items []BatchInsertItem) (accepted []BatchInsertItem, failures []BatchInsertError, err error) {
	accepted = make([]BatchInsertItem, 0, len(items))
	entries := make([]*WALEntry, 0, len(items))
	width := r.dimensions
	for _, item := range items {
		if err := r.check(item.Vector, item.Sparse, width); err != nil {
			failures = append(failures, BatchInsertError{ID: item.ID, Err: err})
			continue
		}
		if width == 0 {
			width = len(item.Vector) // the first accepted item sets an empty index's width
		}
		accepted = append(accepted, item)
		entries = append(entries, &WALEntry{Action: WALActionInsert, ID: item.ID, Vector: item.Vector, Sparse: item.Sparse, Meta: item.Meta})
	}

	if err := r.wal.WriteEntries(ctx, entries); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Int("count", len(entries)).Msg("failed to write WAL batch")
		return nil, nil, err
	}
	return accepted, failures, nil
}

// delete is Engine.Delete for both engines: false without touching the WAL if
// id isn't stored, otherwise the delete goes to the WAL first, then apply
// takes it out of the index.
func (r *records[T]) delete(ctx context.Context, id string, apply func(string) error) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	internalID, err := r.IDMapper.ToUint32ID(id)
	if err != nil {
		return false, nil
	}
	if _, exists := r.Store[internalID]; !exists {
		return false, nil
	}

	if err := r.wal.WriteEntry(ctx, &WALEntry{Action: WALActionDelete, ID: id}); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to write WAL entry")
		return false, err
	}
	if err := apply(id); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to delete vector")
		return false, err
	}
	return true, nil
}

// replay is Engine.ReplayWAL for both engines: each entry is applied with
// insert or remove, without writing it to the WAL again. See replayWALFile
// for what counts as a torn write and what stops recovery.
func (r *records[T]) replay(walPath string, insert func(string, []float32, core.SparseVector, map[string]any) error, remove func(string) error) error {
	return replayWALFile(walPath, func(entry WALEntry) error {
		r.mu.Lock()
		defer r.mu.Unlock()
		switch entry.Action {
		case WALActionInsert:
			return insert(entry.ID, entry.Vector, entry.Sparse, entry.Meta)
		case WALActionDelete:
			return ignoreNotFound(remove(entry.ID))
		default:
			return fmt.Errorf("unknown WAL action %q", entry.Action)
		}
	})
}

// put stores a record, replacing any with the same ID, and keeps the metadata
// and inverted indexes in step. The vector is checked and encoded before
// anything is touched, so a rejected vector changes nothing: not the width,
// not the indexes, not the ID mapping. (Only a failed mmap write, after the ID
// is allocated, leaves that ID mapped.) replaced reports whether the ID
// already had a record, which HNSW must take out of its graph before adding
// the new one.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) put(id string, vec []float32, sparse core.SparseVector, meta map[string]any) (node *core.VectorNode[T], replaced bool, err error) {
	if err := validateVectorDims(vec, r.dimensions); err != nil {
		return nil, false, err
	}
	if err := validateSparse(sparse); err != nil {
		return nil, false, err
	}
	encoded, err := r.encode(vec)
	if err != nil {
		return nil, false, err
	}

	internalID, err := r.IDMapper.GetOrCreate(id)
	if err != nil {
		return nil, false, err
	}

	// With mmap the stored vector is the mmap-backed slice, and no heap copy
	// is kept; without it, an exact-size copy (see storedCopy).
	if r.vectorStore != nil {
		mmapVec, err := putToMmapStore(r.vectorStore, internalID, encoded)
		if err != nil {
			return nil, false, fmt.Errorf("mmap store put: %w", err)
		}
		encoded = mmapVec
	} else {
		encoded = storedCopy(encoded)
	}

	old, replaced := r.Store[internalID]
	if r.InvertedIndex != nil {
		if replaced {
			r.removePostings(internalID, old.Sparse)
		}
		r.addPostings(internalID, sparse)
	}
	if r.metaIndex != nil {
		if replaced {
			r.metaIndex.Remove(internalID, old.Metadata)
		}
		r.metaIndex.Add(internalID, meta)
	}

	if r.dimensions == 0 {
		r.dimensions = len(vec)
	}

	node = &core.VectorNode[T]{
		InternalID: internalID,
		ExternalID: id,
		Vector:     encoded,
		Sparse:     sparse,
		Metadata:   meta,
	}
	r.Store[internalID] = node
	return node, replaced, nil
}

// remove deletes the record for id and its index entries, returning
// core.ErrNotFound if there is none. detach, if not nil, gets the record's
// internal ID before the ID mapping is released, so an engine can drop its
// own structures for it.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) remove(id string, detach func(internalID uint32)) error {
	internalID, err := r.IDMapper.ToUint32ID(id)
	if err != nil {
		return core.ErrNotFound
	}
	node, exists := r.Store[internalID]
	if !exists {
		return core.ErrNotFound
	}

	if r.InvertedIndex != nil {
		r.removePostings(internalID, node.Sparse)
	}
	if r.metaIndex != nil {
		r.metaIndex.Remove(internalID, node.Metadata)
	}
	delete(r.Store, internalID)
	if detach != nil {
		detach(internalID)
	}

	return r.IDMapper.Delete(id)
}

// reset is Engine.Reset for both engines. The empty state reaches disk first
// -- writeEmpty publishes an empty snapshot, which truncates the WAL last --
// and only then is memory cleared, with clearEngine for engine-specific
// state. A reset used to clear memory and the WAL but leave the snapshot, so a
// restart brought the deleted vectors back.
func (r *records[T]) reset(snapshotPath string, writeEmpty func(path string) error, clearEngine func()) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if snapshotPath != "" {
		if err := writeEmpty(snapshotPath); err != nil {
			return fmt.Errorf("write empty snapshot: %w", err)
		}
	} else if r.wal != nil {
		if err := r.wal.Clear(); err != nil {
			return err
		}
	}

	if clearEngine != nil {
		clearEngine()
	}
	r.clear()
	return nil
}

// emptyContents is the snapshot of a reset index: no records, a fresh ID
// mapping, and only a configured width.
func (r *records[T]) emptyContents() snapshotContents[T] {
	contents := r.snapshotContents()
	contents.nodes = map[uint32]*core.VectorNode[T]{}
	contents.idMapper = core.NewIDMapper()
	contents.dimensions = r.configuredDimensions
	return contents
}

// clear empties the records, both indexes and the mmap store. The WAL is
// reset's to truncate: only after the empty state is on disk.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) clear() {
	r.Store = make(map[uint32]*core.VectorNode[T])
	if r.InvertedIndex != nil {
		r.InvertedIndex = make(map[uint32][]core.Posting)
	}
	if r.metaIndex != nil {
		r.metaIndex.Clear()
	}
	if r.vectorStore != nil {
		if err := r.vectorStore.Reset(); err != nil {
			// Logged, not returned: the in-memory state is already reset.
			log.Error().Err(err).Msg("failed to reset mmap vector store")
		}
	}
	if r.IDMapper != nil {
		r.IDMapper.Clear()
	}

	// An empty index has no width to enforce. Keeping a learned one here left
	// a reset index rejecting every vector of a different size -- so clearing
	// to start over with different embeddings needed a server restart. A
	// configured width is the operator's choice and survives.
	r.dimensions = r.configuredDimensions
}

// encode converts vec to the stored type, after the fix-up scalar
// quantization needs (see prepareForEncode).
func (r *records[T]) encode(vec []float32) (T, error) {
	input, err := prepareForEncode(vec, r.quantization, r.distanceMetric)
	if err != nil {
		var zero T
		return zero, err
	}
	return r.encodeFunc(input), nil
}

// encodeQuery checks a query -- its width, and its sparse part -- and encodes
// it as stored vectors are. A query of the wrong width can't be compared
// against anything stored, and the comparison functions report that as a bare
// "vector dimensions mismatch" from deep inside the search. Checking here
// names the required width, classifies the failure as the caller's, and makes
// both engines reject a bad query the same way, even when the index is empty.
// PRECONDITION: r.mu.RLock() held.
func (r *records[T]) encodeQuery(query []float32, sparseQuery core.SparseVector) (T, error) {
	var zero T
	if err := validateVectorDims(query, r.dimensions); err != nil {
		return zero, err
	}
	if err := validateSparse(sparseQuery); err != nil {
		return zero, err
	}
	return r.encode(query)
}

// filterAllowlist returns the internal IDs matching filters, from the
// metadata index. indexed is false when there are no filters or no metadata
// index; the caller then checks each candidate with core.MatchFilter instead.
// PRECONDITION: r.mu.RLock() held.
func (r *records[T]) filterAllowlist(filters map[string]any) (allow map[uint32]struct{}, indexed bool) {
	if filters == nil || r.metaIndex == nil {
		return nil, false
	}
	return r.metaIndex.Allowlist(filters), true
}

// hybridAlpha weights a hybrid search's dense score against its sparse score:
// 70% semantic, 30% keyword.
const hybridAlpha = 0.7

// scorer returns how a candidate's final score is formed from its dense
// score. With a sparse query and hybrid search on, it is blended with the
// candidate's sparse score:
//
//	final = hybridAlpha*dense + (1-hybridAlpha)*sparse
//
// Otherwise it is the dense score.
// PRECONDITION: r.mu.RLock() held.
func (r *records[T]) scorer(sparseQuery core.SparseVector) func(internalID uint32, dense float32) float32 {
	if r.InvertedIndex == nil || sparseQuery.IsEmpty() {
		return func(_ uint32, dense float32) float32 { return dense }
	}

	sparseScores := make(map[uint32]float32)
	for id, node := range r.Store {
		if node.Sparse.IsEmpty() {
			continue
		}
		if score := core.SparseDotProduct(sparseQuery, node.Sparse); score > 0 {
			sparseScores[id] = score
		}
	}
	return func(internalID uint32, dense float32) float32 {
		return hybridAlpha*dense + (1-hybridAlpha)*sparseScores[internalID]
	}
}

// rankTop sorts results best first and keeps the top k, or all of them when
// k <= 0.
func rankTop(results []SearchResult, k int) []SearchResult {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if k > 0 && len(results) > k {
		results = results[:k]
	}
	return results
}

// rebuildIndexes recomputes the metadata and inverted indexes from the stored
// records, as every load path must: snapshots don't store either.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) rebuildIndexes() {
	if r.metaIndex != nil {
		r.metaIndex.Clear()
		for internalID, node := range r.Store {
			r.metaIndex.Add(internalID, node.Metadata)
		}
	}
	r.rebuildInvertedIndex()
}

// rebuildInvertedIndex recomputes every posting list from the stored sparse
// vectors, discarding whatever the inverted index held before.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) rebuildInvertedIndex() {
	if r.InvertedIndex == nil {
		return // hybrid search disabled
	}

	// A fresh map rather than clear(): nothing outside the index should hold
	// the old one, but a new map makes that not matter.
	r.InvertedIndex = make(map[uint32][]core.Posting)
	for internalID, node := range r.Store {
		r.addPostings(internalID, node.Sparse)
	}
}

// addPostings adds a document's sparse vector to the inverted index.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) addPostings(docID uint32, sparse core.SparseVector) {
	for i, tokenID := range sparse.Indices {
		r.InvertedIndex[tokenID] = append(r.InvertedIndex[tokenID], core.Posting{DocID: docID, Weight: sparse.Values[i]})
	}
}

// removePostings removes a document's postings from the inverted index,
// dropping any posting list it leaves empty.
// PRECONDITION: r.mu.Lock() held.
func (r *records[T]) removePostings(docID uint32, sparse core.SparseVector) {
	for _, tokenID := range sparse.Indices {
		postings := r.InvertedIndex[tokenID]
		filtered := make([]core.Posting, 0, len(postings))
		for _, posting := range postings {
			if posting.DocID != docID {
				filtered = append(filtered, posting)
			}
		}
		if len(filtered) == 0 {
			delete(r.InvertedIndex, tokenID)
		} else {
			r.InvertedIndex[tokenID] = filtered
		}
	}
}

// snapshotContents is what this index's snapshot holds besides anything
// engine-specific (see startSnapshot).
func (r *records[T]) snapshotContents() snapshotContents[T] {
	return snapshotContents[T]{
		indexType:      r.indexType,
		distanceMetric: r.distanceMetric,
		dimensions:     r.dimensions,
		idMapper:       r.IDMapper,
		nodes:          r.Store,
		vectorStore:    r.vectorStore,
	}
}
