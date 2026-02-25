package index

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"sync"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
)

// HNSWIndex is a thread-safe vector index backed by an HNSW graph for approximate nearest neighbor search.
// T is the vector storage type ([]float32 or []int8). Dual storage is used: the HNSW graph holds
// vectors for fast search; the metadata map holds full node data (sparse vectors, external ID, metadata).
type HNSWIndex[T hnsw.VectorType] struct {
	graph         *hnsw.Graph[uint32, T]
	metadata      map[uint32]*core.VectorNode[T]
	InvertedIndex map[uint32][]core.Posting
	IDMapper      *core.IDMapper

	mu           sync.RWMutex
	wal          *WAL
	encodeFunc   func([]float32) T
	distanceFunc func(T, T) (float32, error)
	hnswDistFunc hnsw.DistanceFunc[T] // stored for Clear() graph reset
	hnswM        int                  // stored for Clear() graph reset
	hnswEfSearch int                  // stored for Clear() graph reset
}

// NewHNSWIndex creates an empty HNSWIndex ready for use.
func NewHNSWIndex[T hnsw.VectorType](
	wal *WAL,
	invertedIndex map[uint32][]core.Posting,
	idMapper *core.IDMapper,
	encodeFunc func([]float32) T,
	distanceFunc func(T, T) (float32, error),
	hnswDistFunc hnsw.DistanceFunc[T],
	m int,
	efSearch int,
) *HNSWIndex[T] {
	g := hnsw.NewGraph[uint32, T]()
	g.M = m
	g.EfSearch = efSearch
	g.Distance = hnswDistFunc

	return &HNSWIndex[T]{
		graph:         g,
		metadata:      make(map[uint32]*core.VectorNode[T]),
		InvertedIndex: invertedIndex,
		IDMapper:      idMapper,
		wal:           wal,
		encodeFunc:    encodeFunc,
		distanceFunc:  distanceFunc,
		hnswDistFunc:  hnswDistFunc,
		hnswM:         m,
		hnswEfSearch:  efSearch,
	}
}

// Insert adds or updates a vector in the index with thread-safety.
func (idx *HNSWIndex[T]) Insert(id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	err := idx.wal.WriteEntry(&WALEntry{
		Action: WALActionInsert,
		ID:     id,
		Vector: vec,
		Sparse: sparse,
		Meta:   meta,
	})
	if err != nil {
		log.Printf("Failed to write WAL entry: %v", err)
		return err
	}

	return idx.insertInternal(id, vec, sparse, meta)
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

	encoded := idx.encodeFunc(vec)

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
// When filters are provided, over-fetching (k*5) compensates for candidates that fail the filter.
// When hybrid search is enabled, dense and sparse scores are combined with alpha=0.7.
func (idx *HNSWIndex[T]) Search(query []float32, sparseQuery core.SparseVector, k int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) == 0 {
		return nil, errors.New("empty query vector")
	}

	if idx.graph.Len() == 0 {
		return nil, nil
	}

	encodedQuery := idx.encodeFunc(query)

	useHybridSearch := idx.InvertedIndex != nil && !sparseQuery.IsEmpty()
	const alpha = 0.7

	var sparseScores map[uint32]float32
	if useHybridSearch {
		sparseScores = idx.computeSparseScores(sparseQuery)
	}

	// Over-fetch when filters are present: extra candidates compensate for filtered-out results.
	fetchCount := k
	if filters != nil {
		fetchCount = k * 5
	}

	candidates := idx.graph.SearchWithDistance(encodedQuery, fetchCount)
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

		// HNSW returns cosine distance (lower = closer); invert to get similarity score.
		denseScore := 1 - candidate.Distance

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

// Delete removes a vector from the index by ID.
func (idx *HNSWIndex[T]) Delete(id string) (bool, error) {
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

	err = idx.wal.WriteEntry(&WALEntry{
		Action: WALActionDelete,
		ID:     id,
	})
	if err != nil {
		log.Printf("Failed to write WAL entry: %v", err)
		return false, err
	}

	if err := idx.deleteInternal(id); err != nil {
		log.Printf("Failed to delete vector: %v", err)
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

	return idx.IDMapper.Delete(id)
}

// SaveToFile serializes the HNSW index to a snapshot file (version 4).
// The HNSW graph is exported to bytes and embedded in the GOB stream.
func (idx *HNSWIndex[T]) SaveToFile(path string) error {
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

	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			return err
		}
	}

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path comes from operator config, not user input
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(f)

	header := SnapshotHeader{
		Version:             4, // v4: HNSWIndex format
		Quantization:        quantType,
		DistanceMetric:      "cosine",
		HybridSearchEnabled: idx.InvertedIndex != nil,
		IndexType:           "hnsw",
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	if err := idx.IDMapper.EncodeGOB(encoder); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	if err := encoder.Encode(idx.metadata); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	if idx.InvertedIndex != nil {
		if err := encoder.Encode(idx.InvertedIndex); err != nil {
			_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
			return err
		}
	}

	// Serialize HNSW graph to a byte slice and embed it in the GOB stream.
	// This keeps the format consistent (one GOB stream) while leveraging
	// the graph's own binary encoding for compact, version-aware storage.
	var graphBuf bytes.Buffer
	if err := idx.graph.Export(&graphBuf); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return fmt.Errorf("failed to export HNSW graph: %w", err)
	}
	if err := encoder.Encode(graphBuf.Bytes()); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	return nil
}

// LoadFromFile reads the HNSW index from a v4 snapshot file.
// A missing file is not an error — the server starts with an empty index.
func (idx *HNSWIndex[T]) LoadFromFile(path string) (err error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path) //nolint:gosec // path comes from operator config, not user input
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

	if header.Version != 4 || header.IndexType != "hnsw" {
		if header.IndexType == "" || header.IndexType == "brute" {
			return fmt.Errorf("snapshot was created with brute-force index (version %d). Delete data files or change index_type to 'brute' in config", header.Version)
		}
		return fmt.Errorf("unsupported snapshot: version=%d, index_type=%q", header.Version, header.IndexType)
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

	return nil
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
	count := 0
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		var entry WALEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			log.Printf("WARNING: Skipping malformed WAL entry at line %d: %v", lineNum, err)
			continue
		}

		switch entry.Action {
		case WALActionInsert:
			idx.mu.Lock()
			if err := idx.insertInternal(entry.ID, entry.Vector, entry.Sparse, entry.Meta); err != nil {
				log.Printf("WARNING: Failed to replay insert for '%s' at line %d: %v", entry.ID, lineNum, err)
				idx.mu.Unlock()
				continue
			}
			idx.mu.Unlock()

		case WALActionDelete:
			idx.mu.Lock()
			if err := idx.deleteInternal(entry.ID); err != nil {
				if !errors.Is(err, core.ErrNotFound) {
					log.Printf("WARNING: Failed to replay delete for '%s' at line %d: %v", entry.ID, lineNum, err)
				}
				idx.mu.Unlock()
				continue
			}
			idx.mu.Unlock()
		}
		count++
	}

	log.Printf("Replayed %d WAL entries", count)
	return nil
}

// Clear removes all vectors from the HNSW index, recreating the graph with the same parameters.
func (idx *HNSWIndex[T]) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	g := hnsw.NewGraph[uint32, T]()
	g.M = idx.hnswM
	g.EfSearch = idx.hnswEfSearch
	g.Distance = idx.hnswDistFunc
	idx.graph = g

	idx.metadata = make(map[uint32]*core.VectorNode[T])
	if idx.InvertedIndex != nil {
		idx.InvertedIndex = make(map[uint32][]core.Posting)
	}
}

// Len returns the number of vectors in the HNSW index.
func (idx *HNSWIndex[T]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Len()
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

// Ensure sort is used (referenced in TODO(human) guidance).
var _ = sort.Slice
