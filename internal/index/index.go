package index

import (
	"errors"
	"log"
	"sort"
	"sync"

	"github.com/Pradyothsp/govec/internal/core"
)

// VectorIndex is a thread-safe in-memory store for vector embeddings.
// T is the vector storage type (e.g., []float32 for no quantization, []int8 for scalar quantization)
type VectorIndex[T any] struct {
	Store         map[string]*core.VectorNode[T]
	InvertedIndex map[uint32][]core.Posting

	mu           sync.RWMutex
	wal          *WAL
	encodeFunc   func([]float32) T           // Converts input vectors to storage format
	distanceFunc func(T, T) (float32, error) // Computes distance between stored vectors
}

// NewVectorIndex creates an empty VectorIndex ready for use.
func NewVectorIndex[T any](wal *WAL, invertedIndex map[uint32][]core.Posting, encodeFunc func([]float32) T, distanceFunc func(T, T) (float32, error)) *VectorIndex[T] {
	return &VectorIndex[T]{
		Store:         make(map[string]*core.VectorNode[T]),
		InvertedIndex: invertedIndex,
		wal:           wal,
		encodeFunc:    encodeFunc,
		distanceFunc:  distanceFunc,
	}
}

// Insert adds or updates a vector in the index with thread-safety
func (idx *VectorIndex[T]) Insert(id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Update WAL (stores original float32)
	err := idx.wal.WriteEntry(&WALEntry{
		Action: WALActionInsert,
		ID:     id,
		Vector: vec,
		Sparse: sparse,
		Meta:   meta,
	})

	if err != nil {
		log.Printf("Failed to write WAL entry: %v", err)
		return err // If disk fails, we fail the request
	}

	// If hybrid search is enabled, update the inverted index
	if idx.InvertedIndex != nil {
		// Remove old postings if document exists (update case)
		if existingNode, exists := idx.Store[id]; exists {
			idx.removeFromInvertedIndex(id, existingNode.Sparse)
		}

		// Add new postings for the new sparse vector
		idx.addToInvertedIndex(id, sparse)
	}

	// Update memory (convert to storage format using encodeFunc)
	idx.Store[id] = &core.VectorNode[T]{
		ID:       id,
		Vector:   idx.encodeFunc(vec),
		Sparse:   sparse,
		Metadata: meta,
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
func (idx *VectorIndex[T]) Search(query []float32, sparseQuery core.SparseVector, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) == 0 {
		return nil, errors.New("empty query vector")
	}

	// Encode query to storage format
	encodedQuery := idx.encodeFunc(query)

	// Determine if we're doing hybrid search
	useHybridSearch := idx.InvertedIndex != nil && !sparseQuery.IsEmpty()

	// Alpha parameter for hybrid search (0.7 = 70% dense, 30% sparse)
	const alpha = 0.7

	// Pre-compute sparse scores if doing hybrid search
	var sparseScores map[string]float32
	if useHybridSearch {
		sparseScores = idx.computeSparseScores(sparseQuery)
	}

	results := make([]SearchResult, 0, limit)
	for id, node := range idx.Store {
		// Filter check
		if !core.MatchFilter(node.Metadata, filters) {
			continue
		}

		// Compute dense similarity score
		denseScore, err := idx.distanceFunc(encodedQuery, node.Vector)
		if err != nil {
			return nil, err
		}

		// Compute final score (hybrid or dense-only)
		var finalScore float32
		if useHybridSearch {
			// Get sparse score for this document (0.0 if not in sparse results)
			sparseScore := sparseScores[id]

			// Combine dense and sparse scores
			finalScore = alpha*denseScore + (1-alpha)*sparseScore
		} else {
			// Dense-only search
			finalScore = denseScore
		}

		results = append(results, SearchResult{
			ID:    id,
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
func (idx *VectorIndex[T]) computeSparseScores(sparseQuery core.SparseVector) map[string]float32 {
	scores := make(map[string]float32)

	// For each document in the store, compute sparse dot product
	for id, node := range idx.Store {
		if !node.Sparse.IsEmpty() {
			score := core.SparseDotProduct(sparseQuery, node.Sparse)
			if score > 0 {
				scores[id] = score
			}
		}
	}

	return scores
}

// Delete removes a vector from the index by ID
func (idx *VectorIndex[T]) Delete(id string) (bool, error) {
	idx.mu.Lock() // BLOCK everyone (Reads & Writes)
	defer idx.mu.Unlock()

	node, exists := idx.Store[id]
	if !exists {
		return false, nil
	}

	// Update WAL
	err := idx.wal.WriteEntry(&WALEntry{
		Action: WALActionDelete,
		ID:     id,
	})
	if err != nil {
		log.Printf("Failed to write WAL entry: %v", err)
		return false, err
	}

	// If hybrid search is enabled, remove from the inverted index
	if idx.InvertedIndex != nil {
		idx.removeFromInvertedIndex(id, node.Sparse)
	}

	// Update memory
	delete(idx.Store, id) // The built-in Go delete function
	return true, nil
}

// Clear removes all vectors from the index
func (idx *VectorIndex[T]) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.Store = make(map[string]*core.VectorNode[T])

	// Clear inverted index if hybrid search is enabled
	if idx.InvertedIndex != nil {
		idx.InvertedIndex = make(map[uint32][]core.Posting)
	}
}

// Len returns the number of vectors in the index
func (idx *VectorIndex[T]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.Store)
}

// addToInvertedIndex adds postings for a document's sparse vector to the inverted index.
// MUST be called with idx.mu.Lock() held.
func (idx *VectorIndex[T]) addToInvertedIndex(docID string, sparse core.SparseVector) {
	if sparse.IsEmpty() {
		return // No sparse vector, nothing to add
	}

	// For each token in the sparse vector, add a posting
	for i := 0; i < len(sparse.Indices); i++ {
		tokenID := sparse.Indices[i]
		weight := sparse.Values[i]

		// Append posting to the token's posting list
		idx.InvertedIndex[tokenID] = append(idx.InvertedIndex[tokenID], core.Posting{
			DocID:  docID,
			Weight: weight,
		})
	}
}

// removeFromInvertedIndex removes all postings for a document from the inverted index.
// MUST be called with idx.mu.Lock() held.
func (idx *VectorIndex[T]) removeFromInvertedIndex(docID string, sparse core.SparseVector) {
	if sparse.IsEmpty() {
		return // No sparse vector, nothing to remove
	}

	// For each token in the old sparse vector, remove the posting
	for _, tokenID := range sparse.Indices {
		postings := idx.InvertedIndex[tokenID]

		// Filter out the posting for this document
		filtered := make([]core.Posting, 0, len(postings))
		for _, posting := range postings {
			if posting.DocID != docID {
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
