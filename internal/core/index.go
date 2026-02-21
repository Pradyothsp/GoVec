package core

import (
	"errors"
	"log"
	"sort"
	"sync"
)

// VectorIndex is a thread-safe in-memory store for vector embeddings.
// T is the vector storage type (e.g., []float32 for no quantization, []int8 for scalar quantization)
type VectorIndex[T any] struct {
	mu           sync.RWMutex
	Store        map[string]*VectorNode[T]
	wal          *WAL
	encodeFunc   func([]float32) T           // Converts input vectors to storage format
	distanceFunc func(T, T) (float32, error) // Computes distance between stored vectors
}

// NewVectorIndex creates an empty VectorIndex ready for use.
func NewVectorIndex[T any](wal *WAL, encodeFunc func([]float32) T, distanceFunc func(T, T) (float32, error)) *VectorIndex[T] {
	return &VectorIndex[T]{
		Store:        make(map[string]*VectorNode[T]),
		wal:          wal,
		encodeFunc:   encodeFunc,
		distanceFunc: distanceFunc,
	}
}

// Insert adds or updates a vector in the index with thread-safety
func (idx *VectorIndex[T]) Insert(id string, vec []float32, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Update WAL (stores original float32)
	err := idx.wal.WriteEntry(WALEntry{
		Action: WALActionInsert,
		ID:     id,
		Vector: vec,
		Meta:   meta,
	})

	if err != nil {
		log.Printf("Failed to write WAL entry: %v", err)
		return err // If disk fails, we fail the request
	}

	// Update memory (convert to storage format using encodeFunc)
	idx.Store[id] = &VectorNode[T]{
		ID:       id,
		Vector:   idx.encodeFunc(vec),
		Metadata: meta,
	}

	return nil
}

// Search returns the top-limit nearest neighbours to query using the configured distance metric.
func (idx *VectorIndex[T]) Search(query []float32, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) == 0 {
		return nil, errors.New("empty query vector")
	}

	// Encode query to storage format
	encodedQuery := idx.encodeFunc(query)

	results := make([]SearchResult, 0, limit)
	for id, node := range idx.Store {
		// Filter check
		if !matchFilter(node.Metadata, filters) {
			continue
		}

		// Similarity calculation using injected distance function
		score, err := idx.distanceFunc(encodedQuery, node.Vector)
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{
			ID:    id,
			Score: score,
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

// Delete removes a vector from the index by ID
func (idx *VectorIndex[T]) Delete(id string) (bool, error) {
	idx.mu.Lock() // BLOCK everyone (Reads & Writes)
	defer idx.mu.Unlock()

	_, exists := idx.Store[id]
	if !exists {
		return false, nil
	}

	// Update WAL
	err := idx.wal.WriteEntry(WALEntry{
		Action: WALActionDelete,
		ID:     id,
	})
	if err != nil {
		log.Printf("Failed to write WAL entry: %v", err)
		return false, err
	}

	// Update memory
	delete(idx.Store, id) // The built-in Go delete function
	return true, nil
}

// Clear removes all vectors from the index
func (idx *VectorIndex[T]) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.Store = make(map[string]*VectorNode[T])
}

// Len returns the number of vectors in the index
func (idx *VectorIndex[T]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.Store)
}
