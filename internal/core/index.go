package core

import (
	"errors"
	"sort"
	"sync"
)

// VectorIndex is a thread-safe in-memory store for vector embeddings.
type VectorIndex struct {
	mu    sync.RWMutex
	Store map[string]*VectorNode
}

// SearchResult represents a single match
type SearchResult struct {
	ID    string                 `json:"ID" example:"vec-001"`
	Score float32                `json:"Score" example:"0.97"`
	Meta  map[string]interface{} `json:"Meta" swaggertype:"object"` // Return metadata so user sees what it is
}

// NewVectorIndex creates an empty VectorIndex ready for use.
func NewVectorIndex() *VectorIndex {
	return &VectorIndex{
		Store: make(map[string]*VectorNode),
	}
}

// Insert adds or updates a vector in the index with thread-safety
func (idx *VectorIndex) Insert(id string, vec []float32, meta map[string]any) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.Store[id] = &VectorNode{
		ID:       id,
		Vector:   vec,
		Metadata: meta,
	}
}

// Search returns the top-limit nearest neighbours to query using cosine similarity.
func (idx *VectorIndex) Search(query []float32, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) == 0 {
		return nil, errors.New("empty query vector")
	}

	results := make([]SearchResult, 0, limit)
	for id, node := range idx.Store {
		// Filter check
		if !matchFilter(node.Metadata, filters) {
			continue
		}

		// Similarity calculation
		score, err := CosineSimilarity(query, node.Vector)
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
func (idx *VectorIndex) Delete(id string) bool {
	idx.mu.Lock() // BLOCK everyone (Reads & Writes)
	defer idx.mu.Unlock()

	_, exists := idx.Store[id]
	if !exists {
		return false
	}

	delete(idx.Store, id) // The built-in Go delete function
	return true
}
