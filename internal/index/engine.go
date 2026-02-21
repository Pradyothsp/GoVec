package index

import "github.com/Pradyothsp/govec/internal/core"

// Engine defines the common interface for all vector storage engines
// regardless of the underlying quantization method.
type Engine interface {
	// Insert adds or updates a vector with the given ID, optional sparse vector, and metadata
	Insert(id string, vec []float32, sparse core.SparseVector, meta map[string]interface{}) error

	// Search finds the k nearest neighbors to the query vector (with optional sparse query for hybrid search)
	Search(query []float32, sparseQuery core.SparseVector, k int, filters map[string]interface{}) ([]SearchResult, error)

	// Delete removes a vector by ID
	Delete(id string) (bool, error)

	// SaveToFile persists the index to disk
	SaveToFile(path string) error

	// LoadFromFile loads the index from disk
	LoadFromFile(path string) error

	// ReplayWAL replays the write-ahead log to recover uncommitted changes
	ReplayWAL(path string) error

	// Clear removes all vectors from the index
	Clear()

	// Len returns the number of vectors in the index
	Len() int
}

// SearchResult represents a single search result with score and metadata
type SearchResult struct {
	ID    string                 `json:"ID" example:"vec-001"`
	Score float32                `json:"Score" example:"0.97"`
	Meta  map[string]interface{} `json:"Meta" swaggertype:"object"`
}
