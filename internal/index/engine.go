package index

import (
	"context"

	"github.com/Pradyothsp/govec/internal/core"
)

// EngineInfo holds static and dynamic configuration details about the engine.
type EngineInfo struct {
	Quantization   string `json:"quantization" example:"none"`
	IndexType      string `json:"index_type" example:"brute"`
	DistanceMetric string `json:"distance_metric" example:"cosine"`
	Dimensions     int    `json:"dimensions" example:"0"`
	VectorCount    int    `json:"vector_count" example:"0"`
}

// Engine defines the common interface for all vector storage engines
// regardless of the underlying quantization method.
type Engine interface {
	// Insert adds or updates a vector with the given ID, optional sparse vector, and metadata
	Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]interface{}) error

	// Search finds the k nearest neighbors to the query vector (with optional sparse query for hybrid search)
	Search(ctx context.Context, query []float32, sparseQuery core.SparseVector, k int, filters map[string]interface{}) ([]SearchResult, error)

	// Delete removes a vector by ID
	Delete(ctx context.Context, id string) (bool, error)

	// SaveToFile persists the index to disk
	SaveToFile(ctx context.Context, path string) error

	// LoadFromFile loads the index from disk
	LoadFromFile(ctx context.Context, path string) error

	// ReplayWAL replays the write-ahead log to recover uncommitted changes
	ReplayWAL(path string) error

	// Clear removes all vectors from the index
	Clear()

	// Len returns the number of vectors in the index
	Len() int

	// Info returns engine configuration and runtime statistics
	Info() EngineInfo
}

// SearchResult represents a single search result with score and metadata
type SearchResult struct {
	ID    string                 `json:"id" example:"vec-001"`
	Score float32                `json:"score" example:"0.97"`
	Meta  map[string]interface{} `json:"meta" swaggertype:"object"`
}
