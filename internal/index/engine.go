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
	EnableMmap     bool   `json:"enable_mmap" example:"false"`
}

// Engine defines the common interface for all vector storage engines
// regardless of the underlying quantization method.
type Engine interface {
	// Insert adds or updates a vector with the given ID, optional sparse vector, and metadata
	Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]interface{}) error

	// BatchInsert adds or updates multiple vectors as a single durable unit: one
	// WAL fsync for the whole batch rather than one per vector. A failure writing
	// the batch to the WAL aborts the entire batch (nothing is applied); once the
	// WAL write succeeds, per-item application errors are returned individually
	// without aborting the rest of the batch.
	BatchInsert(ctx context.Context, items []BatchInsertItem) ([]BatchInsertError, error)

	// GetByID retrieves the full record for a vector by ID
	GetByID(ctx context.Context, id string) (*VectorRecord, error)

	// Search finds the k nearest neighbors to the query vector (with an optional sparse query for hybrid search)
	Search(ctx context.Context, query []float32, sparseQuery core.SparseVector, k int, filters map[string]interface{}) ([]SearchResult, error)

	// Delete removes a vector by ID
	Delete(ctx context.Context, id string) (bool, error)

	// SaveToFile persists the index to disk
	SaveToFile(ctx context.Context, path string) error

	// LoadFromFile loads the index from disk
	LoadFromFile(ctx context.Context, path string) error

	// ReplayWAL replays the write-ahead log to recover uncommitted changes
	ReplayWAL(path string) error

	// Reset removes every vector durably: an empty snapshot replaces the one
	// at snapshotPath and the WAL is truncated before memory is cleared, so a
	// restart comes back empty and a failed reset changes nothing. With no
	// snapshotPath, only memory and the WAL are cleared.
	Reset(ctx context.Context, snapshotPath string) error

	// Len returns the number of vectors in the index
	Len() int

	// Info returns engine configuration and runtime statistics
	Info() EngineInfo
}

// BatchInsertItem is a single vector to insert as part of a BatchInsert call.
type BatchInsertItem struct {
	ID     string
	Vector []float32
	Sparse core.SparseVector
	Meta   map[string]interface{}
}

// BatchInsertError pairs a batch item's ID with the error that occurred applying it.
type BatchInsertError struct {
	ID  string
	Err error
}

// VectorRecord holds the full stored data for a vector, returned by GetByID.
//
// sparse_vector and metadata are omitted from the JSON when the record has
// none, rather than sent as null -- an absent field means absent, and callers
// can test for the key.
//
// SparseVector is a pointer for exactly that reason: encoding/json honours
// omitempty for pointers, maps and slices but silently ignores it on a struct
// value, which this field used to be. Every dense-only record therefore went
// out as {"sparse_vector":{"indices":null,"values":null}}, claiming a sparse
// component that was not there. Build the field with sparseOrNil, never by
// taking the address of a stored node's Sparse.
type VectorRecord struct {
	ID           string                 `json:"id" example:"vec-001"`
	Vector       []float32              `json:"vector" swaggertype:"array,number"`
	SparseVector *core.SparseVector     `json:"sparse_vector,omitempty" swaggertype:"object"`
	Metadata     map[string]interface{} `json:"metadata,omitempty" swaggertype:"object"`
}

// sparseOrNil returns a pointer to sv, or nil when sv holds nothing, so that a
// dense-only VectorRecord omits sparse_vector from its JSON entirely.
//
// sv is taken by value, so the pointer is to the copy. Returning &node.Sparse
// instead would hand a caller a path back into stored index state.
func sparseOrNil(sv core.SparseVector) *core.SparseVector {
	if sv.IsEmpty() {
		return nil
	}

	return &sv
}

// SearchResult represents a single search result with score and metadata.
//
// meta is omitted when the vector carries no metadata, matching
// VectorRecord.Metadata. It previously had no omitempty at all, so every
// result in every response ended with "meta":null.
type SearchResult struct {
	ID    string                 `json:"id" example:"vec-001"`
	Score float32                `json:"score" example:"0.97"`
	Meta  map[string]interface{} `json:"meta,omitempty" swaggertype:"object"`
}
