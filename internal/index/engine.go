package index

import (
	"context"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/metric"
)

// EngineInfo holds static and dynamic configuration details about the engine.
//
// It and the other result types here are plain data: each transport decides
// its own wire format (REST in internal/api/handlers, gRPC in grpcserver).
type EngineInfo struct {
	Quantization   config.Quantization
	IndexType      config.IndexType
	DistanceMetric metric.Name
	Dimensions     int
	VectorCount    int
	EnableMmap     bool
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

	// Flush writes a snapshot to the engine's snapshot path, then truncates
	// the WAL. It fails if the engine has no snapshot path.
	Flush(ctx context.Context) error

	// Recover loads the engine's snapshot, if there is one, then replays its
	// WAL over it. Call it once, before serving any request.
	Recover(ctx context.Context) error

	// Reset removes every vector durably: an empty snapshot replaces the
	// engine's snapshot and the WAL is truncated before memory is cleared, so
	// a restart comes back empty and a failed reset changes nothing. With no
	// snapshot path, only memory and the WAL are cleared.
	Reset(ctx context.Context) error

	// Len returns the number of vectors in the index
	Len() int

	// Info returns engine configuration and runtime statistics
	Info() EngineInfo
}

// FileEngine is an Engine that can also save, load and replay at paths the
// caller chooses, rather than only the ones it was built with. Both engines
// implement it, and NewEngine returns one, for tests and tools that manage
// their own files. The server holds a plain Engine, so the only code that
// knows where its files live is the engine's.
type FileEngine interface {
	Engine

	// SaveToFile writes a snapshot to path, then truncates the WAL.
	SaveToFile(ctx context.Context, path string) error

	// LoadFromFile replaces the engine's contents with the snapshot at path.
	// A missing file is not an error: the engine stays empty.
	LoadFromFile(ctx context.Context, path string) error

	// ReplayWAL applies the WAL at path over the engine's contents.
	ReplayWAL(path string) error
}

var (
	_ FileEngine = (*VectorIndex[[]float32])(nil)
	_ FileEngine = (*HNSWIndex[[]int8])(nil)
)

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
// Sparse is empty for a dense-only vector.
type VectorRecord struct {
	ID       string
	Vector   []float32
	Sparse   core.SparseVector
	Metadata map[string]any
}

// SearchResult is one search match: its ID, its score (higher = closer) and
// its metadata, if any.
type SearchResult struct {
	ID    string
	Score float32
	Meta  map[string]any
}
