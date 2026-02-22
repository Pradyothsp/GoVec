package core

// VectorNode represents a stored vector with its ID and associated metadata.
// T is the vector storage type (e.g., []float32 for no quantization, []int8 for scalar quantization)
// Uses dual IDs: InternalID (uint32) for efficient indexing, ExternalID (string) for API responses.
type VectorNode[T any] struct {
	InternalID uint32         `json:"internal_id"` // Internal uint32 ID for indexing
	ExternalID string         `json:"id"`          // Original string ID for API
	Vector     T              `json:"vector"`
	Sparse     SparseVector   `json:"sparse_vector,omitempty"`
	Metadata   map[string]any `json:"metadata"`
}
