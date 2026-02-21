package core

// VectorNode represents a stored vector with its ID and associated metadata.
// T is the vector storage type (e.g., []float32 for no quantization, []int8 for scalar quantization)
type VectorNode[T any] struct {
	ID       string
	Vector   T
	Metadata map[string]any
}
