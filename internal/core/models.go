package core

// VectorNode represents a stored vector with its ID and associated metadata.
type VectorNode struct {
	ID       string
	Vector   []float32
	Metadata map[string]any
}
