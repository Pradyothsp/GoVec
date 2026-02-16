package core

type VectorIndex struct {
	// OLD: Store map[string][]float32
	// NEW: Store pointers to the full node
	Store map[string]*VectorNode
}

func NewVectorIndex() *VectorIndex {
	return &VectorIndex{
		Store: make(map[string]*VectorNode),
	}
}

// Update Insert to accept metadata
func (idx *VectorIndex) Insert(id string, vec []float32, meta map[string]any) {
	idx.Store[id] = &VectorNode{
		ID:       id,
		Vector:   vec,
		Metadata: meta,
	}
}
