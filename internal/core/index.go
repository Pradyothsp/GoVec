package core

import "sync"

type VectorIndex struct {
	mu    sync.RWMutex
	Store map[string]*VectorNode
}

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
