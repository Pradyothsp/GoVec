package core

// SparseVector holds token IDs and their mathematical weights (e.g., BM25 scores).
// Indices must be strictly ascending -- sorted, no repeats -- which Dot relies
// on; the index rejects a sparse vector that isn't.
type SparseVector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// IsEmpty checks if the vector contains any data
func (sv *SparseVector) IsEmpty() bool {
	return len(sv.Indices) == 0
}

// Dot is the dot product of sv and other: the sum of their weights' products
// over the token IDs both hold, 0 if they share none. It walks both index
// lists at once, in O(n+m), which needs both strictly ascending.
func (sv *SparseVector) Dot(other SparseVector) float32 {
	var dot float32
	i, j := 0, 0
	for i < len(sv.Indices) && j < len(other.Indices) {
		switch {
		case sv.Indices[i] == other.Indices[j]:
			dot += sv.Values[i] * other.Values[j]
			i++
			j++
		case sv.Indices[i] < other.Indices[j]:
			i++
		default:
			j++
		}
	}

	return dot
}

// Posting represents a single document's score for a specific keyword.
// Uses uint32 DocID for 77.8% memory reduction (8 bytes vs 36 bytes per posting).
// External string IDs are translated via IDMapper at API boundaries.
type Posting struct {
	DocID  uint32
	Weight float32
}
