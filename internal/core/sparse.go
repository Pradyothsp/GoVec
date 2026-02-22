package core

// SparseVector holds token IDs and their mathematical weights (e.g., BM25 scores).
// Indices MUST be sorted in ascending order by the client for fast math.
type SparseVector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// IsEmpty checks if the vector contains any data
func (sv *SparseVector) IsEmpty() bool {
	return len(sv.Indices) == 0
}

// IsValid checks if the vector's Indices and Values are of the same length and non-empty
func (sv *SparseVector) IsValid() bool {
	return !sv.IsEmpty() && len(sv.Indices) == len(sv.Values)
}

// Posting represents a single document's score for a specific keyword.
// Uses uint32 DocID for 77.8% memory reduction (8 bytes vs 36 bytes per posting).
// External string IDs are translated via IDMapper at API boundaries.
type Posting struct {
	DocID  uint32
	Weight float32
}
