package core

// SparseDotProduct computes the dot product between two sparse vectors.
// Assumes indices are sorted in ascending order (as required by SparseVector contract).
// Uses a two-pointer merge algorithm for O(n+m) complexity where n and m are the number of non-zero elements.
//
// Returns 0.0 if either vector is empty or if there are no overlapping indices.
func SparseDotProduct(a, b SparseVector) float32 {
	// Handle empty vectors
	if a.IsEmpty() || b.IsEmpty() {
		return 0.0
	}

	var dotProduct float32
	i, j := 0, 0

	// Two-pointer merge: iterate through both sorted index arrays
	for i < len(a.Indices) && j < len(b.Indices) {
		switch {
		case a.Indices[i] == b.Indices[j]:
			// Matching index: multiply weights and accumulate
			dotProduct += a.Values[i] * b.Values[j]
			i++
			j++
		case a.Indices[i] < b.Indices[j]:
			// Advance pointer in 'a'
			i++
		default:
			// Advance pointer in 'b'
			j++
		}
	}

	return dotProduct
}
