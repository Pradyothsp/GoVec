package index

import (
	"errors"
	"fmt"

	"github.com/Pradyothsp/govec/internal/core"
)

var (
	// ErrEmptyVector is returned when an insert supplies a zero-length vector.
	ErrEmptyVector = errors.New("vector must not be empty")

	// ErrDimensionMismatch is returned when an insert supplies a vector whose
	// width differs from the index's.
	ErrDimensionMismatch = errors.New("vector dimension mismatch")

	// ErrInvalidSparseVector is returned when a sparse vector's indices and
	// values differ in length.
	ErrInvalidSparseVector = errors.New("invalid sparse_vector: indices and values must have the same length")
)

// IsInvalidVectorError reports whether err means the caller supplied an
// unusable vector, as opposed to the server failing.
//
// Transports use this to answer with a client-error status -- HTTP 400, gRPC
// InvalidArgument -- rather than reporting a 500 for a request the server
// correctly rejected. Keeping the classification here means the transports do
// not each need to know which sentinels the index layer defines.
func IsInvalidVectorError(err error) bool {
	return errors.Is(err, ErrEmptyVector) || errors.Is(err, ErrDimensionMismatch) || errors.Is(err, ErrInvalidSparseVector)
}

// validateSparse rejects a sparse vector whose indices and values don't pair
// up: every index needs its weight, and the inverted index and sparse scoring
// read them pairwise. An empty sparse vector is valid and means none.
func validateSparse(sparse core.SparseVector) error {
	if len(sparse.Indices) != len(sparse.Values) {
		return fmt.Errorf("%w: %d indices, %d values", ErrInvalidSparseVector, len(sparse.Indices), len(sparse.Values))
	}
	return nil
}

// validateVectorDims rejects any vector that could not participate in a
// similarity comparison against the rest of the index.
//
// Both engines derive the index's width from the first vector inserted and
// then never revisit it. Nothing downstream re-checks: CosineSimilarity and
// the Euclidean equivalent compare the query against every stored vector and
// return "vector dimensions mismatch" the moment two lengths disagree. So a
// single mis-sized vector does not fail its own insert -- it makes every
// later *search* fail, for every client, until that vector is deleted.
// Rejecting at insert is what keeps that invariant true.
//
// An empty vector is rejected outright rather than compared: the width-tracking
// in both engines is guarded on len(vec) > 0, so an empty vector slipped into a
// fresh index would set no width, be stored anyway, and then break every search
// once a real vector arrived and established one. It also has no meaning for
// similarity search -- CosineSimilarity already errors on it.
//
// dimensions == 0 means the index has no width yet (empty index, no configured
// engine.dimensions), so any non-empty vector is acceptable and becomes the
// width for everything after it.
func validateVectorDims(vec []float32, dimensions int) error {
	if len(vec) == 0 {
		return ErrEmptyVector
	}

	if dimensions > 0 && len(vec) != dimensions {
		return fmt.Errorf("%w: vector has %d dimensions, index requires %d", ErrDimensionMismatch, len(vec), dimensions)
	}

	return nil
}
