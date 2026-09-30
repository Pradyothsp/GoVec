package index

import (
	"errors"
	"fmt"
)

// ErrEmptyVector is returned when an insert supplies a zero-length vector.
var ErrEmptyVector = errors.New("vector must not be empty")

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
		return fmt.Errorf("vector has %d dimensions, index requires %d", len(vec), dimensions)
	}

	return nil
}
