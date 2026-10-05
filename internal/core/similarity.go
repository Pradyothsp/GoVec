package core

import (
	"errors"
	"math"
)

// CosineSimilarity calculates the cosine similarity between two vectors
// Returns a score between -1.0 and 1.0 (1.0 is identical)
// Returns 0.0 for zero vectors (no meaningful similarity)
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector dimensions mismatch")
	}

	if len(a) == 0 {
		return 0, errors.New("empty vectors")
	}

	dotProduct, normA, normB := cosineSums(a, b)

	// Zero vectors have undefined direction - return 0.0 similarity
	if normA == 0 || normB == 0 {
		return 0.0, nil
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))), nil
}

// CosineSimilarityInt8 computes cosine similarity for int8 vectors
// Uses int64 for intermediate calculations to avoid overflow
// Returns 0.0 for zero vectors (no meaningful similarity)
func CosineSimilarityInt8(a, b []int8) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector dimensions mismatch")
	}

	if len(a) == 0 {
		return 0, errors.New("empty vectors")
	}

	var dotProduct int64
	var normA int64
	var normB int64

	for i := range a {
		dotProduct += int64(a[i]) * int64(b[i])
		normA += int64(a[i]) * int64(a[i])
		normB += int64(b[i]) * int64(b[i])
	}

	// Zero vectors have undefined direction - return 0.0 similarity
	if normA == 0 || normB == 0 {
		return 0.0, nil
	}

	// Calculate sqrt and similarity
	magnitude := math.Sqrt(float64(normA)) * math.Sqrt(float64(normB))
	return float32(dotProduct) / float32(magnitude), nil
}

// EuclideanSimilarity calculates a similarity score from the Euclidean (L2) distance
// between two vectors. Search results are ranked by descending score (see VectorIndex.Search),
// but Euclidean distance itself ranks ascending (0 = identical), so it's converted via
// 1 / (1 + distance): 1.0 for identical vectors, approaching 0 as distance grows.
func EuclideanSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector dimensions mismatch")
	}

	if len(a) == 0 {
		return 0, errors.New("empty vectors")
	}

	distance := float32(math.Sqrt(float64(SquaredEuclideanFloat32(a, b))))
	return 1 / (1 + distance), nil
}

// EuclideanSimilarityInt8 computes Euclidean similarity for int8 vectors.
// Uses int64 for intermediate calculations to avoid overflow. See EuclideanSimilarity
// for the distance-to-score conversion.
func EuclideanSimilarityInt8(a, b []int8) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector dimensions mismatch")
	}

	if len(a) == 0 {
		return 0, errors.New("empty vectors")
	}

	var sumSquares int64
	for i := range a {
		diff := int64(a[i]) - int64(b[i])
		sumSquares += diff * diff
	}

	distance := float32(math.Sqrt(float64(sumSquares)))
	return 1 / (1 + distance), nil
}

// The float32 loops below keep several independent running totals. With one,
// every add waits for the one before it, and Go neither reorders float adds
// nor vectorises the loop, so at 1536 dimensions a single-total loop runs
// 4-5x slower than the CPU allows. int8 loops are left simple: integer adds
// don't stall like this, so unrolling bought nothing for the int8 dot product
// and about 1.4x for int8 squared Euclidean, not worth the extra code.

// cosineSums returns a·b, a·a and b·b in one pass, four totals each: twelve
// accumulators, which still fit in registers on amd64. b must be at least as
// long as a.
func cosineSums(a, b []float32) (dot, normA, normB float32) {
	var d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, b2, b3 float32
	b = b[:len(a)]
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		// Fixed-length subslices let the compiler drop the bounds checks.
		x, y := a[i:i+4:i+4], b[i:i+4:i+4]
		d0 += x[0] * y[0]
		d1 += x[1] * y[1]
		d2 += x[2] * y[2]
		d3 += x[3] * y[3]
		a0 += x[0] * x[0]
		a1 += x[1] * x[1]
		a2 += x[2] * x[2]
		a3 += x[3] * x[3]
		b0 += y[0] * y[0]
		b1 += y[1] * y[1]
		b2 += y[2] * y[2]
		b3 += y[3] * y[3]
	}
	for i := n; i < len(a); i++ {
		d0 += a[i] * b[i]
		a0 += a[i] * a[i]
		b0 += b[i] * b[i]
	}
	return (d0 + d1) + (d2 + d3), (a0 + a1) + (a2 + a3), (b0 + b1) + (b2 + b3)
}

// SquaredEuclideanFloat32 is the squared L2 distance between a and b, with
// eight running totals. Shared with the HNSW graph's Euclidean distance. b
// must be at least as long as a.
func SquaredEuclideanFloat32(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	b = b[:len(a)]
	n := len(a) &^ 7
	for i := 0; i < n; i += 8 {
		x, y := a[i:i+8:i+8], b[i:i+8:i+8]
		d0, d1, d2, d3 := x[0]-y[0], x[1]-y[1], x[2]-y[2], x[3]-y[3]
		d4, d5, d6, d7 := x[4]-y[4], x[5]-y[5], x[6]-y[6], x[7]-y[7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	for i := n; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}
