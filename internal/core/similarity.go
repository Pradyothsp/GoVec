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

	var dotProduct float32
	var normA float32
	var normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

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
