package hnsw

import (
	"math"
	"reflect"

	"github.com/viterin/vek/vek32"

	"github.com/Pradyothsp/govec/internal/core"
)

// DistanceFunc is a function that computes the distance between two vectors of type V.
// Lower values indicate closer vectors.
type DistanceFunc[V VectorType] func(a, b V) float32

// CosineDistanceFloat32 computes the cosine distance between two float32 vectors.
// Uses vek32 optimized implementation for float32 performance.
func CosineDistanceFloat32(a, b []float32) float32 {
	return 1 - vek32.CosineSimilarity(a, b)
}

// EuclideanDistanceFloat32 computes the Euclidean distance between two float32 vectors.
func EuclideanDistanceFloat32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// CosineDistanceInt8 computes the cosine distance between two int8 vectors.
// Reuses internal/core implementation which uses int64 intermediate calculations to prevent overflow.
func CosineDistanceInt8(a, b []int8) float32 {
	// Use core implementation (returns similarity, error never occurs in valid HNSW graph)
	similarity, err := core.CosineSimilarityInt8(a, b)
	if err != nil {
		// This should never happen in a well-formed HNSW graph (dimensions always match)
		// Return maximum distance as fallback
		return 1.0
	}
	return 1.0 - similarity
}

// EuclideanDistanceInt8 computes the Euclidean distance between two int8 vectors.
func EuclideanDistanceInt8(a, b []int8) float32 {
	var sum int64
	for i := range a {
		diff := int64(a[i]) - int64(b[i])
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

var distanceFuncsFloat32 = map[string]DistanceFunc[[]float32]{
	"euclidean_float32": EuclideanDistanceFloat32,
	"cosine_float32":    CosineDistanceFloat32,
}

var distanceFuncsInt8 = map[string]DistanceFunc[[]int8]{
	"euclidean_int8": EuclideanDistanceInt8,
	"cosine_int8":    CosineDistanceInt8,
}

func distanceFuncToNameFloat32(fn DistanceFunc[[]float32]) (string, bool) {
	for name, f := range distanceFuncsFloat32 {
		fnptr := reflect.ValueOf(fn).Pointer()
		fptr := reflect.ValueOf(f).Pointer()
		if fptr == fnptr {
			return name, true
		}
	}
	return "", false
}

func distanceFuncToNameInt8(fn DistanceFunc[[]int8]) (string, bool) {
	for name, f := range distanceFuncsInt8 {
		fnptr := reflect.ValueOf(fn).Pointer()
		fptr := reflect.ValueOf(f).Pointer()
		if fptr == fnptr {
			return name, true
		}
	}
	return "", false
}

// RegisterDistanceFuncFloat32 registers a float32 distance function with a name.
// A distance function must be registered here before a graph can be exported and imported.
func RegisterDistanceFuncFloat32(name string, fn DistanceFunc[[]float32]) {
	distanceFuncsFloat32[name] = fn
}

// RegisterDistanceFuncInt8 registers an int8 distance function with a name.
// A distance function must be registered here before a graph can be exported and imported.
func RegisterDistanceFuncInt8(name string, fn DistanceFunc[[]int8]) {
	distanceFuncsInt8[name] = fn
}
