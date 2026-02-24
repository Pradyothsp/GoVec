package hnsw

import (
	"math"
	"reflect"

	"github.com/viterin/vek/vek32"
)

// DistanceFunc is a function that computes the distance between two vectors of type V.
// Lower values indicate closer vectors.
type DistanceFunc[V VectorType] func(a, b V) float32

// CosineDistanceFloat32 computes the cosine distance between two float32 vectors.
func CosineDistanceFloat32(a, b []float32) float32 {
	return 1 - vek32.CosineSimilarity(a, b)
}

// EuclideanDistanceFloat32 computes the Euclidean distance between two float32 vectors.
func EuclideanDistanceFloat32(a, b []float32) float32 {
	// TODO: can we speedup with vek?
	var sum float32 = 0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// CosineDistanceInt8 computes the cosine distance between two int8 vectors.
// Uses int64 intermediate calculations to prevent overflow.
func CosineDistanceInt8(a, b []int8) float32 {
	var dot, magA, magB int64
	for i := range a {
		dot += int64(a[i]) * int64(b[i])
		magA += int64(a[i]) * int64(a[i])
		magB += int64(b[i]) * int64(b[i])
	}

	if magA == 0 || magB == 0 {
		return 1.0
	}

	return 1.0 - float32(dot)/float32(math.Sqrt(float64(magA*magB)))
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

// CosineDistance is an alias for CosineDistanceFloat32.
// Backward compatibility - keep old names for float32.
var CosineDistance = CosineDistanceFloat32

// EuclideanDistance is an alias for EuclideanDistanceFloat32.
// Backward compatibility - keep old names for float32.
var EuclideanDistance = EuclideanDistanceFloat32

var distanceFuncsFloat32 = map[string]DistanceFunc[[]float32]{
	"euclidean": EuclideanDistanceFloat32,
	"cosine":    CosineDistanceFloat32,
}

var distanceFuncsInt8 = map[string]DistanceFunc[[]int8]{
	"euclidean": EuclideanDistanceInt8,
	"cosine":    CosineDistanceInt8,
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
