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
	return float32(math.Sqrt(float64(core.SquaredEuclideanFloat32(a, b))))
}

// SquaredEuclideanDistanceFloat32 is EuclideanDistanceFloat32 without the
// final sqrt. Used as Graph.SquaredDistance for Euclidean-metric float32
// graphs: sqrt is monotonic, so ranking on squared distance produces
// identical ordering to ranking on real distance while skipping sqrt() on
// every internal comparison -- only the k results a query actually returns
// need the real value back, via FromSquaredEuclideanDistance.
func SquaredEuclideanDistanceFloat32(a, b []float32) float32 {
	return core.SquaredEuclideanFloat32(a, b)
}

// FromSquaredEuclideanDistance converts a SquaredEuclideanDistanceFloat32/
// SquaredEuclideanDistanceInt8 result back into the real Euclidean
// distance. Shared by both vector types -- the squared value itself is
// always a plain float32 regardless of which type produced it.
func FromSquaredEuclideanDistance(sq float32) float32 {
	return float32(math.Sqrt(float64(sq)))
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

// SquaredEuclideanDistanceInt8 is SquaredEuclideanDistanceFloat32's int8
// counterpart -- EuclideanDistanceInt8 without the final sqrt. See
// SquaredEuclideanDistanceFloat32's doc comment.
func SquaredEuclideanDistanceInt8(a, b []int8) float32 {
	var sum int64
	for i := range a {
		diff := int64(a[i]) - int64(b[i])
		sum += diff * diff
	}
	return float32(sum)
}

// PrecomputeSquaredNormFloat32 computes v's squared L2 norm (sum of
// squares). Used as Graph.Precompute for cosine-metric float32 graphs (see
// CachedCosineDistanceFloat32) -- cosine's distance formula needs each
// vector's norm as a value standalone from any particular comparison, and
// that norm never changes once the vector is stored, so computing it once
// here and caching it (rather than inside every single distance() call, as
// vek32.CosineSimilarity does today) removes real, measured redundant work
// from HNSW's hottest loop. float64 accumulation, not float32: see
// layerNode.precomputed's doc comment for why (written for []int8's larger
// magnitudes, but the same headroom argument applies here with room to
// spare, since float32 inputs are typically much smaller-magnitude than
// int8's raw +-127 range).
func PrecomputeSquaredNormFloat32(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return sum
}

// CachedCosineDistanceFloat32 is CosineDistanceFloat32's cache-aware
// counterpart: aCache/bCache are each vector's PrecomputeSquaredNormFloat32
// result, computed once at insert time rather than recomputed here. Only
// the dot product -- which genuinely depends on both vectors and can't be
// cached -- is computed fresh. Deliberately has no zero-vector guard,
// matching vek32.CosineSimilarity's existing behavior (a zero vector's
// cache is 0, giving 0/0 -> NaN) exactly: this is a faster path to the same
// answer, not a place to change what that answer is.
func CachedCosineDistanceFloat32(aVec []float32, aCache float64, bVec []float32, bCache float64) float32 {
	magnitude := math.Sqrt(aCache) * math.Sqrt(bCache)
	return float32(1 - dotFloat32(aVec, bVec)/magnitude)
}

// dotFloat32 is the dot product of a and b, accumulated in float64. It keeps
// eight independent running totals: with one, every add waits for the one
// before it, and Go neither reorders float adds nor vectorises the loop, so
// at 1536 dimensions the single-total loop took 1.73 µs and was 84% of HNSW
// search time. Eight totals let the adds overlap: 0.38 µs, same
// float64 precision. b must be at least as long as a.
func dotFloat32(a, b []float32) float64 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float64
	b = b[:len(a)]
	n := len(a) &^ 7
	for i := 0; i < n; i += 8 {
		// Fixed-length subslices let the compiler drop the bounds checks.
		x, y := a[i:i+8:i+8], b[i:i+8:i+8]
		s0 += float64(x[0]) * float64(y[0])
		s1 += float64(x[1]) * float64(y[1])
		s2 += float64(x[2]) * float64(y[2])
		s3 += float64(x[3]) * float64(y[3])
		s4 += float64(x[4]) * float64(y[4])
		s5 += float64(x[5]) * float64(y[5])
		s6 += float64(x[6]) * float64(y[6])
		s7 += float64(x[7]) * float64(y[7])
	}
	for i := n; i < len(a); i++ {
		s0 += float64(a[i]) * float64(b[i])
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

// PrecomputeSquaredNormInt8 computes v's squared L2 norm (sum of squares),
// accumulated as int64 like core.CosineSimilarityInt8's own normA/normB --
// int8 values squared and summed can overflow int8/int16 almost
// immediately at realistic dimension counts, see layerNode.precomputed's
// doc comment -- then converted to float64 to match Graph.Precompute's
// signature. That conversion is exact: float64 represents any int64 this
// codebase will realistically ever produce here without rounding.
func PrecomputeSquaredNormInt8(v []int8) float64 {
	var sum int64
	for _, x := range v {
		sum += int64(x) * int64(x)
	}
	return float64(sum)
}

// CachedCosineDistanceInt8 is CosineDistanceInt8's cache-aware counterpart,
// mirroring core.CosineSimilarityInt8 exactly -- including its explicit
// zero-vector guard (return the maximum distance, 1.0, rather than let a
// zero norm divide-by-zero into NaN) and its cast order (divide as float32,
// same as the uncached path, rather than accumulating the whole computation
// in float64) -- so caching doesn't shift int8's existing numerical
// behavior any more than necessary to get the speedup. aCache/bCache are
// each vector's PrecomputeSquaredNormInt8 result.
func CachedCosineDistanceInt8(aVec []int8, aCache float64, bVec []int8, bCache float64) float32 {
	if aCache == 0 || bCache == 0 {
		return 1.0
	}
	var dot int64
	for i := range aVec {
		dot += int64(aVec[i]) * int64(bVec[i])
	}
	magnitude := math.Sqrt(aCache) * math.Sqrt(bCache)
	similarity := float32(dot) / float32(magnitude)
	return 1.0 - similarity
}

// distanceFuncsFloat32 is a registry of known float32 distance functions, mapped by name.
// This registry is essential for serializing and deserializing Graph instances,
// allowing the correct distance function to be identified and restored during import.
var distanceFuncsFloat32 = map[string]DistanceFunc[[]float32]{
	"euclidean_float32": EuclideanDistanceFloat32,
	"cosine_float32":    CosineDistanceFloat32,
}

// distanceFuncsInt8 is a registry of known int8 distance functions, mapped by name.
// Similar to distanceFuncsFloat32, this registry supports persistence operations
// by mapping function names to their implementations for int8 vectors.
var distanceFuncsInt8 = map[string]DistanceFunc[[]int8]{
	"euclidean_int8": EuclideanDistanceInt8,
	"cosine_int8":    CosineDistanceInt8,
}

// distanceFuncToNameFloat32 looks up the registered name for a given float32 distance function.
// It is used during graph export to store a string identifier for the distance function.
// Returns the name and true if found, otherwise an empty string and false.
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

// distanceFuncToNameInt8 looks up the registered name for a given int8 distance function.
// Similar to its float32 counterpart, it's used during graph export for persistence.
// Returns the name and true if found, otherwise an empty string and false.
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
