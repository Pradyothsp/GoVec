package metric

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// benchDims are a small embedding and the common 1536-wide one.
var benchDims = []int{128, 1536}

func signedInt8s(rng *rand.Rand, n int) []int8 {
	x := make([]int8, n)
	for i := range x {
		x[i] = int8(rng.IntN(255) - 127) //nolint:gosec // in int8 range by construction
	}
	return x
}

// BenchmarkSimilarity times the kernels brute-force search calls once per
// stored vector per query.
//
//	go test ./internal/metric/ -run '^$' -bench Similarity
func BenchmarkSimilarity(b *testing.B) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic benchmark data
	for _, dims := range benchDims {
		f1, f2 := signedFloats(rng, dims), signedFloats(rng, dims)
		i1, i2 := signedInt8s(rng, dims), signedInt8s(rng, dims)
		kernels := []struct {
			name string
			run  func() (float32, error)
		}{
			{"cosine/float32", func() (float32, error) { return CosineSimilarity(f1, f2) }},
			{"cosine/int8", func() (float32, error) { return CosineSimilarityInt8(i1, i2) }},
			{"euclidean/float32", func() (float32, error) { return EuclideanSimilarity(f1, f2) }},
			{"euclidean/int8", func() (float32, error) { return EuclideanSimilarityInt8(i1, i2) }},
		}
		for _, k := range kernels {
			b.Run(fmt.Sprintf("%s/d%d", k.name, dims), func(b *testing.B) {
				for b.Loop() {
					_, _ = k.run()
				}
			})
		}
	}
}

// BenchmarkDistance times the kernels the HNSW graph calls on every
// comparison while it is built and searched.
//
//	go test ./internal/metric/ -run '^$' -bench Distance
func BenchmarkDistance(b *testing.B) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic benchmark data
	for _, dims := range benchDims {
		f1, f2 := signedFloats(rng, dims), signedFloats(rng, dims)
		n1, n2 := PrecomputeSquaredNormFloat32(f1), PrecomputeSquaredNormFloat32(f2)
		i1, i2 := signedInt8s(rng, dims), signedInt8s(rng, dims)
		m1, m2 := PrecomputeSquaredNormInt8(i1), PrecomputeSquaredNormInt8(i2)
		kernels := []struct {
			name string
			run  func() float32
		}{
			{"cosine/float32", func() float32 { return CosineDistanceFloat32(f1, f2) }},
			{"cached_cosine/float32", func() float32 { return CachedCosineDistanceFloat32(f1, n1, f2, n2) }},
			{"cached_cosine/int8", func() float32 { return CachedCosineDistanceInt8(i1, m1, i2, m2) }},
			{"squared_euclidean/float32", func() float32 { return SquaredEuclideanDistanceFloat32(f1, f2) }},
			{"squared_euclidean/int8", func() float32 { return SquaredEuclideanDistanceInt8(i1, i2) }},
		}
		for _, k := range kernels {
			b.Run(fmt.Sprintf("%s/d%d", k.name, dims), func(b *testing.B) {
				for b.Loop() {
					k.run()
				}
			})
		}
	}
}
