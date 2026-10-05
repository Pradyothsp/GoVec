package hnsw

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEuclideanDistanceFloat32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	require.Equal(t, float32(5.196152), EuclideanDistanceFloat32(a, b))
}

func TestCosineDistanceFloat32(t *testing.T) {
	var a, b []float32
	// Same magnitude, same direction.
	a = []float32{1, 1, 1}
	b = []float32{0.8, 0.8, 0.8}
	require.InDelta(t, 0, CosineDistanceFloat32(a, b), 0.000001)

	// Perpendicular vectors.
	a = []float32{1, 0}
	b = []float32{0, 1}
	require.InDelta(t, 1, CosineDistanceFloat32(a, b), 0.000001)

	// Equivalent vectors.
	a = []float32{1, 0}
	b = []float32{1, 0}
	require.InDelta(t, 0, CosineDistanceFloat32(a, b), 0.000001)
}

func BenchmarkCosineDistanceFloat32(b *testing.B) {
	v1 := randFloats(1536)
	v2 := randFloats(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineDistanceFloat32(v1, v2)
	}
}

// signedFloats returns n values centred on zero at the scale of a unit-length
// 1536-d embedding, so dot products mix signs as real ones do.
func signedFloats(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.025)
	}
	return x
}

// cosineDistanceReference is the textbook formula in float64, the yardstick
// for the fast float32 kernel.
func cosineDistanceReference(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
}

func cachedCosine(a, b []float32) float32 {
	return CachedCosineDistanceFloat32(a, PrecomputeSquaredNormFloat32(a), b, PrecomputeSquaredNormFloat32(b))
}

// TestCachedCosineDistanceFloat32_MatchesReference covers every length an
// unrolled loop can get wrong: shorter than one block, exact blocks, and a
// remainder after them.
func TestCachedCosineDistanceFloat32_MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic test data
	for _, dims := range []int{1, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 1536, 1537, 3072} {
		t.Run(fmt.Sprintf("d%d", dims), func(t *testing.T) {
			for range 50 {
				a, b := signedFloats(rng, dims), signedFloats(rng, dims)

				assert.InDelta(t, cosineDistanceReference(a, b), cachedCosine(a, b), 1e-5)
			}
		})
	}
}

func TestCachedCosineDistanceFloat32_ExactCases(t *testing.T) {
	a := []float32{0.5, -1, 2, 0, 3, -0.25, 1, 1, 4}
	opposite := make([]float32, len(a))
	scaled := make([]float32, len(a))
	for i, v := range a {
		opposite[i], scaled[i] = -v, 3*v
	}
	orthogonal := []float32{1, 0.5, 0, 0, 0, 0, 0, 0, 0} // a·orthogonal = 0.5 - 0.5 = 0

	assert.InDelta(t, 0, cachedCosine(a, a), 1e-6, "identical")
	assert.InDelta(t, 0, cachedCosine(a, scaled), 1e-6, "same direction, different length")
	assert.InDelta(t, 2, cachedCosine(a, opposite), 1e-6, "opposite")
	assert.InDelta(t, 1, cachedCosine(a, orthogonal), 1e-6, "orthogonal")

	// No zero-vector guard, matching vek32.CosineSimilarity: 0/0 is NaN.
	assert.True(t, math.IsNaN(float64(cachedCosine(a, make([]float32, len(a))))))
}

// BenchmarkCachedCosineDistanceFloat32 measures the cosine kernel HNSW calls
// on every comparison:
//
//	go test ./internal/hnsw/ -run '^$' -bench CachedCosine
func BenchmarkCachedCosineDistanceFloat32(b *testing.B) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic benchmark data
	for _, dims := range []int{128, 1536} {
		x, y := signedFloats(rng, dims), signedFloats(rng, dims)
		nx, ny := PrecomputeSquaredNormFloat32(x), PrecomputeSquaredNormFloat32(y)
		b.Run(fmt.Sprintf("d%d", dims), func(b *testing.B) {
			for b.Loop() {
				CachedCosineDistanceFloat32(x, nx, y, ny)
			}
		})
	}
}
