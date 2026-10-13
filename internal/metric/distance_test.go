package metric

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
)

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

func TestEuclideanDistanceFloat32_MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic test data
	for _, dims := range []int{1, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 1536, 1537, 3072} {
		t.Run(fmt.Sprintf("d%d", dims), func(t *testing.T) {
			for range 50 {
				a, b := signedFloats(rng, dims), signedFloats(rng, dims)
				var sum float64
				for i := range a {
					d := float64(a[i]) - float64(b[i])
					sum += d * d
				}

				assert.InDelta(t, sum, SquaredEuclideanDistanceFloat32(a, b), 1e-5*sum)
				assert.InDelta(t, math.Sqrt(sum), EuclideanDistanceFloat32(a, b), 1e-5*math.Sqrt(sum))
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
