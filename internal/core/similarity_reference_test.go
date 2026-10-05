package core

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The similarity loops are unrolled, so these check them against the textbook
// formulas in float64 on every length that can trip an unrolled loop: shorter
// than one block, exact blocks, and a remainder after them.

var unrollLengths = []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 127, 128, 129, 1536, 1537, 3072}

// signedFloats returns n values centred on zero at the scale of a unit-length
// 1536-d embedding, so sums mix signs as real ones do.
func signedFloats(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.025)
	}
	return x
}

func TestCosineSimilarity_MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic test data
	for _, dims := range unrollLengths {
		t.Run(fmt.Sprintf("d%d", dims), func(t *testing.T) {
			for range 50 {
				a, b := signedFloats(rng, dims), signedFloats(rng, dims)
				var dot, na, nb float64
				for i := range a {
					dot += float64(a[i]) * float64(b[i])
					na += float64(a[i]) * float64(a[i])
					nb += float64(b[i]) * float64(b[i])
				}

				got, err := CosineSimilarity(a, b)

				require.NoError(t, err)
				assert.InDelta(t, dot/(math.Sqrt(na)*math.Sqrt(nb)), got, 1e-5)
			}
		})
	}
}

func TestDotFloat32_MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic test data
	for _, dims := range unrollLengths {
		t.Run(fmt.Sprintf("d%d", dims), func(t *testing.T) {
			for range 50 {
				a, b := signedFloats(rng, dims), signedFloats(rng, dims)
				var dot float64
				for i := range a {
					dot += float64(a[i]) * float64(b[i])
				}

				assert.InDelta(t, dot, DotFloat32(a, b), 1e-12)
			}
		})
	}
}

func TestEuclideanSimilarity_MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic test data
	for _, dims := range unrollLengths {
		t.Run(fmt.Sprintf("d%d", dims), func(t *testing.T) {
			for range 50 {
				a, b := signedFloats(rng, dims), signedFloats(rng, dims)
				var sum float64
				for i := range a {
					d := float64(a[i]) - float64(b[i])
					sum += d * d
				}

				got, err := EuclideanSimilarity(a, b)

				require.NoError(t, err)
				assert.InDelta(t, sum, SquaredEuclideanFloat32(a, b), 1e-5*sum+1e-12)
				assert.InDelta(t, 1/(1+math.Sqrt(sum)), got, 1e-6)
			}
		})
	}
}

func BenchmarkEuclideanSimilarity(b *testing.B) {
	rng := rand.New(rand.NewPCG(14, 1536)) //nolint:gosec // deterministic benchmark data
	for _, dims := range []int{128, 1536} {
		x, y := signedFloats(rng, dims), signedFloats(rng, dims)
		b.Run(fmt.Sprintf("d%d", dims), func(b *testing.B) {
			for b.Loop() {
				_, _ = EuclideanSimilarity(x, y)
			}
		})
	}
}
