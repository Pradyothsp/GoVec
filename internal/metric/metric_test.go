package metric

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// metricPairs returns deterministic vector pairs for checking a metric's
// functions against each other, in float32 and quantized to int8.
func metricPairs() (f32 [][2][]float32, i8 [][2][]int8) {
	rng := rand.New(rand.NewPCG(7, 128)) //nolint:gosec // deterministic test data
	for range 20 {
		a, b := signedFloats(rng, 128), signedFloats(rng, 128)
		f32 = append(f32, [2][]float32{a, b})
		i8 = append(i8, [2][]int8{core.QuantizeVector(core.NormalizeVector(a)), core.QuantizeVector(core.NormalizeVector(b))})
	}

	return f32, i8
}

// checkMetric checks that m's functions agree: Score(Distance) is Similarity,
// and each optional pair, when set, gives Distance's answer.
func checkMetric[T VectorType](t *testing.T, m Metric[T], pairs [][2]T) {
	t.Helper()

	require.NotNil(t, m.Similarity)
	require.NotNil(t, m.Distance)
	require.NotNil(t, m.Score)
	require.Equal(t, m.Precompute == nil, m.CachedDistance == nil, "Precompute and CachedDistance are set together")
	require.Equal(t, m.SquaredDistance == nil, m.FromSquaredDistance == nil, "SquaredDistance and FromSquaredDistance are set together")

	for _, p := range pairs {
		a, b := p[0], p[1]
		distance := m.Distance(a, b)

		similarity, err := m.Similarity(a, b)
		require.NoError(t, err)
		assert.InDelta(t, similarity, m.Score(distance), 1e-5, "Score(Distance) must match Similarity")

		if m.CachedDistance != nil {
			cached := m.CachedDistance(a, m.Precompute(a), b, m.Precompute(b))
			assert.InDelta(t, distance, cached, 1e-5, "CachedDistance must match Distance")
		}
		if m.SquaredDistance != nil {
			assert.InDelta(t, distance, m.FromSquaredDistance(m.SquaredDistance(a, b)), 1e-3, "FromSquaredDistance(SquaredDistance) must match Distance")
		}
	}
}

// Every metric, for both vector types, must score a match the same whether
// brute force (Similarity) or HNSW (Score of Distance) found it.
func TestMetrics_FunctionsAgree(t *testing.T) {
	f32, i8 := metricPairs()
	for _, name := range Names() {
		t.Run(name+"/float32", func(t *testing.T) {
			// Arrange
			m, err := Resolve[[]float32](Name(name))
			require.NoError(t, err)

			// Act + Assert
			checkMetric(t, m, f32)
		})
		t.Run(name+"/int8", func(t *testing.T) {
			// Arrange
			m, err := Resolve[[]int8](Name(name))
			require.NoError(t, err)

			// Act + Assert
			checkMetric(t, m, i8)
		})
	}
}

func TestResolve_SetsNameAndScore(t *testing.T) {
	tests := []struct {
		name     Name
		distance float32
		want     float32
	}{
		{Cosine, 0.25, 0.75},  // 1 - distance
		{Euclidean, 4, 0.2},   // 1 / (1 + distance)
		{Euclidean, 99, 0.01}, // farther is lower, never negative
	}
	for _, tt := range tests {
		t.Run(string(tt.name), func(t *testing.T) {
			// Act
			m, err := Resolve[[]float32](tt.name)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, tt.name, m.Name)
			assert.InDelta(t, tt.want, m.Score(tt.distance), 1e-4)
		})
	}
}

func TestResolve_Unknown(t *testing.T) {
	// Act
	_, err := Resolve[[]int8]("manhattan")

	// Assert
	require.ErrorContains(t, err, "unknown distance metric 'manhattan', must be one of: cosine, euclidean")
	assert.False(t, IsKnown("manhattan"))
	assert.True(t, IsKnown(Cosine))
}
