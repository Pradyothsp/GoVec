package integration

import "math"

// generateTestVectors returns count deterministic vectors of the given
// dimension. The same (count, dim) always yields the same vectors, so tests
// and benchmarks built on them are reproducible.
//
// It lives here rather than in a benchmark file because the integration
// tests depend on it too: moving or build-tagging a benchmark file must not
// break them.
func generateTestVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			// Generate more diverse vectors using multiple frequency components
			// This creates vectors with different patterns and magnitudes
			vec[j] = float32(
				math.Sin(float64(i*j)/100.0)+
					math.Cos(float64(i+j*7)/50.0)*0.5+
					math.Sin(float64(i*13+j)/30.0)*0.3,
			) * 0.4
		}
		vectors[i] = vec
	}
	return vectors
}
