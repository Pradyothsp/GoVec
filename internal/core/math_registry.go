package core

import "fmt"

// MathBlock contains distance functions for different vector types
type MathBlock struct {
	FloatFunc func([]float32, []float32) (float32, error)
	Int8Func  func([]int8, []int8) (float32, error) // For Stage 2
}

// resolveMetric returns the appropriate math block for the given distance metric
func resolveMetric(metric string) (MathBlock, error) {
	switch metric {
	case "cosine":
		return MathBlock{
			FloatFunc: CosineSimilarity,
			Int8Func:  nil, // Will be implemented in Stage 2
		}, nil
	default:
		return MathBlock{}, fmt.Errorf("unknown distance metric: %s", metric)
	}
}
