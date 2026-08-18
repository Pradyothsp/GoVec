package core

import (
	"fmt"
	"math"
)

// NormalizeVector L2-normalizes v to unit length. Cosine similarity is
// scale-invariant, so this can never change nearest-neighbor rankings under
// cosine distance -- it only brings v's per-dimension values within
// QuantizeVector's [-1, 1] precondition, which a unit-norm vector satisfies
// automatically (sum(x_i^2) = 1 forces every x_i into [-1, 1]). This is the
// same convention embedding APIs like OpenAI's already emit vectors in.
// Zero vectors are returned unchanged -- already within range, no direction
// to normalize toward.
func NormalizeVector(v []float32) []float32 {
	var sumSquares float64
	for _, val := range v {
		sumSquares += float64(val) * float64(val)
	}
	if sumSquares == 0 {
		return v
	}

	norm := float32(math.Sqrt(sumSquares))
	out := make([]float32, len(v))
	for i, val := range v {
		out[i] = val / norm
	}
	return out
}

// ValidateQuantizationRange rejects vectors QuantizeVector would otherwise
// silently clamp into a different vector. Only used ahead of Euclidean-metric
// scalar quantization, where auto-normalizing (safe under cosine) would
// discard magnitude and corrupt the distance calculation instead of just
// working around a range mismatch.
func ValidateQuantizationRange(v []float32) error {
	for i, val := range v {
		if val < -1.0 || val > 1.0 {
			return fmt.Errorf("scalar quantization with euclidean distance requires vector values in [-1, 1], got %v at dimension %d; normalize the vector before inserting, or use quantization: none", val, i)
		}
	}
	return nil
}

// QuantizeVector converts float32 vectors to int8 using scalar quantization
// Assumes input vectors are normalized to [-1, 1] range
func QuantizeVector(v []float32) []int8 {
	quantized := make([]int8, len(v))
	for i, val := range v {
		// Clamp to [-1, 1] range
		if val > 1.0 {
			val = 1.0
		}
		if val < -1.0 {
			val = -1.0
		}
		// Scale to int8 range [-127, 127]
		quantized[i] = int8(val * 127)
	}
	return quantized
}

// DequantizeVector converts int8 back to float32 (for debugging/testing)
func DequantizeVector(v []int8) []float32 {
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = float32(val) / 127.0
	}
	return result
}
