package core

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
