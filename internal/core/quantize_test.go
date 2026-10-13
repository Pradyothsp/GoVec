package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// QuantizeVector maps [-1, 1] onto [-127, 127], truncating toward zero, and
// clamps anything outside the range.
func TestQuantizeVector(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want []int8
	}{
		{"empty", []float32{}, []int8{}},
		{"zero", []float32{0, 0}, []int8{0, 0}},
		{"range ends", []float32{1, -1}, []int8{127, -127}},
		{"inside the range", []float32{0.5, -0.5, 0.25, -0.75}, []int8{63, -63, 31, -95}},
		{"below one step truncates to zero", []float32{0.001, -0.001}, []int8{0, 0}},
		{"outside the range clamps", []float32{1.5, 10, -1.5, -10}, []int8{127, 127, -127, -127}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			got := QuantizeVector(tt.in)

			// Assert
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDequantizeVector(t *testing.T) {
	// Act
	got := DequantizeVector([]int8{127, -127, 0, 63})

	// Assert
	assert.InDeltaSlice(t, []float32{1, -1, 0, 63.0 / 127}, got, 1e-6)
}

// A round trip loses at most one quantization step per value.
func TestQuantizeVector_RoundTrip(t *testing.T) {
	// Arrange
	in := []float32{0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5}

	// Act
	got := DequantizeVector(QuantizeVector(in))

	// Assert
	assert.InDeltaSlice(t, in, got, 1.0/127)
}

// Normalizing gives unit length, which puts every value in QuantizeVector's
// range, and keeps the direction; a zero vector has none and comes back as is.
func TestNormalizeVector(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want []float32
	}{
		{"3-4-5", []float32{3, 4}, []float32{0.6, 0.8}},
		{"already unit", []float32{0, -1, 0}, []float32{0, -1, 0}},
		{"zero vector", []float32{0, 0}, []float32{0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			got := NormalizeVector(tt.in)

			// Assert
			assert.InDeltaSlice(t, tt.want, got, 1e-6)
		})
	}
}

// Under Euclidean distance a vector can't be normalized into range without
// changing distances, so one outside [-1, 1] is refused rather than clamped.
func TestValidateQuantizationRange(t *testing.T) {
	// Act
	inRange := ValidateQuantizationRange([]float32{1, -1, 0.5})
	outOfRange := ValidateQuantizationRange([]float32{0.5, 1.5})

	// Assert
	require.NoError(t, inRange)
	require.Error(t, outOfRange)
	assert.Contains(t, outOfRange.Error(), "dimension 1", "the error names the offending dimension")
}
