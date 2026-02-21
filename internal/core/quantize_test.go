package core

import (
	"math"
	"testing"
)

func TestQuantizeVector_Basic(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []int8
	}{
		{
			name:     "half values",
			input:    []float32{0.5, 0.5, 0.5},
			expected: []int8{63, 63, 63},
		},
		{
			name:     "negative half values",
			input:    []float32{-0.5, -0.5, -0.5},
			expected: []int8{-63, -63, -63},
		},
		{
			name:     "mixed values",
			input:    []float32{0.25, -0.25, 0.75, -0.75},
			expected: []int8{31, -31, 95, -95},
		},
		{
			name:     "zero vector",
			input:    []float32{0.0, 0.0, 0.0},
			expected: []int8{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := QuantizeVector(tt.input)
			if len(result) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestQuantizeVector_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []int8
	}{
		{
			name:     "max value",
			input:    []float32{1.0},
			expected: []int8{127},
		},
		{
			name:     "min value",
			input:    []float32{-1.0},
			expected: []int8{-127},
		},
		{
			name:     "near zero positive",
			input:    []float32{0.001},
			expected: []int8{0},
		},
		{
			name:     "near zero negative",
			input:    []float32{-0.001},
			expected: []int8{0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := QuantizeVector(tt.input)
			if len(result) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestQuantizeVector_Clamping(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []int8
	}{
		{
			name:     "overflow positive",
			input:    []float32{1.5, 2.0, 10.0},
			expected: []int8{127, 127, 127},
		},
		{
			name:     "overflow negative",
			input:    []float32{-1.5, -2.0, -10.0},
			expected: []int8{-127, -127, -127},
		},
		{
			name:     "mixed overflow",
			input:    []float32{0.5, 1.5, -0.5, -1.5},
			expected: []int8{63, 127, -63, -127},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := QuantizeVector(tt.input)
			if len(result) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestQuantizeVector_RoundTrip(t *testing.T) {
	input := []float32{0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5}
	quantized := QuantizeVector(input)
	dequantized := DequantizeVector(quantized)

	// Check precision loss is acceptable (within 1/127)
	tolerance := float32(1.0 / 127.0)
	for i := range input {
		diff := float32(math.Abs(float64(input[i] - dequantized[i])))
		if diff > tolerance {
			t.Errorf("index %d: precision loss too high: input=%f, dequantized=%f, diff=%f",
				i, input[i], dequantized[i], diff)
		}
	}
}

func TestDequantizeVector(t *testing.T) {
	tests := []struct {
		name     string
		input    []int8
		expected []float32
	}{
		{
			name:     "max value",
			input:    []int8{127},
			expected: []float32{1.0},
		},
		{
			name:     "min value",
			input:    []int8{-127},
			expected: []float32{-1.0},
		},
		{
			name:     "zero",
			input:    []int8{0},
			expected: []float32{0.0},
		},
		{
			name:     "half values",
			input:    []int8{63, -63},
			expected: []float32{0.496063, -0.496063},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DequantizeVector(tt.input)
			if len(result) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				diff := float32(math.Abs(float64(result[i] - tt.expected[i])))
				if diff > 0.001 {
					t.Errorf("index %d: got %f, want %f", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestQuantizeVector_Empty(t *testing.T) {
	input := []float32{}
	result := QuantizeVector(input)
	if len(result) != 0 {
		t.Errorf("expected empty result, got length %d", len(result))
	}
}

func TestQuantizeVector_LargeVector(t *testing.T) {
	// Test with 1536 dimensions (common for embeddings)
	input := make([]float32, 1536)
	for i := range input {
		input[i] = float32(i%256) / 256.0 // Values between 0 and 1
	}

	result := QuantizeVector(input)
	if len(result) != 1536 {
		t.Fatalf("length mismatch: got %d, want 1536", len(result))
	}

	// Verify some values
	for i := 0; i < 10; i++ {
		expected := int8(input[i] * 127)
		if result[i] != expected {
			t.Errorf("index %d: got %d, want %d", i, result[i], expected)
		}
	}
}
