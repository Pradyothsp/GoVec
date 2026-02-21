package core

import (
	"math"
	"testing"
)

func TestCosineSimilarityInt8_IdenticalVectors(t *testing.T) {
	a := []int8{127, 63, 31}
	b := []int8{127, 63, 31}

	score, err := CosineSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Identical vectors should have similarity = 1.0
	if math.Abs(float64(score-1.0)) > 0.001 {
		t.Errorf("expected similarity ~1.0, got %f", score)
	}
}

func TestCosineSimilarityInt8_OrthogonalVectors(t *testing.T) {
	a := []int8{127, 0, 0}
	b := []int8{0, 127, 0}

	score, err := CosineSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Orthogonal vectors should have similarity = 0.0
	if math.Abs(float64(score)) > 0.001 {
		t.Errorf("expected similarity ~0.0, got %f", score)
	}
}

func TestCosineSimilarityInt8_OppositeVectors(t *testing.T) {
	a := []int8{127, 63, 31}
	b := []int8{-127, -63, -31}

	score, err := CosineSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Opposite vectors should have similarity = -1.0
	if math.Abs(float64(score+1.0)) > 0.001 {
		t.Errorf("expected similarity ~-1.0, got %f", score)
	}
}

func TestCosineSimilarityInt8_DimensionMismatch(t *testing.T) {
	a := []int8{127, 63}
	b := []int8{127, 63, 31}

	_, err := CosineSimilarityInt8(a, b)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
}

func TestCosineSimilarityInt8_ZeroVector(t *testing.T) {
	a := []int8{0, 0, 0}
	b := []int8{127, 63, 31}

	score, err := CosineSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Zero vectors should return 0.0 similarity (no meaningful similarity)
	if score != 0.0 {
		t.Errorf("expected score 0.0 for zero vector, got %f", score)
	}
}

func TestCosineSimilarityInt8_CompareWithFloat32(t *testing.T) {
	// Create float32 vectors
	vecA := []float32{0.5, 0.3, 0.8, -0.2}
	vecB := []float32{0.6, 0.2, 0.7, -0.1}

	// Quantize to int8
	quantA := QuantizeVector(vecA)
	quantB := QuantizeVector(vecB)

	// Calculate similarity for both types
	scoreFloat, err := CosineSimilarity(vecA, vecB)
	if err != nil {
		t.Fatalf("float32 similarity error: %v", err)
	}

	scoreInt8, err := CosineSimilarityInt8(quantA, quantB)
	if err != nil {
		t.Fatalf("int8 similarity error: %v", err)
	}

	// Scores should be similar (allow some precision loss)
	diff := math.Abs(float64(scoreFloat - scoreInt8))
	if diff > 0.1 {
		t.Errorf("similarity scores differ too much: float32=%f, int8=%f, diff=%f",
			scoreFloat, scoreInt8, diff)
	}

	t.Logf("Float32 similarity: %f", scoreFloat)
	t.Logf("Int8 similarity: %f", scoreInt8)
	t.Logf("Difference: %f", diff)
}

func TestCosineSimilarityInt8_NormalizedVectors(t *testing.T) {
	tests := []struct {
		name      string
		a         []int8
		b         []int8
		expected  float32
		tolerance float32
	}{
		{
			name:      "similar vectors",
			a:         []int8{127, 100, 50},
			b:         []int8{120, 95, 48},
			expected:  0.999,
			tolerance: 0.01,
		},
		{
			name:      "somewhat similar",
			a:         []int8{127, 0, 0},
			b:         []int8{90, 90, 0},
			expected:  0.707,
			tolerance: 0.01,
		},
		{
			name:      "different vectors",
			a:         []int8{127, 0, 0},
			b:         []int8{0, 0, 127},
			expected:  0.0,
			tolerance: 0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, err := CosineSimilarityInt8(tt.a, tt.b)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			diff := math.Abs(float64(score - tt.expected))
			if diff > float64(tt.tolerance) {
				t.Errorf("expected similarity ~%f, got %f (diff=%f)",
					tt.expected, score, diff)
			}
		})
	}
}

func TestCosineSimilarityInt8_LargeVectors(t *testing.T) {
	// Test with 1536 dimensions (common for embeddings)
	a := make([]int8, 1536)
	b := make([]int8, 1536)

	for i := range a {
		a[i] = int8((i % 127) - 63)
		b[i] = int8((i % 127) - 63)
	}

	score, err := CosineSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should be identical
	if math.Abs(float64(score-1.0)) > 0.001 {
		t.Errorf("expected similarity ~1.0, got %f", score)
	}
}

func BenchmarkCosineSimilarityInt8(b *testing.B) {
	vec1 := make([]int8, 1536)
	vec2 := make([]int8, 1536)
	for i := range vec1 {
		vec1[i] = int8(i % 127)
		vec2[i] = int8((i + 1) % 127)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = CosineSimilarityInt8(vec1, vec2)
	}
}

func BenchmarkCosineSimilarityFloat32(b *testing.B) {
	vec1 := make([]float32, 1536)
	vec2 := make([]float32, 1536)
	for i := range vec1 {
		vec1[i] = float32(i) / 1536.0
		vec2[i] = float32(i+1) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = CosineSimilarity(vec1, vec2)
	}
}
