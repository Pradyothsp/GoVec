package core

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name          string
		a             []float32
		b             []float32
		expectedScore float32
		expectError   bool
		errorMsg      string
	}{
		{
			name:          "identical vectors",
			a:             []float32{1.0, 2.0, 3.0},
			b:             []float32{1.0, 2.0, 3.0},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "opposite vectors",
			a:             []float32{1.0, 2.0, 3.0},
			b:             []float32{-1.0, -2.0, -3.0},
			expectedScore: -1.0,
			expectError:   false,
		},
		{
			name:          "orthogonal vectors",
			a:             []float32{1.0, 0.0},
			b:             []float32{0.0, 1.0},
			expectedScore: 0.0,
			expectError:   false,
		},
		{
			name:          "normalized vectors - high similarity",
			a:             []float32{0.6, 0.8},
			b:             []float32{0.6, 0.8},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "similar direction different magnitude",
			a:             []float32{1.0, 1.0},
			b:             []float32{2.0, 2.0},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:        "dimension mismatch",
			a:           []float32{1.0, 2.0},
			b:           []float32{1.0, 2.0, 3.0},
			expectError: true,
			errorMsg:    "vector dimensions mismatch",
		},
		{
			name:        "zero vector a",
			a:           []float32{0.0, 0.0, 0.0},
			b:           []float32{1.0, 2.0, 3.0},
			expectError: true,
			errorMsg:    "zero vector magnitude",
		},
		{
			name:        "zero vector b",
			a:           []float32{1.0, 2.0, 3.0},
			b:           []float32{0.0, 0.0, 0.0},
			expectError: true,
			errorMsg:    "zero vector magnitude",
		},
		{
			name:        "both zero vectors",
			a:           []float32{0.0, 0.0},
			b:           []float32{0.0, 0.0},
			expectError: true,
			errorMsg:    "zero vector magnitude",
		},
		{
			name:          "single dimension vectors",
			a:             []float32{5.0},
			b:             []float32{5.0},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "high dimensional vectors - identical",
			a:             []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			b:             []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "negative values",
			a:             []float32{-1.0, -2.0, -3.0},
			b:             []float32{-1.0, -2.0, -3.0},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "mixed positive and negative",
			a:             []float32{1.0, -2.0, 3.0},
			b:             []float32{1.0, -2.0, 3.0},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "very small values",
			a:             []float32{0.0001, 0.0002, 0.0003},
			b:             []float32{0.0001, 0.0002, 0.0003},
			expectedScore: 1.0,
			expectError:   false,
		},
		{
			name:          "partial similarity",
			a:             []float32{1.0, 0.0, 0.0},
			b:             []float32{1.0, 1.0, 0.0},
			expectedScore: float32(1.0 / math.Sqrt(2.0)), // ~0.707
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, err := CosineSimilarity(tt.a, tt.b)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				} else if err.Error() != tt.errorMsg {
					t.Errorf("expected error '%s', got '%s'", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				// Use tolerance for floating point comparison
				tolerance := float32(0.0001)
				if math.Abs(float64(score-tt.expectedScore)) > float64(tolerance) {
					t.Errorf("expected score %f, got %f", tt.expectedScore, score)
				}
			}
		})
	}
}

func TestCosineSimilarity_EdgeCases(t *testing.T) {
	t.Run("empty vectors", func(t *testing.T) {
		a := []float32{}
		b := []float32{}
		_, err := CosineSimilarity(a, b)
		if err == nil {
			t.Error("expected error for empty vectors")
		}
	})

	t.Run("one empty one non-empty", func(t *testing.T) {
		a := []float32{}
		b := []float32{1.0, 2.0}
		_, err := CosineSimilarity(a, b)
		if err == nil {
			t.Error("expected error for dimension mismatch")
		}
	})

	t.Run("128 dimensional vectors", func(t *testing.T) {
		a := make([]float32, 128)
		b := make([]float32, 128)
		for i := range a {
			a[i] = float32(i) / 128.0
			b[i] = float32(i) / 128.0
		}
		score, err := CosineSimilarity(a, b)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if math.Abs(float64(score-1.0)) > 0.0001 {
			t.Errorf("expected score 1.0 for identical vectors, got %f", score)
		}
	})

	t.Run("1536 dimensional vectors (OpenAI embeddings)", func(t *testing.T) {
		a := make([]float32, 1536)
		b := make([]float32, 1536)
		for i := range a {
			a[i] = 0.5
			b[i] = 0.5
		}
		score, err := CosineSimilarity(a, b)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if math.Abs(float64(score-1.0)) > 0.0001 {
			t.Errorf("expected score 1.0, got %f", score)
		}
	})
}

func TestCosineSimilarity_Symmetry(t *testing.T) {
	t.Run("cosine similarity should be symmetric", func(t *testing.T) {
		a := []float32{1.0, 2.0, 3.0, 4.0}
		b := []float32{5.0, 6.0, 7.0, 8.0}

		scoreAB, err1 := CosineSimilarity(a, b)
		scoreBA, err2 := CosineSimilarity(b, a)

		if err1 != nil || err2 != nil {
			t.Errorf("unexpected errors: %v, %v", err1, err2)
		}

		if scoreAB != scoreBA {
			t.Errorf("cosine similarity not symmetric: AB=%f, BA=%f", scoreAB, scoreBA)
		}
	})
}

func BenchmarkCosineSimilarity_128D(b *testing.B) {
	vec1 := make([]float32, 128)
	vec2 := make([]float32, 128)
	for i := range vec1 {
		vec1[i] = float32(i) / 128.0
		vec2[i] = float32(i) / 64.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(vec1, vec2)
	}
}

func BenchmarkCosineSimilarity_1536D(b *testing.B) {
	vec1 := make([]float32, 1536)
	vec2 := make([]float32, 1536)
	for i := range vec1 {
		vec1[i] = float32(i) / 1536.0
		vec2[i] = float32(i) / 768.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(vec1, vec2)
	}
}
