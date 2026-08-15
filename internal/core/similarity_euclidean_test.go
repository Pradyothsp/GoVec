package core

import (
	"math"
	"testing"
)

func TestEuclideanSimilarity_IdenticalVectors(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{1.0, 2.0, 3.0}

	score, err := EuclideanSimilarity(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Identical vectors have distance 0, so score = 1/(1+0) = 1.0
	if math.Abs(float64(score-1.0)) > 0.001 {
		t.Errorf("expected similarity 1.0, got %f", score)
	}
}

func TestEuclideanSimilarity_KnownDistance(t *testing.T) {
	a := []float32{0.0, 0.0}
	b := []float32{3.0, 4.0}

	score, err := EuclideanSimilarity(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// distance = sqrt(3^2 + 4^2) = 5, so score = 1/(1+5) = 1/6
	want := float32(1.0 / 6.0)
	if math.Abs(float64(score-want)) > 0.001 {
		t.Errorf("expected similarity %f, got %f", want, score)
	}
}

func TestEuclideanSimilarity_FartherIsLowerScore(t *testing.T) {
	origin := []float32{0.0, 0.0}
	near := []float32{1.0, 0.0}
	far := []float32{10.0, 0.0}

	nearScore, err := EuclideanSimilarity(origin, near)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	farScore, err := EuclideanSimilarity(origin, far)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if nearScore <= farScore {
		t.Errorf("expected nearer vector to score higher: near=%f far=%f", nearScore, farScore)
	}
}

func TestEuclideanSimilarity_DimensionMismatch(t *testing.T) {
	a := []float32{1.0, 2.0}
	b := []float32{1.0, 2.0, 3.0}

	_, err := EuclideanSimilarity(a, b)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
}

func TestEuclideanSimilarity_EmptyVectors(t *testing.T) {
	a := []float32{}
	b := []float32{}

	_, err := EuclideanSimilarity(a, b)
	if err == nil {
		t.Fatal("expected error for empty vectors, got nil")
	}
}

func TestEuclideanSimilarity_Symmetry(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{5.0, 6.0, 7.0, 8.0}

	scoreAB, err1 := EuclideanSimilarity(a, b)
	scoreBA, err2 := EuclideanSimilarity(b, a)
	if err1 != nil || err2 != nil {
		t.Fatalf("unexpected errors: %v, %v", err1, err2)
	}

	if scoreAB != scoreBA {
		t.Errorf("euclidean similarity not symmetric: AB=%f, BA=%f", scoreAB, scoreBA)
	}
}

func TestEuclideanSimilarityInt8_IdenticalVectors(t *testing.T) {
	a := []int8{127, 63, 31}
	b := []int8{127, 63, 31}

	score, err := EuclideanSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if math.Abs(float64(score-1.0)) > 0.001 {
		t.Errorf("expected similarity 1.0, got %f", score)
	}
}

func TestEuclideanSimilarityInt8_KnownDistance(t *testing.T) {
	a := []int8{0, 0}
	b := []int8{3, 4}

	score, err := EuclideanSimilarityInt8(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := float32(1.0 / 6.0)
	if math.Abs(float64(score-want)) > 0.001 {
		t.Errorf("expected similarity %f, got %f", want, score)
	}
}

func TestEuclideanSimilarityInt8_DimensionMismatch(t *testing.T) {
	a := []int8{127, 63}
	b := []int8{127, 63, 31}

	_, err := EuclideanSimilarityInt8(a, b)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
}
