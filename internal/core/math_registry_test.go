package core

import "testing"

func TestResolveMetric_Cosine_ReturnsCosineFuncs(t *testing.T) {
	block, err := ResolveMetric("cosine")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if block.FloatFunc == nil || block.Int8Func == nil {
		t.Fatal("expected both FloatFunc and Int8Func to be set")
	}

	score, err := block.FloatFunc([]float32{1, 0}, []float32{1, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if score != 1.0 {
		t.Errorf("expected cosine similarity 1.0 for identical vectors, got %f", score)
	}
}

func TestResolveMetric_Euclidean_ReturnsEuclideanFuncs(t *testing.T) {
	block, err := ResolveMetric("euclidean")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if block.FloatFunc == nil || block.Int8Func == nil {
		t.Fatal("expected both FloatFunc and Int8Func to be set")
	}

	score, err := block.FloatFunc([]float32{1, 0}, []float32{1, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if score != 1.0 {
		t.Errorf("expected euclidean similarity 1.0 for identical vectors, got %f", score)
	}
}

func TestResolveMetric_Unknown_ReturnsError(t *testing.T) {
	_, err := ResolveMetric("manhattan")
	if err == nil {
		t.Fatal("expected error for unknown metric, got nil")
	}
}
