package core

import (
	"os"
	"testing"

	"github.com/Pradyothsp/govec/internal/config"
)

func TestNewEngine_NoneQuantization(t *testing.T) {
	// Create temporary WAL file
	walPath := t.TempDir() + "/test.wal"
	wal, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()
	defer os.Remove(walPath)

	cfg := config.EngineConfig{
		Quantization:   "none",
		DistanceMetric: "cosine",
	}

	engine, err := NewEngine(cfg, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	if engine == nil {
		t.Fatal("expected non-nil engine")
	}

	// Verify we can insert and search
	vec := []float32{0.5, 0.5, 0.5}
	err = engine.Insert("test1", vec, nil)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	results, err := engine.Search(vec, 1, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if results[0].ID != "test1" {
		t.Errorf("expected ID 'test1', got '%s'", results[0].ID)
	}
}

func TestNewEngine_ScalarQuantization(t *testing.T) {
	// Create temporary WAL file
	walPath := t.TempDir() + "/test.wal"
	wal, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()
	defer os.Remove(walPath)

	cfg := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engine, err := NewEngine(cfg, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	if engine == nil {
		t.Fatal("expected non-nil engine")
	}

	// Verify we can insert and search
	vec := []float32{0.5, 0.5, 0.5}
	err = engine.Insert("test1", vec, nil)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	results, err := engine.Search(vec, 1, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if results[0].ID != "test1" {
		t.Errorf("expected ID 'test1', got '%s'", results[0].ID)
	}
}

func TestNewEngine_InvalidQuantization(t *testing.T) {
	// Create temporary WAL file
	walPath := t.TempDir() + "/test.wal"
	wal, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()
	defer os.Remove(walPath)

	cfg := config.EngineConfig{
		Quantization:   "invalid",
		DistanceMetric: "cosine",
	}

	_, err = NewEngine(cfg, wal)
	if err == nil {
		t.Fatal("expected error for invalid quantization, got nil")
	}
}

func TestNewEngine_UnsupportedMetric(t *testing.T) {
	// Create temporary WAL file
	walPath := t.TempDir() + "/test.wal"
	wal, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()
	defer os.Remove(walPath)

	cfg := config.EngineConfig{
		Quantization:   "none",
		DistanceMetric: "euclidean", // Not supported yet
	}

	_, err = NewEngine(cfg, wal)
	if err == nil {
		t.Fatal("expected error for unsupported metric, got nil")
	}
}

func TestNewEngine_CompareQuantizationResults(t *testing.T) {
	// Create two engines: one with float32, one with int8
	walPath1 := t.TempDir() + "/test1.wal"
	wal1, err := NewWAL(walPath1)
	if err != nil {
		t.Fatalf("failed to create WAL1: %v", err)
	}
	defer wal1.Close()
	defer os.Remove(walPath1)

	walPath2 := t.TempDir() + "/test2.wal"
	wal2, err := NewWAL(walPath2)
	if err != nil {
		t.Fatalf("failed to create WAL2: %v", err)
	}
	defer wal2.Close()
	defer os.Remove(walPath2)

	cfg1 := config.EngineConfig{
		Quantization:   "none",
		DistanceMetric: "cosine",
	}

	cfg2 := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engineFloat, err := NewEngine(cfg1, wal1)
	if err != nil {
		t.Fatalf("NewEngine (float) failed: %v", err)
	}

	engineInt8, err := NewEngine(cfg2, wal2)
	if err != nil {
		t.Fatalf("NewEngine (int8) failed: %v", err)
	}

	// Insert same data into both
	testVectors := []struct {
		id  string
		vec []float32
	}{
		{"vec1", []float32{0.5, 0.3, 0.8}},
		{"vec2", []float32{0.6, 0.2, 0.7}},
		{"vec3", []float32{0.1, 0.9, 0.4}},
	}

	for _, tv := range testVectors {
		if err := engineFloat.Insert(tv.id, tv.vec, nil); err != nil {
			t.Fatalf("Insert to engineFloat failed: %v", err)
		}
		if err := engineInt8.Insert(tv.id, tv.vec, nil); err != nil {
			t.Fatalf("Insert to engineInt8 failed: %v", err)
		}
	}

	// Search with same query
	query := []float32{0.55, 0.25, 0.75}

	resultsFloat, err := engineFloat.Search(query, 3, nil)
	if err != nil {
		t.Fatalf("Search (float) failed: %v", err)
	}

	resultsInt8, err := engineInt8.Search(query, 3, nil)
	if err != nil {
		t.Fatalf("Search (int8) failed: %v", err)
	}

	// Results should have same ordering (or very similar)
	if len(resultsFloat) != len(resultsInt8) {
		t.Fatalf("result count mismatch: float=%d, int8=%d", len(resultsFloat), len(resultsInt8))
	}

	t.Logf("Float32 results: %+v", resultsFloat)
	t.Logf("Int8 results: %+v", resultsInt8)

	// First result should be the same (most similar vector)
	if resultsFloat[0].ID != resultsInt8[0].ID {
		t.Logf("Warning: Top result differs - float=%s (score=%f), int8=%s (score=%f)",
			resultsFloat[0].ID, resultsFloat[0].Score,
			resultsInt8[0].ID, resultsInt8[0].Score)
	}
}
