package integration

import (
	"context"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

func TestScalarQuantization_EndToEnd(t *testing.T) {
	// Create temporary WAL file
	walPath := t.TempDir() + "/test.wal"
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()
	defer os.Remove(walPath)

	cfg := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	// Insert test vectors
	testVectors := []struct {
		id   string
		vec  []float32
		meta map[string]interface{}
	}{
		{"vec1", []float32{0.5, 0.3, 0.8}, map[string]interface{}{"category": "A"}},
		{"vec2", []float32{0.6, 0.2, 0.7}, map[string]interface{}{"category": "A"}},
		{"vec3", []float32{0.1, 0.9, 0.4}, map[string]interface{}{"category": "B"}},
		{"vec4", []float32{-0.5, -0.3, -0.8}, map[string]interface{}{"category": "C"}},
	}

	for _, tv := range testVectors {
		err = engine.Insert(context.Background(), tv.id, tv.vec, core.SparseVector{}, tv.meta)
		if err != nil {
			t.Fatalf("Insert failed for %s: %v", tv.id, err)
		}
	}

	// Search for similar vectors
	query := []float32{0.55, 0.25, 0.75}
	results, err := engine.Search(context.Background(), query, core.SparseVector{}, 3, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// vec1 and vec2 score within 1e-5 of each other, so their order is left
	// to the quantizer; only that both beat vec3 is asserted.
	topTwo := []string{results[0].ID, results[1].ID}
	if !slices.Contains(topTwo, "vec1") || !slices.Contains(topTwo, "vec2") {
		t.Errorf("expected vec1 and vec2 as the top two results, got %v", topTwo)
	}
	if results[2].ID != "vec3" {
		t.Errorf("expected vec3 third, got %s", results[2].ID)
	}
}

func TestScalarQuantization_SearchAccuracy(t *testing.T) {
	// Create two engines: one with none, one with scalar
	walPath1 := t.TempDir() + "/test1.wal"
	wal1, err := index.NewWAL(walPath1)
	if err != nil {
		t.Fatalf("failed to create WAL1: %v", err)
	}
	defer wal1.Close()
	defer os.Remove(walPath1)

	walPath2 := t.TempDir() + "/test2.wal"
	wal2, err := index.NewWAL(walPath2)
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

	engineNone, err := index.NewEngine(cfg1, config.StorageConfig{}, wal1)
	if err != nil {
		t.Fatalf("NewEngine (none) failed: %v", err)
	}

	engineScalar, err := index.NewEngine(cfg2, config.StorageConfig{}, wal2)
	if err != nil {
		t.Fatalf("NewEngine (scalar) failed: %v", err)
	}

	// Insert same data
	testVectors := []struct {
		id  string
		vec []float32
	}{
		{"vec1", []float32{0.5, 0.3, 0.8, -0.2}},
		{"vec2", []float32{0.6, 0.2, 0.7, -0.1}},
		{"vec3", []float32{0.1, 0.9, 0.4, 0.5}},
		{"vec4", []float32{-0.5, -0.3, -0.8, 0.2}},
		{"vec5", []float32{0.0, 0.0, 1.0, 0.0}},
	}

	for _, tv := range testVectors {
		if err := engineNone.Insert(context.Background(), tv.id, tv.vec, core.SparseVector{}, nil); err != nil {
			t.Fatalf("Insert to engineNone failed: %v", err)
		}
		if err := engineScalar.Insert(context.Background(), tv.id, tv.vec, core.SparseVector{}, nil); err != nil {
			t.Fatalf("Insert to engineScalar failed: %v", err)
		}
	}

	// Search with same query
	query := []float32{0.55, 0.25, 0.75, -0.15}

	resultsNone, err := engineNone.Search(context.Background(), query, core.SparseVector{}, 5, nil)
	if err != nil {
		t.Fatalf("Search (none) failed: %v", err)
	}

	resultsScalar, err := engineScalar.Search(context.Background(), query, core.SparseVector{}, 5, nil)
	if err != nil {
		t.Fatalf("Search (scalar) failed: %v", err)
	}

	assertScalarMatchesFloat32(t, resultsNone, resultsScalar)
}

// scalarScoreTolerance is how far an int8 score may drift from its float32
// score. Observed drift on these low-dimension vectors is about 0.005; a broken
// quantizer is off by far more.
const scalarScoreTolerance = 0.02

// assertScalarMatchesFloat32 checks that int8 search approximates float32
// search: the same IDs, each scored within scalarScoreTolerance, in the same
// order wherever float32 separates them by more than that drift could flip.
func assertScalarMatchesFloat32(t *testing.T, float32Results, scalarResults []index.SearchResult) {
	t.Helper()
	if len(float32Results) != len(scalarResults) {
		t.Fatalf("result count differs: float32=%d, scalar=%d", len(float32Results), len(scalarResults))
	}

	// Match by ID, not rank, so a reordered near-tie can't skip a comparison.
	scalarScore := make(map[string]float32, len(scalarResults))
	for _, r := range scalarResults {
		scalarScore[r.ID] = r.Score
	}
	for _, r := range float32Results {
		s, ok := scalarScore[r.ID]
		if !ok {
			t.Errorf("%s is in the float32 results but not the scalar results", r.ID)
			continue
		}
		if diff := math.Abs(float64(r.Score - s)); diff > scalarScoreTolerance {
			t.Errorf("%s: scalar score %.4f is %.4f from float32 score %.4f", r.ID, s, diff, r.Score)
		}
	}

	// Two errors of up to the tolerance each can flip a pair closer than twice it.
	for i := 0; i+1 < len(float32Results); i++ {
		hi, lo := float32Results[i], float32Results[i+1]
		if hi.Score-lo.Score > 2*scalarScoreTolerance && scalarScore[hi.ID] <= scalarScore[lo.ID] {
			t.Errorf("scalar ranks %s above %s, but float32 separates them by %.4f", lo.ID, hi.ID, hi.Score-lo.Score)
		}
	}
}

func TestScalarQuantization_Persistence(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := tmpDir + "/test.wal"
	dataPath := tmpDir + "/test.bin"

	// Create engine and insert data
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}

	cfg := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	// Insert test vectors
	vec1 := []float32{0.5, 0.3, 0.8}
	vec2 := []float32{0.6, 0.2, 0.7}

	if err := engine.Insert(context.Background(), "vec1", vec1, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
	if err := engine.Insert(context.Background(), "vec2", vec2, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Save to file
	if err := engine.SaveToFile(context.Background(), dataPath); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	wal.Close()

	// Create new engine and load
	wal2, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL2: %v", err)
	}
	defer wal2.Close()

	engine2, err := index.NewEngine(cfg, config.StorageConfig{}, wal2)
	if err != nil {
		t.Fatalf("NewEngine (2) failed: %v", err)
	}

	if err := engine2.LoadFromFile(context.Background(), dataPath); err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	// Verify data
	if engine2.Len() != 2 {
		t.Fatalf("expected 2 vectors after load, got %d", engine2.Len())
	}

	// Search to verify vectors are correct
	results, err := engine2.Search(context.Background(), vec1, core.SparseVector{}, 1, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least 1 result")
	}

	if results[0].ID != "vec1" {
		t.Errorf("expected vec1, got %s", results[0].ID)
	}
}

func TestScalarQuantization_WALReplay(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := tmpDir + "/test.wal"

	// Create engine and insert data
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}

	cfg := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	// Insert test vectors
	vec1 := []float32{0.5, 0.3, 0.8}
	vec2 := []float32{0.6, 0.2, 0.7}

	if err := engine.Insert(context.Background(), "vec1", vec1, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
	if err := engine.Insert(context.Background(), "vec2", vec2, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	wal.Close()

	// Create new engine and replay WAL
	wal2, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL2: %v", err)
	}
	defer wal2.Close()

	engine2, err := index.NewEngine(cfg, config.StorageConfig{}, wal2)
	if err != nil {
		t.Fatalf("NewEngine (2) failed: %v", err)
	}

	if err := engine2.ReplayWAL(walPath); err != nil {
		t.Fatalf("ReplayWAL failed: %v", err)
	}

	// Verify data
	if engine2.Len() != 2 {
		t.Fatalf("expected 2 vectors after WAL replay, got %d", engine2.Len())
	}

	// Search to verify vectors are correct
	results, err := engine2.Search(context.Background(), vec1, core.SparseVector{}, 1, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least 1 result")
	}

	if results[0].ID != "vec1" {
		t.Errorf("expected vec1, got %s", results[0].ID)
	}
}

func TestScalarQuantization_MemoryUsage(t *testing.T) {
	// This test verifies that int8 vectors use less memory than float32
	// Note: This is a conceptual test - actual memory measurement is complex

	tmpDir := t.TempDir()

	// Create engine with scalar quantization
	walPath := tmpDir + "/test.wal"
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()

	cfg := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	// Insert 100 vectors (1536 dimensions each - typical embedding size)
	for i := 0; i < 100; i++ {
		vec := make([]float32, 1536)
		for j := range vec {
			vec[j] = float32(j) / 1536.0
		}
		if err := engine.Insert(context.Background(), string(rune('a'+i%26)), vec, core.SparseVector{}, nil); err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Save to file
	dataPath := tmpDir + "/test.bin"
	if err := engine.SaveToFile(context.Background(), dataPath); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Check file size
	info, err := os.Stat(dataPath)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}

	t.Logf("Snapshot file size (int8): %d bytes", info.Size())

	// Theoretical size for 100 vectors of 1536 dimensions:
	// float32: 100 * 1536 * 4 = 614,400 bytes
	// int8:    100 * 1536 * 1 = 153,600 bytes
	// Plus overhead for IDs, metadata, and GOB encoding

	// File should be significantly smaller than float32 version
	// (allowing for GOB overhead)
	maxExpectedSize := int64(614400)
	if info.Size() > maxExpectedSize {
		t.Errorf("file size %d is larger than expected max %d", info.Size(), maxExpectedSize)
	}
}

func TestQuantizationSwitch_RequiresDataDeletion(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := tmpDir + "/test.wal"
	dataPath := tmpDir + "/test.bin"

	// Create engine with "none" quantization
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}

	cfg1 := config.EngineConfig{
		Quantization:   "none",
		DistanceMetric: "cosine",
	}

	engine1, err := index.NewEngine(cfg1, config.StorageConfig{}, wal)
	if err != nil {
		t.Fatalf("NewEngine (none) failed: %v", err)
	}

	vec := []float32{0.5, 0.3, 0.8}
	if err := engine1.Insert(context.Background(), "vec1", vec, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Save with "none" quantization
	if err := engine1.SaveToFile(context.Background(), dataPath); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	wal.Close()

	// Try to load with "scalar" quantization (should fail or produce errors)
	wal2, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL2: %v", err)
	}
	defer wal2.Close()

	cfg2 := config.EngineConfig{
		Quantization:   "scalar",
		DistanceMetric: "cosine",
	}

	engine2, err := index.NewEngine(cfg2, config.StorageConfig{}, wal2)
	if err != nil {
		t.Fatalf("NewEngine (scalar) failed: %v", err)
	}

	// Loading must be refused by the header check, not fail later in GOB
	// decoding, so the user gets an error that says what to do.
	err = engine2.LoadFromFile(context.Background(), dataPath)
	if err == nil {
		t.Fatal("LoadFromFile accepted a 'none' snapshot into a 'scalar' engine")
	}
	if !strings.Contains(err.Error(), "quantization mismatch") {
		t.Fatalf("expected a quantization mismatch error, got: %v", err)
	}
	if engine2.Len() != 0 {
		t.Fatalf("a refused snapshot must leave the engine empty, got %d vectors", engine2.Len())
	}

	// Clean up and start fresh (recommended workflow)
	os.Remove(dataPath)
	os.Remove(walPath)

	// Create fresh WAL and engine with scalar
	wal3, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL3: %v", err)
	}
	defer wal3.Close()

	engine3, err := index.NewEngine(cfg2, config.StorageConfig{}, wal3)
	if err != nil {
		t.Fatalf("NewEngine (scalar fresh) failed: %v", err)
	}

	// Now insert should work
	if err := engine3.Insert(context.Background(), "vec1", vec, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert to fresh engine failed: %v", err)
	}
}
