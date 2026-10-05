package index

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
)

// TestNewEngine builds every engine and quantization combination and checks
// it stores, finds and reports what it was configured as.
func TestNewEngine(t *testing.T) {
	ctx := context.Background()
	for _, indexType := range parityIndexTypes {
		for _, quant := range []config.Quantization{config.QuantizationNone, config.QuantizationScalar} {
			t.Run(string(indexType)+"/"+string(quant), func(t *testing.T) {
				// Arrange
				engine := newParityEngine(t,
					config.EngineConfig{IndexType: indexType, Quantization: quant, DistanceMetric: config.DistanceMetricCosine},
					config.StorageConfig{})
				vec := []float32{0.5, 0.5, 0.5}

				// Act
				insertErr := engine.Insert(ctx, "test1", vec, core.SparseVector{}, nil)
				results, searchErr := engine.Search(ctx, vec, core.SparseVector{}, 1, nil)

				// Assert
				require.NoError(t, insertErr)
				require.NoError(t, searchErr)
				require.Len(t, results, 1)
				assert.Equal(t, "test1", results[0].ID)
				info := engine.Info()
				assert.Equal(t, string(indexType), info.IndexType)
				assert.Equal(t, string(quant), info.Quantization)
			})
		}
	}
}

func TestNewEngine_RejectsUnknownSettings(t *testing.T) {
	tests := []struct {
		name string
		cfg  config.EngineConfig
	}{
		{"unknown quantization", config.EngineConfig{Quantization: "invalid", DistanceMetric: config.DistanceMetricCosine}},
		{"unknown metric", config.EngineConfig{Quantization: config.QuantizationNone, DistanceMetric: "manhattan"}},
	}
	for _, indexType := range parityIndexTypes {
		for _, tt := range tests {
			t.Run(string(indexType)+"/"+tt.name, func(t *testing.T) {
				// Arrange
				tt.cfg.IndexType = indexType

				// Act
				_, err := NewEngine(tt.cfg, config.StorageConfig{}, newTestWAL(t))

				// Assert
				assert.Error(t, err)
			})
		}
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

	engineFloat, err := NewEngine(cfg1, config.StorageConfig{}, wal1)
	if err != nil {
		t.Fatalf("NewEngine (float) failed: %v", err)
	}

	engineInt8, err := NewEngine(cfg2, config.StorageConfig{}, wal2)
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
		if err := engineFloat.Insert(context.Background(), tv.id, tv.vec, core.SparseVector{}, nil); err != nil {
			t.Fatalf("Insert to engineFloat failed: %v", err)
		}
		if err := engineInt8.Insert(context.Background(), tv.id, tv.vec, core.SparseVector{}, nil); err != nil {
			t.Fatalf("Insert to engineInt8 failed: %v", err)
		}
	}

	// Search with same query
	query := []float32{0.55, 0.25, 0.75}

	resultsFloat, err := engineFloat.Search(context.Background(), query, core.SparseVector{}, 3, nil)
	if err != nil {
		t.Fatalf("Search (float) failed: %v", err)
	}

	resultsInt8, err := engineInt8.Search(context.Background(), query, core.SparseVector{}, 3, nil)
	if err != nil {
		t.Fatalf("Search (int8) failed: %v", err)
	}

	// Results should have same ordering (or very similar)
	if len(resultsFloat) != len(resultsInt8) {
		t.Fatalf("result count mismatch: float=%d, int8=%d", len(resultsFloat), len(resultsInt8))
	}

	// Match by ID, not rank: vec1 and vec2 score within 1e-5 under int8, so
	// their order is left to the quantizer. Observed drift is about 0.008.
	const tolerance = 0.02
	int8Score := make(map[string]float32, len(resultsInt8))
	for _, r := range resultsInt8 {
		int8Score[r.ID] = r.Score
	}
	for _, r := range resultsFloat {
		s, ok := int8Score[r.ID]
		if !ok {
			t.Errorf("%s is in the float32 results but not the int8 results", r.ID)
		} else if diff := math.Abs(float64(r.Score - s)); diff > tolerance {
			t.Errorf("%s: int8 score %.4f is %.4f from float32 score %.4f", r.ID, s, diff, r.Score)
		}
	}
	if resultsInt8[2].ID != "vec3" {
		t.Errorf("expected vec3 last under int8, got %s", resultsInt8[2].ID)
	}
}

func TestNewEngine_HNSW_EuclideanMetric(t *testing.T) {
	walPath := t.TempDir() + "/test.wal"
	wal, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()
	defer os.Remove(walPath)

	cfg := config.EngineConfig{
		IndexType:      config.IndexTypeHNSW,
		Quantization:   "none",
		DistanceMetric: config.DistanceMetricEuclidean,
	}

	engine, err := NewEngine(cfg, config.StorageConfig{}, wal)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	if engine.Info().DistanceMetric != string(config.DistanceMetricEuclidean) {
		t.Fatalf("expected distance metric 'euclidean', got '%s'", engine.Info().DistanceMetric)
	}

	origin := []float32{0, 0, 0}
	near := []float32{1, 0, 0}
	far := []float32{10, 0, 0}

	if err := engine.Insert(context.Background(), "near", near, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert(near) failed: %v", err)
	}
	if err := engine.Insert(context.Background(), "far", far, core.SparseVector{}, nil); err != nil {
		t.Fatalf("Insert(far) failed: %v", err)
	}

	results, err := engine.Search(context.Background(), origin, core.SparseVector{}, 2, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != "near" {
		t.Errorf("expected 'near' ranked first under Euclidean distance, got '%s'", results[0].ID)
	}
	if results[0].Score <= results[1].Score {
		t.Errorf("expected closer vector to score higher: near=%f far=%f", results[0].Score, results[1].Score)
	}
}

func TestHnswDistanceFuncFloat32_UnsupportedMetric(t *testing.T) {
	_, err := hnswDistanceFuncFloat32("manhattan")
	if err == nil {
		t.Fatal("expected error for unsupported metric, got nil")
	}
}

func TestHnswDistanceFuncInt8_UnsupportedMetric(t *testing.T) {
	_, err := hnswDistanceFuncInt8("manhattan")
	if err == nil {
		t.Fatal("expected error for unsupported metric, got nil")
	}
}
