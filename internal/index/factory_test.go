package index

import (
	"context"
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
				assert.Equal(t, indexType, info.IndexType)
				assert.Equal(t, quant, info.Quantization)
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

	if engine.Info().DistanceMetric != config.DistanceMetricEuclidean {
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
