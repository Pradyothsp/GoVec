package index

import (
	"context"
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

// Under Euclidean distance the nearer vector must score higher, on both
// engines: distance is turned into a similarity, not used as the score.
func TestNewEngine_EuclideanMetric(t *testing.T) {
	ctx := context.Background()
	for _, indexType := range parityIndexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			// Arrange
			engine := newParityEngine(t,
				config.EngineConfig{IndexType: indexType, Quantization: config.QuantizationNone, DistanceMetric: config.DistanceMetricEuclidean},
				config.StorageConfig{})
			require.NoError(t, engine.Insert(ctx, "far", []float32{10, 0, 0}, core.SparseVector{}, nil))
			require.NoError(t, engine.Insert(ctx, "near", []float32{1, 0, 0}, core.SparseVector{}, nil))

			// Act
			results, err := engine.Search(ctx, []float32{0, 0, 0}, core.SparseVector{}, 2, nil)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, config.DistanceMetricEuclidean, engine.Info().DistanceMetric)
			assert.Equal(t, []string{"near", "far"}, resultIDs(results))
			assert.Greater(t, results[0].Score, results[1].Score)
		})
	}
}
