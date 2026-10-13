package index

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/metric"
)

// TestLoadFromFile_RejectsDistanceMetricMismatch is the regression test for
// every snapshot recording "cosine" whatever the index's metric was. A
// snapshot's vectors are only meaningful under the metric that wrote them --
// scalar quantization unit-normalizes under cosine, and an HNSW graph's edges
// are chosen by its metric -- so loading one under the other must fail, as a
// quantization mismatch already does.
func TestLoadFromFile_RejectsDistanceMetricMismatch(t *testing.T) {
	engines := []struct {
		name string
		open func(t *testing.T, name metric.Name) Engine
	}{
		{"brute", func(t *testing.T, name metric.Name) Engine { return NewVectorIndex(metricOptions(t, name)) }},
		{"hnsw", func(t *testing.T, name metric.Name) Engine {
			return NewHNSWIndex(metricOptions(t, name), hnsw.Config{})
		}},
	}

	for _, e := range engines {
		t.Run(e.name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "snapshot.bin")

			saved := e.open(t, metric.Euclidean)
			require.NoError(t, saved.Insert(ctx, "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
			require.NoError(t, saved.SaveToFile(ctx, path))

			t.Run("same_metric_loads", func(t *testing.T) {
				// Arrange
				loaded := e.open(t, metric.Euclidean)

				// Act
				err := loaded.LoadFromFile(ctx, path)

				// Assert
				require.NoError(t, err)
				assert.Equal(t, 1, loaded.Len())
			})

			t.Run("other_metric_is_rejected", func(t *testing.T) {
				// Arrange
				loaded := e.open(t, metric.Cosine)

				// Act
				err := loaded.LoadFromFile(ctx, path)

				// Assert
				require.Error(t, err)
				assert.Contains(t, err.Error(), "distance metric mismatch")
				assert.Contains(t, err.Error(), "euclidean", "the error must name the snapshot's metric")
				assert.Equal(t, 0, loaded.Len(), "a rejected snapshot must not be partly loaded")
			})
		})
	}
}

// metricOptions is float32 Options using the metric called name, logging to a
// temporary WAL.
func metricOptions(t *testing.T, name metric.Name) Options[[]float32] {
	t.Helper()
	m, err := metric.Resolve[[]float32](name)
	require.NoError(t, err)

	return Options[[]float32]{WAL: newTestWAL(t), Metric: m}
}
