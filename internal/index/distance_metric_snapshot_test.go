package index

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/metric"
)

// TestLoadFromFile_RejectsDistanceMetricMismatch is the regression test for
// every snapshot recording "cosine" whatever the index's metric was. A
// snapshot's vectors are only meaningful under the metric that wrote them --
// scalar quantization unit-normalizes under cosine, and an HNSW graph's edges
// are chosen by its metric -- so loading one under the other must fail, as a
// quantization mismatch already does.
func TestLoadFromFile_RejectsDistanceMetricMismatch(t *testing.T) {
	type metricIndex interface {
		Engine
		setMetric(metric.Name)
	}

	engines := []struct {
		name string
		open func(t *testing.T) metricIndex
	}{
		{"brute", func(t *testing.T) metricIndex { return bruteMetric{newTestIndex(t)} }},
		{"hnsw", func(t *testing.T) metricIndex { return hnswMetric{newTestHNSWIndex(t)} }},
	}

	for _, e := range engines {
		t.Run(e.name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "snapshot.bin")

			saved := e.open(t)
			saved.setMetric(metric.Euclidean)
			require.NoError(t, saved.Insert(ctx, "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
			require.NoError(t, saved.SaveToFile(ctx, path))

			t.Run("same_metric_loads", func(t *testing.T) {
				loaded := e.open(t)
				loaded.setMetric(metric.Euclidean)
				require.NoError(t, loaded.LoadFromFile(ctx, path))
				assert.Equal(t, 1, loaded.Len())
			})

			t.Run("other_metric_is_rejected", func(t *testing.T) {
				loaded := e.open(t)
				loaded.setMetric(metric.Cosine)
				err := loaded.LoadFromFile(ctx, path)
				require.Error(t, err)
				assert.Contains(t, err.Error(), "distance metric mismatch")
				assert.Contains(t, err.Error(), "euclidean", "the error must name the snapshot's metric")
				assert.Equal(t, 0, loaded.Len(), "a rejected snapshot must not be partly loaded")
			})
		})
	}
}

type bruteMetric struct{ *VectorIndex[[]float32] }

func (b bruteMetric) setMetric(m metric.Name) { b.distanceMetric = m }

type hnswMetric struct{ *HNSWIndex[[]float32] }

func (h hnswMetric) setMetric(m metric.Name) { h.distanceMetric = m }
