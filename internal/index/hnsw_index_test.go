package index

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

func newTestHNSWIndex(t testing.TB) *HNSWIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "hnsw_test.wal"))
	if err != nil {
		t.Fatalf("newTestHNSWIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	return NewHNSWIndex(Options[[]float32]{WAL: wal, Metric: testCosineMetric, MetadataIndex: true}, hnsw.Config{M: 16, EfSearch: 20, EfConstruction: 200})
}

func newTestHNSWIndexWithHybrid(t testing.TB) *HNSWIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "hnsw_hybrid_test.wal"))
	if err != nil {
		t.Fatalf("newTestHNSWIndexWithHybrid: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	return NewHNSWIndex(Options[[]float32]{WAL: wal, Metric: testCosineMetric, HybridSearch: true, MetadataIndex: true}, hnsw.Config{M: 16, EfSearch: 20, EfConstruction: 200})
}

// What HNSW does differently from brute force is tested here; the rules both
// engines share are tested over both in the rest of the package.

// TestHNSWIndex_LoadFromFile_KeepsConfiguredEfSearch is the regression test for
// a snapshot's saved EfSearch replacing the configured one on load, which kept
// existing data on the old ef after hnsw_ef_search or its default changed.
func TestHNSWIndex_LoadFromFile_KeepsConfiguredEfSearch(t *testing.T) {
	// Arrange
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "hnsw.bin")
	saved := newTestHNSWIndex(t) // EfSearch 20
	require.NoError(t, saved.Insert(ctx, "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
	require.NoError(t, saved.SaveToFile(ctx, path))

	wal, err := NewWAL(filepath.Join(t.TempDir(), "loaded.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })
	loaded := NewHNSWIndex(Options[[]float32]{WAL: wal, Metric: testCosineMetric, MetadataIndex: true}, hnsw.Config{M: 16, EfSearch: 100, EfConstruction: 200})

	// Act
	err = loaded.LoadFromFile(ctx, path)

	// Assert
	require.NoError(t, err)
	assert.Equal(t, 1, loaded.Len())
	assert.Equal(t, 100, loaded.graph.EfSearch, "the configured ef_search must win over the snapshot's")
}

// TestHNSWIndex_FilteredSearch_SelectiveFilter verifies that a highly selective filter
// still returns the correct k results using the graph-allowlist path.
func TestHNSWIndex_FilteredSearch_SelectiveFilter(t *testing.T) {
	idx := newTestHNSWIndex(t)

	vecs := fixtures.GenerateVectors(50, 3)
	// Insert 50 vectors: only 2 are "rare"
	for i, vec := range vecs {
		tier := "common"
		if i == 10 || i == 20 {
			tier = "rare"
		}
		require.NoError(t, idx.Insert(context.Background(),
			fmt.Sprintf("v%d", i), vec, core.SparseVector{}, map[string]any{"tier": tier},
		))
	}

	results, err := idx.Search(context.Background(), vecs[10], core.SparseVector{}, 2, map[string]any{"tier": "rare"})
	require.NoError(t, err)
	require.Len(t, results, 2)
	for _, r := range results {
		assert.Equal(t, "rare", r.Meta["tier"], "all results must have tier=rare")
	}
}
