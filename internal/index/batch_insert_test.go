package index

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

// =============================================================================
// VectorIndex.BatchInsert
// =============================================================================

func TestVectorIndex_BatchInsert_Success(t *testing.T) {
	idx := newTestIndex(t)

	items := []BatchInsertItem{
		{ID: "v1", Vector: fixtures.Vec3dSimple, Meta: fixtures.MetaSimple},
		{ID: "v2", Vector: fixtures.Vec3dAlternate},
		{ID: "v3", Vector: fixtures.Vec3dThird},
	}

	failures, err := idx.BatchInsert(context.Background(), items)
	require.NoError(t, err)
	assert.Empty(t, failures)
	assert.Equal(t, 3, idx.Len())

	for _, item := range items {
		internalID, idErr := idx.IDMapper.ToUint32ID(item.ID)
		require.NoError(t, idErr)
		assert.Contains(t, idx.Store, internalID)
	}
}

func TestVectorIndex_BatchInsert_EmptyBatchIsNoop(t *testing.T) {
	idx := newTestIndex(t)

	failures, err := idx.BatchInsert(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, failures)
	assert.Equal(t, 0, idx.Len())
}

// TestVectorIndex_BatchInsert_RecoversViaReplayWAL exercises the actual reason a
// batched fsync is safe: after a successful BatchInsert, a from-scratch index
// replaying the same WAL file must reconstruct the identical state, exactly as
// it would for entries written one at a time.
func TestVectorIndex_BatchInsert_RecoversViaReplayWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	idx := NewVectorIndex(Options[[]float32]{WAL: wal, Metric: testCosineMetric})

	items := []BatchInsertItem{
		{ID: "v1", Vector: fixtures.Vec3dSimple, Meta: fixtures.MetaSimple},
		{ID: "v2", Vector: fixtures.Vec3dAlternate},
		{ID: "v3", Vector: fixtures.Vec3dThird},
	}
	failures, err := idx.BatchInsert(context.Background(), items)
	require.NoError(t, err)
	require.Empty(t, failures)
	require.NoError(t, wal.Close())

	recovered := newTestIndex(t)
	require.NoError(t, recovered.ReplayWAL(walPath))

	assert.Equal(t, 3, recovered.Len())
	for _, item := range items {
		internalID, idErr := recovered.IDMapper.ToUint32ID(item.ID)
		require.NoError(t, idErr)
		assert.Contains(t, recovered.Store, internalID)
	}
}

// =============================================================================
// HNSWIndex.BatchInsert
// =============================================================================

func TestHNSWIndex_BatchInsert_Success(t *testing.T) {
	idx := newTestHNSWIndex(t)

	items := []BatchInsertItem{
		{ID: "v1", Vector: fixtures.Vec3dSimple, Meta: fixtures.MetaSimple},
		{ID: "v2", Vector: fixtures.Vec3dAlternate},
		{ID: "v3", Vector: fixtures.Vec3dThird},
	}

	failures, err := idx.BatchInsert(context.Background(), items)
	require.NoError(t, err)
	assert.Empty(t, failures)
	assert.Equal(t, 3, idx.Len())
}
