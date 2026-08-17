package index

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
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

// TestVectorIndex_BatchInsert_WALFailureAbortsWholeBatch is the test that actually
// matters for this change: the whole point of batching the fsync is that a batch
// is durable as one unit. If the WAL write fails, NONE of the batch should be
// applied to the in-memory index -- otherwise a crash right after a failed WAL
// write would leave the index holding vectors that were never made durable,
// silently breaking the "if it's not in the WAL, it's not committed" guarantee
// that ReplayWAL-based recovery depends on.
func TestVectorIndex_BatchInsert_WALFailureAbortsWholeBatch(t *testing.T) {
	idx := newTestIndex(t)

	// Force the WAL write to fail by closing its underlying file out from under it.
	require.NoError(t, idx.wal.Close())

	items := []BatchInsertItem{
		{ID: "v1", Vector: fixtures.Vec3dSimple},
		{ID: "v2", Vector: fixtures.Vec3dAlternate},
	}

	failures, err := idx.BatchInsert(context.Background(), items)
	require.Error(t, err, "a WAL write failure must surface as an error, not be silently swallowed")
	assert.Nil(t, failures, "no per-item failures should be reported -- the batch never got far enough to attempt applying any item")
	assert.Equal(t, 0, idx.Len(), "nothing should be applied to the index when the WAL write for the batch failed")

	_, idErr := idx.IDMapper.ToUint32ID("v1")
	assert.Error(t, idErr, "v1 must not exist -- BatchInsert must not partially apply a batch whose WAL write failed")
}

// TestVectorIndex_BatchInsert_RecoversViaReplayWAL exercises the actual reason a
// batched fsync is safe: after a successful BatchInsert, a from-scratch index
// replaying the same WAL file must reconstruct the identical state, exactly as
// it would for entries written one at a time.
func TestVectorIndex_BatchInsert_RecoversViaReplayWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	idx := NewVectorIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity, nil)

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

func TestHNSWIndex_BatchInsert_WALFailureAbortsWholeBatch(t *testing.T) {
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.wal.Close())

	items := []BatchInsertItem{
		{ID: "v1", Vector: fixtures.Vec3dSimple},
		{ID: "v2", Vector: fixtures.Vec3dAlternate},
	}

	failures, err := idx.BatchInsert(context.Background(), items)
	require.Error(t, err)
	assert.Nil(t, failures)
	assert.Equal(t, 0, idx.Len(), "nothing should be applied to the HNSW graph when the WAL write for the batch failed")
}
