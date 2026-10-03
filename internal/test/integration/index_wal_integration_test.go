package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/index"

	"github.com/Pradyothsp/govec/internal/core"
)

// =============================================================================
// A. Insert Flow Tests (3 tests)
// =============================================================================

func TestVectorIndex_Insert_WritesToWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert vector
	err = idx.Insert(context.Background(), "vec1", []float32{1.0, 2.0, 3.0}, core.SparseVector{}, map[string]any{"label": "test"})
	require.NoError(t, err)

	// Read WAL file directly
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	// Parse entry
	var entry index.WALEntry
	err = json.Unmarshal(data[:len(data)-1], &entry) // Strip newline
	require.NoError(t, err)

	// Verify WAL contains correct data
	assert.Equal(t, index.WALActionInsert, entry.Action)
	assert.Equal(t, "vec1", entry.ID)
	assert.Equal(t, []float32{1.0, 2.0, 3.0}, entry.Vector)
	assert.Equal(t, "test", entry.Meta["label"])
}

func TestVectorIndex_Insert_WALFailurePreventMemoryUpdate(t *testing.T) {
	// 🔴 CRITICAL: Write-ahead guarantee test
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	wal.Close() // Close file to make writes fail

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Attempt Insert (should fail)
	err = idx.Insert(context.Background(), "vec1", []float32{1.0, 2.0}, core.SparseVector{}, map[string]any{"label": "test"})
	assert.Error(t, err, "Insert should fail when WAL write fails")

	// Verify memory NOT updated (vec1 should not be in IDMapper)
	_, err = idx.IDMapper.ToUint32ID("vec1")
	assert.Error(t, err, "Memory should NOT be updated on WAL failure")
}

func TestVectorIndex_Insert_ConcurrentWithReplay(t *testing.T) {
	// 🔴 CRITICAL: Race condition test - ReplayWAL has no locking
	walPath := filepath.Join(t.TempDir(), "test.wal")
	replayWalPath := filepath.Join(t.TempDir(), "replay.wal")

	// Create replay WAL with 10 entries
	replayWal, err := index.NewWAL(replayWalPath)
	require.NoError(t, err)
	for i := 0; i < 10; i++ {
		err = replayWal.WriteEntry(context.Background(), &index.WALEntry{
			Action: index.WALActionInsert,
			ID:     "replay" + string(rune('0'+i)),
			Vector: []float32{float32(i)},
		})
		require.NoError(t, err)
	}
	replayWal.Close()

	// Create index
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()
	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	var wg sync.WaitGroup
	wg.Add(2)

	// Track errors/panics
	var replayErr, insertErr error

	// Goroutine 1: Replay WAL
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				replayErr = assert.AnError
			}
		}()
		if err := idx.ReplayWAL(replayWalPath); err != nil {
			replayErr = err
		}
	}()

	// Goroutine 2: Insert vectors
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				insertErr = assert.AnError
			}
		}()
		for i := 0; i < 10; i++ {
			err := idx.Insert(context.Background(), "insert"+string(rune('0'+i)), []float32{float32(i + 100)}, core.SparseVector{}, nil)
			if err != nil && insertErr == nil {
				insertErr = err
			}
		}
	}()

	wg.Wait()

	// Assert no panics or errors
	assert.NoError(t, replayErr, "ReplayWAL should not panic or error")
	assert.NoError(t, insertErr, "Insert should not panic or error")

	// The replayed and inserted IDs are disjoint, so every one must be present.
	assert.Equal(t, 20, idx.Len(), "all 10 replayed and 10 inserted vectors must be present")
}

// =============================================================================
// B. Delete Flow Tests (3 tests)
// =============================================================================

func TestVectorIndex_Delete_WritesToWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert then delete
	err = idx.Insert(context.Background(), "vec1", []float32{1.0, 2.0}, core.SparseVector{}, nil)
	require.NoError(t, err)

	deleted, err := idx.Delete(context.Background(), "vec1")
	require.NoError(t, err)
	assert.True(t, deleted)

	// Read WAL file
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	// Parse entries (should be INSERT + DELETE)
	lines := bytes.Split(bytes.TrimSpace(data), []byte("\n"))
	assert.Len(t, lines, 2, "WAL should have INSERT + DELETE entries")

	var deleteEntry index.WALEntry
	err = json.Unmarshal(lines[1], &deleteEntry)
	require.NoError(t, err)

	assert.Equal(t, index.WALActionDelete, deleteEntry.Action)
	assert.Equal(t, "vec1", deleteEntry.ID)
}

func TestVectorIndex_Delete_WALFailurePreventsDelete(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert vector
	err = idx.Insert(context.Background(), "vec1", []float32{1.0, 2.0}, core.SparseVector{}, nil)
	require.NoError(t, err)

	// Close WAL to make writes fail
	wal.Close()

	// Attempt Delete (should fail)
	deleted, err := idx.Delete(context.Background(), "vec1")
	assert.Error(t, err, "Delete should fail when WAL write fails")
	assert.False(t, deleted, "Delete should return false on WAL failure")

	// Verify vector still in memory
	vec1ID, err := idx.IDMapper.ToUint32ID("vec1")
	require.NoError(t, err, "vec1 should still exist")
	assert.Contains(t, idx.Store, vec1ID, "Vector should still be in memory on WAL failure")
}

func TestVectorIndex_Delete_NonExistentID(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Delete non-existent ID
	deleted, err := idx.Delete(context.Background(), "nonexistent")
	require.NoError(t, err)
	assert.False(t, deleted, "Should return false for non-existent ID")

	// Verify no WAL entry written (optimization)
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size(), "No WAL entry should be written for non-existent ID")
}

// =============================================================================
// C. Snapshot + Clear Workflow Tests (4 tests)
// =============================================================================

func TestVectorIndex_SaveToFile_ClearsWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	snapPath := filepath.Join(t.TempDir(), "snapshot.bin")

	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert 10 vectors
	for i := 0; i < 10; i++ {
		err = idx.Insert(context.Background(), "vec"+string(rune('0'+i)), []float32{float32(i)}, core.SparseVector{}, nil)
		require.NoError(t, err)
	}

	// Verify WAL has entries
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0), "WAL should have entries before save")

	// Save to file
	err = idx.SaveToFile(context.Background(), snapPath)
	require.NoError(t, err)

	// Verify snapshot created
	_, err = os.Stat(snapPath)
	assert.NoError(t, err, "Snapshot file should exist")

	// Verify WAL cleared
	info, err = os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size(), "WAL should be cleared after save")
}

func TestVectorIndex_SaveToFile_WALClearFailure(t *testing.T) {
	// 🔴 CRITICAL: This test verifies two-phase commit prevents partial success
	walPath := filepath.Join(t.TempDir(), "test.wal")
	snapPath := filepath.Join(t.TempDir(), "snapshot.bin")

	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert vectors
	for i := 0; i < 5; i++ {
		err = idx.Insert(context.Background(), "vec"+string(rune('0'+i)), []float32{float32(i)}, core.SparseVector{}, nil)
		require.NoError(t, err)
	}

	// Close WAL to force Clear() failure
	wal.Close()

	// Save to file (should fail atomically)
	err = idx.SaveToFile(context.Background(), snapPath)
	assert.Error(t, err, "SaveToFile should return error when WAL.Clear fails")

	// CRITICAL: After two-phase commit fix:
	// - If WAL.Clear fails, snapshot should NOT exist (atomic failure)
	// - Verify snapshot doesn't exist
	_, statErr := os.Stat(snapPath)
	assert.True(t, os.IsNotExist(statErr),
		"Snapshot should NOT exist if WAL.Clear fails (atomic failure)")

	// WAL should still contain entries (not cleared)
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0), "WAL should still have entries")
}

func TestVectorIndex_Recovery_SnapshotPlusWAL(t *testing.T) {
	// 🔴 CRITICAL: Full recovery test
	walPath := filepath.Join(t.TempDir(), "test.wal")
	snapPath := filepath.Join(t.TempDir(), "snapshot.bin")

	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert vec1-3
	for i := 1; i <= 3; i++ {
		err = idx.Insert(context.Background(), "vec"+string(rune('0'+i)), []float32{float32(i)}, core.SparseVector{}, nil)
		require.NoError(t, err)
	}

	// Save snapshot
	err = idx.SaveToFile(context.Background(), snapPath)
	require.NoError(t, err)

	// Insert vec4-5 (not saved)
	for i := 4; i <= 5; i++ {
		err = idx.Insert(context.Background(), "vec"+string(rune('0'+i)), []float32{float32(i)}, core.SparseVector{}, nil)
		require.NoError(t, err)
	}
	wal.Close()

	// Simulate crash and recovery
	newWal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer newWal.Close()

	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Load from snapshot
	err = recoveredIdx.LoadFromFile(context.Background(), snapPath)
	require.NoError(t, err)

	// Replay WAL
	err = recoveredIdx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify all 5 vectors restored
	assert.Len(t, recoveredIdx.Store, 5, "All 5 vectors should be recovered")
	for i := 1; i <= 5; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		internalID, err := recoveredIdx.IDMapper.ToUint32ID(vecID)
		require.NoError(t, err, "Vector %s should exist", vecID)
		assert.Contains(t, recoveredIdx.Store, internalID)
	}
}

func TestVectorIndex_Recovery_WALOnly(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	snapPath := filepath.Join(t.TempDir(), "snapshot.bin")

	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	idx := index.NewVectorIndex[[]float32](wal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Insert 10 vectors (no save)
	for i := 0; i < 10; i++ {
		err = idx.Insert(context.Background(), "vec"+string(rune('0'+i)), []float32{float32(i)}, core.SparseVector{}, nil)
		require.NoError(t, err)
	}
	wal.Close()

	// Simulate crash and recovery
	newWal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	defer newWal.Close()

	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, core.NewIDMapper(), func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Load from missing snapshot (graceful)
	err = recoveredIdx.LoadFromFile(context.Background(), snapPath)
	require.NoError(t, err, "Missing snapshot should be handled gracefully")

	// Replay WAL
	err = recoveredIdx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify all 10 vectors recovered from WAL alone
	assert.Len(t, recoveredIdx.Store, 10, "All 10 vectors should be recovered from WAL")
}
