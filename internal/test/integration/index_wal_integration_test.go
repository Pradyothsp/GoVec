package integration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// =============================================================================
// A. Insert Flow Tests (3 tests)
// =============================================================================

func TestVectorIndex_Insert_WritesToWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert vector
	err = idx.Insert("vec1", []float32{1.0, 2.0, 3.0}, map[string]any{"label": "test"})
	require.NoError(t, err)

	// Read WAL file directly
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	// Parse entry
	var entry core.WALEntry
	err = json.Unmarshal(data[:len(data)-1], &entry) // Strip newline
	require.NoError(t, err)

	// Verify WAL contains correct data
	assert.Equal(t, core.WALActionInsert, entry.Action)
	assert.Equal(t, "vec1", entry.ID)
	assert.Equal(t, []float32{1.0, 2.0, 3.0}, entry.Vector)
	assert.Equal(t, "test", entry.Meta["label"])
}

func TestVectorIndex_Insert_WALFailurePreventMemoryUpdate(t *testing.T) {
	// 🔴 CRITICAL: Write-ahead guarantee test
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	wal.Close() // Close file to make writes fail

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Attempt Insert (should fail)
	err = idx.Insert("vec1", []float32{1.0, 2.0}, map[string]any{"label": "test"})
	assert.Error(t, err, "Insert should fail when WAL write fails")

	// Verify memory NOT updated
	assert.NotContains(t, idx.Store, "vec1", "Memory should NOT be updated on WAL failure")
}

func TestVectorIndex_Insert_ConcurrentWithReplay(t *testing.T) {
	// 🔴 CRITICAL: Race condition test - ReplayWAL has no locking
	walPath := filepath.Join(t.TempDir(), "test.wal")
	replayWalPath := filepath.Join(t.TempDir(), "replay.wal")

	// Create replay WAL with 10 entries
	replayWal, err := core.NewWAL(replayWalPath)
	require.NoError(t, err)
	for i := 0; i < 10; i++ {
		err = replayWal.WriteEntry(core.WALEntry{
			Action: core.WALActionInsert,
			ID:     "replay" + string(rune('0'+i)),
			Vector: []float32{float32(i)},
		})
		require.NoError(t, err)
	}
	replayWal.Close()

	// Create index
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()
	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

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
			err := idx.Insert("insert"+string(rune('0'+i)), []float32{float32(i + 100)}, nil)
			if err != nil && insertErr == nil {
				insertErr = err
			}
		}
	}()

	wg.Wait()

	// Assert no panics or errors
	assert.NoError(t, replayErr, "ReplayWAL should not panic or error")
	assert.NoError(t, insertErr, "Insert should not panic or error")

	// Assert data integrity - all vectors should be present
	assert.GreaterOrEqual(t, len(idx.Store), 10, "At least 10 vectors should be present")
}

// =============================================================================
// B. Delete Flow Tests (3 tests)
// =============================================================================

func TestVectorIndex_Delete_WritesToWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert then delete
	err = idx.Insert("vec1", []float32{1.0, 2.0}, nil)
	require.NoError(t, err)

	deleted, err := idx.Delete("vec1")
	require.NoError(t, err)
	assert.True(t, deleted)

	// Read WAL file
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	// Parse entries (should be INSERT + DELETE)
	lines := splitLines(data)
	assert.Len(t, lines, 2, "WAL should have INSERT + DELETE entries")

	var deleteEntry core.WALEntry
	err = json.Unmarshal(lines[1], &deleteEntry)
	require.NoError(t, err)

	assert.Equal(t, core.WALActionDelete, deleteEntry.Action)
	assert.Equal(t, "vec1", deleteEntry.ID)
}

func TestVectorIndex_Delete_WALFailurePreventsDelete(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert vector
	err = idx.Insert("vec1", []float32{1.0, 2.0}, nil)
	require.NoError(t, err)

	// Close WAL to make writes fail
	wal.Close()

	// Attempt Delete (should fail)
	deleted, err := idx.Delete("vec1")
	assert.Error(t, err, "Delete should fail when WAL write fails")
	assert.False(t, deleted, "Delete should return false on WAL failure")

	// Verify vector still in memory
	assert.Contains(t, idx.Store, "vec1", "Vector should still be in memory on WAL failure")
}

func TestVectorIndex_Delete_NonExistentID(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Delete non-existent ID
	deleted, err := idx.Delete("nonexistent")
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

	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert 10 vectors
	for i := 0; i < 10; i++ {
		err = idx.Insert("vec"+string(rune('0'+i)), []float32{float32(i)}, nil)
		require.NoError(t, err)
	}

	// Verify WAL has entries
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0), "WAL should have entries before save")

	// Save to file
	err = idx.SaveToFile(snapPath)
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

	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert vectors
	for i := 0; i < 5; i++ {
		err = idx.Insert("vec"+string(rune('0'+i)), []float32{float32(i)}, nil)
		require.NoError(t, err)
	}

	// Close WAL to force Clear() failure
	wal.Close()

	// Save to file (should fail atomically)
	err = idx.SaveToFile(snapPath)
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

	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert vec1-3
	for i := 1; i <= 3; i++ {
		err = idx.Insert("vec"+string(rune('0'+i)), []float32{float32(i)}, nil)
		require.NoError(t, err)
	}

	// Save snapshot
	err = idx.SaveToFile(snapPath)
	require.NoError(t, err)

	// Insert vec4-5 (not saved)
	for i := 4; i <= 5; i++ {
		err = idx.Insert("vec"+string(rune('0'+i)), []float32{float32(i)}, nil)
		require.NoError(t, err)
	}
	wal.Close()

	// Simulate crash and recovery
	newWal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer newWal.Close()

	recoveredIdx := core.NewVectorIndex[[]float32](newWal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Load from snapshot
	err = recoveredIdx.LoadFromFile(snapPath)
	require.NoError(t, err)

	// Replay WAL
	err = recoveredIdx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify all 5 vectors restored
	assert.Len(t, recoveredIdx.Store, 5, "All 5 vectors should be recovered")
	for i := 1; i <= 5; i++ {
		assert.Contains(t, recoveredIdx.Store, "vec"+string(rune('0'+i)))
	}
}

func TestVectorIndex_Recovery_WALOnly(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	snapPath := filepath.Join(t.TempDir(), "snapshot.bin")

	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)

	idx := core.NewVectorIndex[[]float32](wal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Insert 10 vectors (no save)
	for i := 0; i < 10; i++ {
		err = idx.Insert("vec"+string(rune('0'+i)), []float32{float32(i)}, nil)
		require.NoError(t, err)
	}
	wal.Close()

	// Simulate crash and recovery
	newWal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer newWal.Close()

	recoveredIdx := core.NewVectorIndex[[]float32](newWal, func(v []float32) []float32 { return v }, core.CosineSimilarity)

	// Load from missing snapshot (graceful)
	err = recoveredIdx.LoadFromFile(snapPath)
	require.NoError(t, err, "Missing snapshot should be handled gracefully")

	// Replay WAL
	err = recoveredIdx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify all 10 vectors recovered from WAL alone
	assert.Len(t, recoveredIdx.Store, 10, "All 10 vectors should be recovered from WAL")
}

// =============================================================================
// Helper Functions
// =============================================================================

func splitLines(data []byte) [][]byte {
	var lines [][]byte
	start := 0
	for i := 0; i < len(data); i++ {
		if data[i] == '\n' {
			if i > start {
				lines = append(lines, data[start:i])
			}
			start = i + 1
		}
	}
	return lines
}
