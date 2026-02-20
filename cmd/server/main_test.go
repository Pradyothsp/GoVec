package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

func TestWALCloseOnShutdown(t *testing.T) {
	// Create temporary WAL
	walPath := filepath.Join(t.TempDir(), "test.wal")

	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)

	// Write an entry
	err = wal.WriteEntry(core.WALEntry{
		Action: core.WALActionInsert,
		ID:     "test1",
		Vector: []float32{1.0, 2.0},
	})
	require.NoError(t, err)

	// Close WAL (simulating defer wal.Close() in main)
	err = wal.Close()
	require.NoError(t, err)

	// Verify file is properly closed by trying to reopen
	wal2, err := core.NewWAL(walPath)
	require.NoError(t, err, "Should be able to reopen WAL after close")
	defer wal2.Close()

	// Verify data is persisted
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0), "WAL should have data")
}

func TestWALDoubleClose(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)

	// First close
	err = wal.Close()
	require.NoError(t, err)

	// Second close should error
	err = wal.Close()
	assert.Error(t, err, "Double close should return error")
}

func TestRecoverySequence(t *testing.T) {
	tempDir := t.TempDir()
	snapPath := filepath.Join(tempDir, "snapshot.bin")
	walPath := filepath.Join(tempDir, "test.wal")

	// Setup: Create snapshot with 3 vectors
	setupWal, err := core.NewWAL(filepath.Join(tempDir, "setup.wal"))
	require.NoError(t, err)
	setupIdx := core.NewVectorIndex(setupWal)

	for i := 1; i <= 3; i++ {
		err = setupIdx.Insert(fmt.Sprintf("snap%d", i), []float32{float32(i)}, nil)
		require.NoError(t, err)
	}
	err = setupIdx.SaveToFile(snapPath)
	require.NoError(t, err)
	setupWal.Close()

	// Create WAL with 2 more vectors
	wal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	err = wal.WriteEntry(core.WALEntry{
		Action: core.WALActionInsert,
		ID:     "wal1",
		Vector: []float32{4.0},
	})
	require.NoError(t, err)
	err = wal.WriteEntry(core.WALEntry{
		Action: core.WALActionInsert,
		ID:     "wal2",
		Vector: []float32{5.0},
	})
	require.NoError(t, err)
	wal.Close()

	// Recovery: Create new index
	recoveryWal, err := core.NewWAL(walPath)
	require.NoError(t, err)
	defer recoveryWal.Close()

	recoveredIdx := core.NewVectorIndex(recoveryWal)

	// CORRECT sequence: LoadFromFile(DataPath) then ReplayWAL(WalPath)
	err = recoveredIdx.LoadFromFile(snapPath) // Load snapshot
	require.NoError(t, err)

	err = recoveredIdx.ReplayWAL(walPath) // Replay WAL
	require.NoError(t, err)

	// Verify: Should have 5 vectors (3 from snapshot + 2 from WAL)
	assert.Len(t, recoveredIdx.Store, 5, "Should recover all 5 vectors")
	assert.Contains(t, recoveredIdx.Store, "snap1")
	assert.Contains(t, recoveredIdx.Store, "snap2")
	assert.Contains(t, recoveredIdx.Store, "snap3")
	assert.Contains(t, recoveredIdx.Store, "wal1")
	assert.Contains(t, recoveredIdx.Store, "wal2")
}
