package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

func TestWALCloseOnShutdown(t *testing.T) {
	// Create temporary WAL
	walPath := filepath.Join(t.TempDir(), "test.wal")

	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	// Write an entry
	err = wal.WriteEntry(context.Background(), &index.WALEntry{
		Action: index.WALActionInsert,
		ID:     "test1",
		Vector: []float32{1.0, 2.0},
	})
	require.NoError(t, err)

	// Close WAL (simulating defer wal.Close() in main)
	err = wal.Close()
	require.NoError(t, err)

	// Verify file is properly closed by trying to reopen
	wal2, err := index.NewWAL(walPath)
	require.NoError(t, err, "Should be able to reopen WAL after close")
	defer wal2.Close()

	// Verify data is persisted
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0), "WAL should have data")
}

func TestWALDoubleClose(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	// First close
	err = wal.Close()
	require.NoError(t, err)

	// Second close should error
	err = wal.Close()
	assert.Error(t, err, "Double close should return error")
}

// newRecoveryTarget opens the WAL and engine the way main does, for a config
// whose data and WAL paths live in a temp dir.
func newRecoveryTarget(t *testing.T) (*config.Config, index.Engine) {
	t.Helper()
	dir := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.Storage.DataPath = filepath.Join(dir, "snapshot.bin")
	cfg.Storage.WalPath = filepath.Join(dir, "test.wal")
	return cfg, openEngine(t, cfg)
}

func openEngine(t *testing.T, cfg *config.Config) index.Engine {
	t.Helper()
	wal, err := index.NewWAL(cfg.Storage.WalPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })
	engine, err := index.NewEngine(cfg.Engine, cfg.Storage, wal)
	require.NoError(t, err)
	return engine
}

// TestRecoverySequence runs the real runRecovery: snapshot first, then the WAL
// written after it.
func TestRecoverySequence(t *testing.T) {
	ctx := context.Background()
	cfg, before := newRecoveryTarget(t)

	for i := 1; i <= 3; i++ {
		require.NoError(t, before.Insert(ctx, fmt.Sprintf("snap%d", i), []float32{float32(i), 1}, core.SparseVector{}, nil))
	}
	require.NoError(t, before.SaveToFile(ctx, cfg.Storage.DataPath)) // also clears the WAL
	for i := 1; i <= 2; i++ {
		require.NoError(t, before.Insert(ctx, fmt.Sprintf("wal%d", i), []float32{float32(i), 2}, core.SparseVector{}, nil))
	}

	after := openEngine(t, cfg)
	require.NoError(t, runRecovery(cfg, after))

	assert.Equal(t, 5, after.Len(), "3 from the snapshot plus 2 from the WAL")
	for _, id := range []string{"snap1", "snap2", "snap3", "wal1", "wal2"} {
		_, err := after.GetByID(ctx, id)
		assert.NoError(t, err, "%s must be recovered", id)
	}
}

func TestRunRecovery_NoSnapshot_ReplaysWAL(t *testing.T) {
	ctx := context.Background()
	cfg, before := newRecoveryTarget(t)
	require.NoError(t, before.Insert(ctx, "only-in-wal", []float32{1, 2}, core.SparseVector{}, nil))

	after := openEngine(t, cfg)
	require.NoError(t, runRecovery(cfg, after), "a missing snapshot is a fresh start, not an error")
	assert.Equal(t, 1, after.Len())
}

// A corrupt snapshot must stop startup. Carrying on would replay only the
// WAL -- the writes since the last save -- and serve that as the whole index.
func TestRunRecovery_CorruptSnapshot_ReturnsError(t *testing.T) {
	ctx := context.Background()
	cfg, before := newRecoveryTarget(t)
	require.NoError(t, before.Insert(ctx, "only-in-wal", []float32{1, 2}, core.SparseVector{}, nil))
	require.NoError(t, os.WriteFile(cfg.Storage.DataPath, []byte("not a snapshot"), 0o600))

	after := openEngine(t, cfg)
	err := runRecovery(cfg, after)

	require.Error(t, err)
	assert.Contains(t, err.Error(), cfg.Storage.DataPath, "the error must name the file to restore or delete")
	assert.Equal(t, 0, after.Len(), "the WAL must not be replayed over a snapshot that failed to load")
}
