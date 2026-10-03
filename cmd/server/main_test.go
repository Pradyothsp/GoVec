package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

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

func TestStartAutoSave_SavesUntilStopped(t *testing.T) {
	ctx := context.Background()
	cfg, engine := newRecoveryTarget(t)
	cfg.Storage.AutoSaveEnabled = true
	cfg.Storage.AutoSaveInterval = 10 * time.Millisecond
	require.NoError(t, engine.Insert(ctx, "v1", []float32{1, 2}, core.SparseVector{}, nil))

	stop := startAutoSave(cfg, engine)
	require.Eventually(t, func() bool {
		_, err := os.Stat(cfg.Storage.DataPath)
		return err == nil
	}, 2*time.Second, 5*time.Millisecond, "auto-save must write a snapshot")

	stop()
	require.NoError(t, os.Remove(cfg.Storage.DataPath))
	time.Sleep(10 * cfg.Storage.AutoSaveInterval)
	assert.NoFileExists(t, cfg.Storage.DataPath, "no save may happen after stop returns")
}

func TestStartAutoSave_Disabled_StopIsANoOp(t *testing.T) {
	cfg, engine := newRecoveryTarget(t)
	cfg.Storage.AutoSaveEnabled = false
	cfg.Storage.AutoSaveInterval = 10 * time.Millisecond

	stop := startAutoSave(cfg, engine)
	time.Sleep(10 * cfg.Storage.AutoSaveInterval)
	stop()
	assert.NoFileExists(t, cfg.Storage.DataPath)
}

// saveObservingEngine runs a check at the moment the final save starts.
type saveObservingEngine struct {
	index.Engine
	beforeSave func()
}

func (e saveObservingEngine) SaveToFile(ctx context.Context, path string) error {
	e.beforeSave()
	return e.Engine.SaveToFile(ctx, path)
}

// The final save must come after the servers stop taking requests and after
// auto-save stops, so it's the last write and contains every accepted request.
func TestShutdownServer_StopsTrafficAndAutoSaveBeforeFinalSave(t *testing.T) {
	ctx := context.Background()
	cfg, engine := newRecoveryTarget(t)
	require.NoError(t, engine.Insert(ctx, "v1", []float32{1, 2}, core.SparseVector{}, nil))

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	srv := &http.Server{ReadHeaderTimeout: time.Second, Handler: http.NotFoundHandler()}
	go func() { _ = srv.Serve(ln) }()

	// Wait for a real response: until Serve is running, Shutdown has no
	// listener to close, and a TCP dial still succeeds against the kernel's
	// accept queue -- which made this test flaky under load.
	resp, err := http.Get("http://" + ln.Addr().String()) //nolint:noctx // local test server
	require.NoError(t, err)
	_ = resp.Body.Close()

	autoSaveStopped := false
	saved := false
	observed := saveObservingEngine{Engine: engine, beforeSave: func() {
		saved = true
		assert.True(t, autoSaveStopped, "auto-save must be stopped before the final save")
		conn, dialErr := net.DialTimeout("tcp", ln.Addr().String(), 200*time.Millisecond)
		if dialErr == nil {
			_ = conn.Close()
		}
		assert.Error(t, dialErr, "the HTTP server must have stopped accepting before the final save")
	}}

	shutdownServer(cfg, observed, srv, nil, nil, func() { autoSaveStopped = true })

	assert.True(t, saved, "shutdown must still do a final save")
	assert.FileExists(t, cfg.Storage.DataPath)
}
