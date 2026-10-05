package integration

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

// snapshotFormat is one snapshot writer: an engine, with vectors inline or in
// the mmap store.
type snapshotFormat struct {
	name      string
	indexType config.IndexType
	mmap      bool
}

var snapshotFormats = []snapshotFormat{
	{"brute", config.IndexTypeBrute, false},
	{"hnsw", config.IndexTypeHNSW, false},
	{"brute_mmap", config.IndexTypeBrute, true},
	{"hnsw_mmap", config.IndexTypeHNSW, true},
}

// startSnapshotEngine opens the WAL at walPath and builds an engine over it.
// Calling it again with the same paths is a server restart.
func startSnapshotEngine(t *testing.T, f snapshotFormat, dir, walPath string) (index.Engine, *index.WAL) {
	t.Helper()
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	engineCfg := config.EngineConfig{
		Quantization:   config.QuantizationNone,
		DistanceMetric: config.DistanceMetricCosine,
		IndexType:      f.indexType,
		Dimensions:     3,
	}
	storageCfg := config.StorageConfig{
		EnableMmap:    f.mmap,
		MmapStorePath: filepath.Join(dir, "mmap"),
	}
	engine, err := index.NewEngine(engineCfg, storageCfg, wal)
	require.NoError(t, err)
	return engine, wal
}

// TestSaveToFile_FailedSave_KeepsUnsavedWritesRecoverable checks, for every
// snapshot format, that a save which fails leaves the WAL able to recover every
// write made since the previous snapshot.
//
// The WAL is the only durable copy of those writes until a new snapshot is on
// disk. A save that truncates it first and then fails -- out of memory, disk
// full, an encode error -- leaves a server that still answers from memory but
// loses all of them at the next restart. govec-bench hit exactly this: the
// container was OOM-killed mid-snapshot, right after "WAL cleared".
//
// The save is made to fail by putting a directory where its temporary file
// goes, so os.Create fails. Unlike chmod, that also fails when tests run as root.
func TestSaveToFile_FailedSave_KeepsUnsavedWritesRecoverable(t *testing.T) {
	const savedCount, unsavedCount = 3, 2

	for _, f := range snapshotFormats {
		t.Run(f.name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			snapPath := filepath.Join(dir, "govec_data.bin")
			walPath := filepath.Join(dir, "govec.wal")

			engine, wal := startSnapshotEngine(t, f, dir, walPath)

			for i := range savedCount {
				require.NoError(t, engine.Insert(ctx, fmt.Sprintf("saved-%d", i), []float32{1, float32(i), 0}, core.SparseVector{}, nil))
			}
			require.NoError(t, engine.SaveToFile(ctx, snapPath))

			// Acknowledged writes that only the WAL holds durably from here on.
			for i := range unsavedCount {
				require.NoError(t, engine.Insert(ctx, fmt.Sprintf("unsaved-%d", i), []float32{0, 1, float32(i)}, core.SparseVector{}, nil))
			}

			require.NoError(t, os.Mkdir(snapPath+".tmp", 0o750))
			require.Error(t, engine.SaveToFile(ctx, snapPath), "the save must fail for this test to mean anything")

			// Restart: the same recovery sequence as cmd/server's runRecovery.
			require.NoError(t, wal.Close())
			restarted, _ := startSnapshotEngine(t, f, dir, walPath)
			require.NoError(t, restarted.LoadFromFile(ctx, snapPath))
			require.NoError(t, restarted.ReplayWAL(walPath))

			assert.Equal(t, savedCount+unsavedCount, restarted.Len(), "writes made since the last good snapshot were lost")
			for i := range unsavedCount {
				_, err := restarted.GetByID(ctx, fmt.Sprintf("unsaved-%d", i))
				assert.NoError(t, err, "unsaved-%d must be recovered from the WAL", i)
			}
		})
	}
}

// TestRecovery_WALAlreadyInSnapshot_ReplaysToTheSameState covers a crash after
// a save's rename but before it truncates the WAL. Saving keeps the WAL until
// the new snapshot is safely on disk, so this window exists by design, and
// recovery then replays entries the snapshot already holds. That must land on
// the same state: replay applies inserts as upserts and skips deletes of
// missing IDs, and it applies them in order, so the update and the delete here
// win over the earlier inserts of the same IDs.
func TestRecovery_WALAlreadyInSnapshot_ReplaysToTheSameState(t *testing.T) {
	for _, f := range snapshotFormats {
		t.Run(f.name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			snapPath := filepath.Join(dir, "govec_data.bin")
			walPath := filepath.Join(dir, "govec.wal")

			engine, wal := startSnapshotEngine(t, f, dir, walPath)

			require.NoError(t, engine.Insert(ctx, "kept", []float32{1, 0, 0}, core.SparseVector{}, nil))
			require.NoError(t, engine.Insert(ctx, "updated", []float32{0, 1, 0}, core.SparseVector{}, nil))
			require.NoError(t, engine.Insert(ctx, "deleted", []float32{0, 0, 1}, core.SparseVector{}, nil))
			require.NoError(t, engine.Insert(ctx, "updated", []float32{0, 0, 1}, core.SparseVector{}, nil))
			deleted, err := engine.Delete(ctx, "deleted")
			require.NoError(t, err)
			require.True(t, deleted)

			// The WAL as it stood before the save truncated it: a crash after the
			// rename leaves exactly this next to the new snapshot.
			walBeforeSave, err := os.ReadFile(walPath) //nolint:gosec // test temp dir
			require.NoError(t, err)
			require.NoError(t, engine.SaveToFile(ctx, snapPath))
			require.NoError(t, wal.Close())
			require.NoError(t, os.WriteFile(walPath, walBeforeSave, 0o600))

			restarted, _ := startSnapshotEngine(t, f, dir, walPath)
			require.NoError(t, restarted.LoadFromFile(ctx, snapPath))
			require.NoError(t, restarted.ReplayWAL(walPath))

			assert.Equal(t, 2, restarted.Len())

			kept, err := restarted.GetByID(ctx, "kept")
			require.NoError(t, err)
			assert.InDeltaSlice(t, []float32{1, 0, 0}, kept.Vector, 1e-6)

			updated, err := restarted.GetByID(ctx, "updated")
			require.NoError(t, err)
			assert.InDeltaSlice(t, []float32{0, 0, 1}, updated.Vector, 1e-6, "the later update must win")

			_, err = restarted.GetByID(ctx, "deleted")
			assert.Error(t, err, "the delete must still hold after replaying the insert before it")
		})
	}
}
