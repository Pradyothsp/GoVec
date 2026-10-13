package index

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
)

// openOnDisk opens an engine of indexType on the snapshot and WAL in dir, the
// way the server does: a second call on the same dir is a restart.
func openOnDisk(t *testing.T, dir string, indexType config.IndexType) Engine {
	t.Helper()
	wal, err := NewWAL(filepath.Join(dir, "govec.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	engine, err := NewEngine(
		config.EngineConfig{IndexType: indexType, Quantization: config.QuantizationNone, DistanceMetric: config.DistanceMetricCosine},
		config.StorageConfig{DataPath: filepath.Join(dir, "govec.bin")},
		wal,
	)
	require.NoError(t, err)
	return engine
}

func TestFlush_NoSnapshotPath_Fails(t *testing.T) {
	// Arrange
	idx := newTestIndex(t)

	// Act
	err := idx.Flush(context.Background())

	// Assert
	require.ErrorIs(t, err, errNoSnapshotPath)
}

// Recover restores exactly what a restart must: the snapshot, then every write
// made after it, which only the WAL holds.
func TestRecover_SnapshotThenWAL(t *testing.T) {
	ctx := context.Background()
	for _, indexType := range parityIndexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			// Arrange: 3 vectors in a snapshot, 2 more only in the WAL.
			dir := t.TempDir()
			before := openOnDisk(t, dir, indexType)
			for i := 1; i <= 3; i++ {
				require.NoError(t, before.Insert(ctx, fmt.Sprintf("snap%d", i), []float32{float32(i), 1, 0}, core.SparseVector{}, nil))
			}
			require.NoError(t, before.Flush(ctx))
			for i := 1; i <= 2; i++ {
				require.NoError(t, before.Insert(ctx, fmt.Sprintf("wal%d", i), []float32{float32(i), 0, 1}, core.SparseVector{}, nil))
			}
			after := openOnDisk(t, dir, indexType)

			// Act
			err := after.Recover(ctx)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, 5, after.Len())
			for _, id := range []string{"snap1", "snap2", "snap3", "wal1", "wal2"} {
				_, err := after.GetByID(ctx, id)
				assert.NoError(t, err, "%s must be recovered", id)
			}
		})
	}
}

func TestRecover_NothingOnDisk_StartsEmpty(t *testing.T) {
	// Arrange
	engine := openOnDisk(t, t.TempDir(), config.IndexTypeBrute)

	// Act
	err := engine.Recover(context.Background())

	// Assert
	require.NoError(t, err)
	assert.Zero(t, engine.Len())
}
