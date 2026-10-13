package testutil

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/metric"
)

// NewTestIndex creates an empty VectorIndex backed by a temp WAL, which
// t.Cleanup closes, and snapshotting to a temp file.
func NewTestIndex(t testing.TB) *index.VectorIndex[[]float32] {
	t.Helper()
	return NewTestIndexAt(t, filepath.Join(t.TempDir(), "test.bin"))
}

// NewTestIndexAt is NewTestIndex snapshotting to snapshotPath.
func NewTestIndexAt(t testing.TB, snapshotPath string) *index.VectorIndex[[]float32] {
	t.Helper()
	wal, err := index.NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("NewTestIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() }) //nolint:errcheck // test cleanup

	return newCosineIndex(t, index.Options[[]float32]{WAL: wal, SnapshotPath: snapshotPath})
}

// NewIndexOnWAL creates an empty brute-force cosine VectorIndex that logs to
// wal, or logs nothing when wal is nil, and has no snapshot path.
func NewIndexOnWAL(t testing.TB, wal *index.WAL) *index.VectorIndex[[]float32] {
	t.Helper()
	return newCosineIndex(t, index.Options[[]float32]{WAL: wal})
}

// newCosineIndex builds a brute-force VectorIndex from opts with the cosine
// metric.
func newCosineIndex(t testing.TB, opts index.Options[[]float32]) *index.VectorIndex[[]float32] {
	t.Helper()
	m, err := metric.Resolve[[]float32](metric.Cosine)
	require.NoError(t, err)

	opts.Metric = m
	return index.NewVectorIndex(opts)
}
