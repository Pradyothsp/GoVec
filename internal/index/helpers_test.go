package index

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Pradyothsp/govec/internal/metric"
)

// uncreatableDir returns a directory path that can never be created: its
// parent is a regular file. Unlike "/nonexistent/...", this fails for every
// user -- including root on a CI runner, who could create /nonexistent.
func uncreatableDir(t testing.TB) string {
	t.Helper()
	file := filepath.Join(t.TempDir(), "a-file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatalf("uncreatableDir: %v", err)
	}
	return filepath.Join(file, "dir")
}

// newTestIndex creates a VectorIndex backed by a temp WAL for testing.
// The WAL file lives in t.TempDir() and is closed automatically on test cleanup.
func newTestIndex(t testing.TB) *VectorIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("newTestIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	return NewVectorIndex(Options[[]float32]{WAL: wal, Metric: testCosineMetric})
}

// testCosineMetric is the cosine metric, for tests that build an index
// directly rather than through NewEngine.
var testCosineMetric = func() metric.Metric[[]float32] {
	m, err := metric.Resolve[[]float32](metric.Cosine)
	if err != nil {
		panic(err)
	}

	return m
}()
