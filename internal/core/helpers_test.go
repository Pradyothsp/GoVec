package core

import (
	"path/filepath"
	"testing"
)

// newTestIndex creates a VectorIndex backed by a temp WAL for testing.
// The WAL file lives in t.TempDir() and is closed automatically on test cleanup.
func newTestIndex(t testing.TB) *VectorIndex {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("newTestIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })
	return NewVectorIndex(wal)
}
