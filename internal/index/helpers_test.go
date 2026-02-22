package index

import (
	"path/filepath"
	"testing"

	"github.com/Pradyothsp/govec/internal/core"
)

// newTestIndex creates a VectorIndex backed by a temp WAL for testing.
// The WAL file lives in t.TempDir() and is closed automatically on test cleanup.
func newTestIndex(t testing.TB) *VectorIndex[[]float32] {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("newTestIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() })

	// Create an index with IDMapper, identity encode function and cosine similarity
	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	return NewVectorIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity)
}
