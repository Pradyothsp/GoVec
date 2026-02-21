package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// A. Basic Replay Tests (4 tests)
// =============================================================================

func TestReplayWAL_EmptyFile(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "empty.wal")

	// Create empty WAL file
	err := os.WriteFile(walPath, []byte(""), 0644)
	require.NoError(t, err)

	// Create index and replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err, "Replaying empty WAL should not error")

	assert.Len(t, idx.Store, 0, "Index should stay empty")
}

func TestReplayWAL_MissingFile(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "nonexistent.wal")

	// Create index and replay non-existent file
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err := idx.ReplayWAL(walPath)
	require.NoError(t, err, "Replaying missing WAL should not error (graceful handling)")

	assert.Len(t, idx.Store, 0, "Index should stay empty")
}

func TestReplayWAL_SingleInsert(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Write single INSERT entry
	entry := WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{1.0, 2.0, 3.0},
		Meta:   map[string]any{"key": "value"},
	}
	writeWALEntry(t, walPath, entry)

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err := idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify
	require.Contains(t, idx.Store, "vec1")
	node := idx.Store["vec1"]
	assert.Equal(t, "vec1", node.ID)
	assert.Equal(t, []float32{1.0, 2.0, 3.0}, node.Vector)
	assert.Equal(t, map[string]any{"key": "value"}, node.Metadata)
}

func TestReplayWAL_SingleDelete(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Write DELETE entry for non-existent ID
	entry := WALEntry{
		Action: WALActionDelete,
		ID:     "vec1",
	}
	writeWALEntry(t, walPath, entry)

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err := idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify index stays empty (safe no-op)
	assert.Len(t, idx.Store, 0, "Deleting non-existent ID should be a no-op")
}

// =============================================================================
// B. Multi-Entry Replay Tests (4 tests)
// =============================================================================

func TestReplayWAL_MultipleInserts(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// Write 10 INSERT entries
	for i := 0; i < 10; i++ {
		entry := WALEntry{
			Action: WALActionInsert,
			ID:     "vec" + string(rune('0'+i)),
			Vector: []float32{float32(i)},
		}
		err = wal.WriteEntry(entry)
		require.NoError(t, err)
	}
	wal.Close()

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify all 10 vectors in index
	assert.Len(t, idx.Store, 10, "All 10 vectors should be in index")
}

func TestReplayWAL_InsertThenDelete(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// INSERT vec1
	err = wal.WriteEntry(WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{1.0, 2.0},
	})
	require.NoError(t, err)

	// DELETE vec1
	err = wal.WriteEntry(WALEntry{
		Action: WALActionDelete,
		ID:     "vec1",
	})
	require.NoError(t, err)
	wal.Close()

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify index empty (DELETE removed it)
	assert.Len(t, idx.Store, 0, "DELETE should remove the inserted vector")
}

func TestReplayWAL_InsertUpdateInsert(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// 3 INSERTs for same ID with different vectors
	err = wal.WriteEntry(WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{1.0},
		Meta:   map[string]any{"version": 1},
	})
	require.NoError(t, err)

	err = wal.WriteEntry(WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{2.0},
		Meta:   map[string]any{"version": 2},
	})
	require.NoError(t, err)

	err = wal.WriteEntry(WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{3.0},
		Meta:   map[string]any{"version": 3},
	})
	require.NoError(t, err)
	wal.Close()

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify final value is last INSERT
	require.Contains(t, idx.Store, "vec1")
	node := idx.Store["vec1"]
	assert.Equal(t, []float32{3.0}, node.Vector, "Final value should be last INSERT")
	// JSON unmarshaling converts numbers to float64
	assert.Equal(t, float64(3), node.Metadata["version"])
}

func TestReplayWAL_ComplexSequence(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// INSERT vec1-10
	for i := 1; i <= 10; i++ {
		err = wal.WriteEntry(WALEntry{
			Action: WALActionInsert,
			ID:     fmt.Sprintf("vec%d", i),
			Vector: []float32{float32(i)},
		})
		require.NoError(t, err)
	}

	// DELETE vec5-7
	for i := 5; i <= 7; i++ {
		err = wal.WriteEntry(WALEntry{
			Action: WALActionDelete,
			ID:     fmt.Sprintf("vec%d", i),
		})
		require.NoError(t, err)
	}

	// INSERT vec11-12
	for i := 11; i <= 12; i++ {
		err = wal.WriteEntry(WALEntry{
			Action: WALActionInsert,
			ID:     fmt.Sprintf("vec%d", i),
			Vector: []float32{float32(i)},
		})
		require.NoError(t, err)
	}

	// DELETE vec1
	err = wal.WriteEntry(WALEntry{
		Action: WALActionDelete,
		ID:     "vec1",
	})
	require.NoError(t, err)
	wal.Close()

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify final state: vec2-4, vec8-10, vec11-12 (8 vectors total)
	expectedIDs := []string{"vec2", "vec3", "vec4", "vec8", "vec9", "vec10", "vec11", "vec12"}
	assert.Len(t, idx.Store, len(expectedIDs), "Index should have correct final state")

	for _, id := range expectedIDs {
		assert.Contains(t, idx.Store, id, "Vector %s should exist", id)
	}
}

// =============================================================================
// C. Error Handling Tests (4 tests)
// =============================================================================

func TestReplayWAL_MalformedJSON(t *testing.T) {
	// This test verifies that malformed WAL entries are logged but skipped gracefully
	// After fix: Should see "WARNING: Skipping malformed WAL entry..." in logs
	// Note: We don't capture log output in tests (uses stdlib log)

	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Write: valid entry, malformed JSON, valid entry
	data := `{"action":"INSERT","id":"vec1","vec":[1.0,2.0]}
{"action":"INSERT","id":"vec2" MALFORMED
{"action":"INSERT","id":"vec3","vec":[3.0,4.0]}
`
	err := os.WriteFile(walPath, []byte(data), 0644)
	require.NoError(t, err)

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify 2 vectors recovered (malformed entry skipped)
	assert.Len(t, idx.Store, 2, "2 vectors should be recovered, malformed entry skipped")
	assert.Contains(t, idx.Store, "vec1")
	assert.Contains(t, idx.Store, "vec3")

	// NOTE: After fix, "WARNING: Skipping malformed WAL entry at line 2"
	// should appear in server logs (not tested, uses stdlib log)
}

func TestReplayWAL_PartialEntry(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Write partial entry (missing vector field)
	data := `{"action":"INSERT","id":"vec1"}
`
	err := os.WriteFile(walPath, []byte(data), 0644)
	require.NoError(t, err)

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify partial entry is processed (documents behavior)
	require.Contains(t, idx.Store, "vec1")
	node := idx.Store["vec1"]
	assert.Nil(t, node.Vector, "Vector should be nil for partial entry")
}

func TestReplayWAL_UnknownAction(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Write: INSERT, UPDATE (unknown), INSERT
	data := `{"action":"INSERT","id":"vec1","vec":[1.0]}
{"action":"UPDATE","id":"vec2","vec":[2.0]}
{"action":"INSERT","id":"vec3","vec":[3.0]}
`
	err := os.WriteFile(walPath, []byte(data), 0644)
	require.NoError(t, err)

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify vec1 and vec3 present, UPDATE skipped
	assert.Len(t, idx.Store, 2, "Unknown action should be skipped")
	assert.Contains(t, idx.Store, "vec1")
	assert.Contains(t, idx.Store, "vec3")
	assert.NotContains(t, idx.Store, "vec2", "UPDATE action should be skipped")
}

func TestReplayWAL_CorruptedFile(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Write binary garbage
	data := []byte{0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8}
	err := os.WriteFile(walPath, data, 0644)
	require.NoError(t, err)

	// Replay
	idx := NewVectorIndex[[]float32](newTestWAL(t), func(v []float32) []float32 { return v }, CosineSimilarity)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err, "Corrupted file should not panic")

	// Verify index is empty (no valid entries)
	assert.Len(t, idx.Store, 0, "No entries should be recovered from corrupted file")
}

// =============================================================================
// Helper Functions
// =============================================================================

// newTestWAL creates a temporary WAL for testing
func newTestWAL(t *testing.T) *WAL {
	t.Helper()
	walPath := filepath.Join(t.TempDir(), "temp.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })
	return wal
}

// writeWALEntry writes a single entry to a WAL file
func writeWALEntry(t *testing.T, path string, entry WALEntry) {
	t.Helper()
	data, err := json.Marshal(entry)
	require.NoError(t, err)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	require.NoError(t, err)
	defer f.Close()
	_, err = f.Write(append(data, '\n'))
	require.NoError(t, err)
}
