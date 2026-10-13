package index

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

// A nested GOVEC_WAL_PATH must not fail startup just because its directory
// doesn't exist yet -- the mmap store already creates its own directory.
func TestNewWAL_CreatesParentDirectories(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "nested", "deep", "path", "test.wal")

	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	assert.FileExists(t, walPath)
}

func TestNewWAL_UncreatableParent_ReturnsError(t *testing.T) {
	_, err := NewWAL(filepath.Join(uncreatableDir(t), "test.wal"))
	assert.Error(t, err)
}

// readWALEntries reads a WAL back through replay, so tests check what was
// written rather than how the format spells it.
func readWALEntries(t *testing.T, path string) []WALEntry {
	t.Helper()
	var entries []WALEntry
	require.NoError(t, replayWALFile(path, func(e WALEntry) error {
		entries = append(entries, e)
		return nil
	}))
	return entries
}

// A record holds raw float32s plus a small frame, not text per number.
func TestWriteEntry_LargeVector_IsCompact(t *testing.T) {
	// Arrange
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	// Act
	err = wal.WriteEntry(context.Background(), &WALEntry{Action: WALActionInsert, ID: "large", Vector: fixtures.Vec1536d})

	// Assert
	require.NoError(t, err)
	entries := readWALEntries(t, walPath)
	require.Len(t, entries, 1)
	assert.Equal(t, fixtures.Vec1536d, entries[0].Vector)
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Less(t, info.Size(), int64(1536*4+64))
}

func TestWAL_WriteEntries_WritesAllEntriesInOneSync(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	entries := []*WALEntry{
		{Action: WALActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple},
		{Action: WALActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
		{Action: WALActionDelete, ID: "v1"},
	}

	err = wal.WriteEntries(context.Background(), entries)
	require.NoError(t, err)

	// No extra fsync needed to observe the data -- WriteEntries syncs internally,
	// so it should already be on disk without a Close().
	got := readWALEntries(t, walPath)
	require.Len(t, got, 3, "each entry should be its own record, in order")
	for i, want := range entries {
		assert.Equal(t, *want, got[i])
	}
}

func TestWAL_WriteEntries_EmptySliceIsNoop(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	err = wal.WriteEntries(context.Background(), nil)
	require.NoError(t, err)

	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size())
}

// TestWAL_WriteEntries_MarshalErrorLeavesFileUntouched verifies the batch is
// all-or-nothing at the marshaling stage too: if any entry in the batch can't
// be JSON-encoded, nothing from that batch -- not even the entries before the
// bad one -- should reach disk.
func TestWAL_WriteEntries_MarshalErrorLeavesFileUntouched(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	entries := []*WALEntry{
		{Action: WALActionInsert, ID: "good-entry-before", Vector: fixtures.Vec3dSimple},
		{Action: WALActionInsert, ID: "bad-entry", Meta: map[string]any{"unmarshalable": make(chan int)}},
	}

	err = wal.WriteEntries(context.Background(), entries)
	require.Error(t, err)

	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size(), "a marshal failure anywhere in the batch must not leak earlier entries to disk")
}

// TestWAL_WriteEntries_WriteErrorIsAllOrNothing verifies that if the underlying
// file write fails, no partial batch is left behind -- the whole batch is one
// Write() call, so there's no torn-entry state to worry about, only "wrote
// everything" or "wrote nothing."
func TestWAL_WriteEntries_WriteErrorIsAllOrNothing(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	// Close the underlying file out from under the WAL to force the write to fail.
	require.NoError(t, wal.file.Close())

	entries := []*WALEntry{
		{Action: WALActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple},
		{Action: WALActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
	}

	err = wal.WriteEntries(context.Background(), entries)
	require.Error(t, err)

	data, readErr := os.ReadFile(walPath)
	require.NoError(t, readErr)
	assert.Empty(t, data, "a failed batch write must not leave a partial batch on disk")
}

func TestWAL_Close(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	// Write entry
	entry := WALEntry{Action: WALActionInsert, ID: "test", Vector: fixtures.Vec3dSimple}
	err = wal.WriteEntry(context.Background(), &entry)
	require.NoError(t, err)

	require.NoError(t, wal.Close())
	assert.Error(t, wal.Close(), "a second close must report the file is already closed")

	// Verify data was flushed
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)
	assert.NotEmpty(t, data)
}

func TestWriteEntry_Concurrency(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	var wg sync.WaitGroup
	numGoroutines := 50
	errs := make(chan error, numGoroutines)
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(n int) {
			defer wg.Done()
			entry := WALEntry{
				Action: WALActionInsert,
				ID:     fmt.Sprintf("v%d", n),
				Vector: []float32{float32(n), float32(n * 2)},
			}
			errs <- wal.WriteEntry(context.Background(), &entry)
		}(i)
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	// -race checks memory, not the file: also check every write landed as its
	// own record, whole and exactly once.
	entries := readWALEntries(t, walPath)
	require.Len(t, entries, numGoroutines)

	seen := make(map[string]bool, numGoroutines)
	for _, entry := range entries {
		seen[entry.ID] = true
	}
	for i := 0; i < numGoroutines; i++ {
		assert.True(t, seen[fmt.Sprintf("v%d", i)], "entry v%d is missing", i)
	}
}

// A missing WAL is a first start, and an empty one a clean shutdown: neither
// is an error, and neither adds anything.
func TestReplayWAL_NothingToReplay(t *testing.T) {
	files := map[string]func(t *testing.T) string{
		"missing": func(t *testing.T) string { return filepath.Join(t.TempDir(), "none.wal") },
		"empty":   func(t *testing.T) string { return writeWAL(t, nil) },
	}
	for _, e := range bothEngines {
		for name, path := range files {
			t.Run(e.name+"/"+name, func(t *testing.T) {
				// Arrange
				idx, _ := e.open(t)

				// Act
				err := idx.ReplayWAL(path(t))

				// Assert
				require.NoError(t, err)
				assert.Zero(t, idx.Len())
			})
		}
	}
}

// TestReplayWAL_ComplexSequence tests insert→update→delete chains
func TestReplayWAL_ComplexSequence(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	// Complex sequence: INSERT → UPDATE (overwrite) → DELETE
	entries := []WALEntry{
		{Action: WALActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple, Meta: fixtures.MetaSimple},
		{Action: WALActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
		{Action: WALActionInsert, ID: "v1", Vector: fixtures.Vec3dThird, Meta: fixtures.MetaNested}, // Overwrite
		{Action: WALActionDelete, ID: "v2"},
	}

	for _, e := range entries {
		err := wal.WriteEntry(context.Background(), &e)
		require.NoError(t, err)
	}
	_ = wal.Close() //nolint:errcheck // test cleanup

	// Replay
	idx := newTestIndex(t)
	err = idx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify final state
	assert.Len(t, idx.nodes, 1, "Should have 1 vector (v1 updated, v2 deleted)")

	internalID, err := idx.ids.ToUint32ID("v1")
	require.NoError(t, err)
	assert.Contains(t, idx.nodes, internalID)
	assert.Equal(t, fixtures.Vec3dThird, idx.nodes[internalID].Vector, "Should have updated vector")
	// Note: JSON unmarshal changes types (int→float64, []string→[]interface{})
	// Just verify key fields exist
	assert.Contains(t, idx.nodes[internalID].Metadata, "category")
	assert.Contains(t, idx.nodes[internalID].Metadata, "nested")
	assert.Contains(t, idx.nodes[internalID].Metadata, "tags")

	// v2 should be deleted
	_, err = idx.ids.ToUint32ID("v2")
	assert.Error(t, err, "v2 should be deleted")
}
