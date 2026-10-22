package wal

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
func TestOpen_CreatesParentDirectories(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "nested", "deep", "path", "test.wal")

	wal, err := Open(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	assert.FileExists(t, walPath)
}

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

func TestOpen_UncreatableParent_ReturnsError(t *testing.T) {
	_, err := Open(filepath.Join(uncreatableDir(t), "test.wal"))
	assert.Error(t, err)
}

// readEntries reads a WAL back through replay, so tests check what was
// written rather than how the format spells it.
func readEntries(t *testing.T, path string) []Entry {
	t.Helper()
	var entries []Entry
	require.NoError(t, Replay(path, func(e Entry) error {
		entries = append(entries, e)
		return nil
	}))
	return entries
}

// A record holds raw float32s plus a small frame, not text per number.
func TestWriteEntry_LargeVector_IsCompact(t *testing.T) {
	// Arrange
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := Open(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	// Act
	err = wal.WriteEntry(context.Background(), &Entry{Action: ActionInsert, ID: "large", Vector: fixtures.Vec1536d})

	// Assert
	require.NoError(t, err)
	entries := readEntries(t, walPath)
	require.Len(t, entries, 1)
	assert.Equal(t, fixtures.Vec1536d, entries[0].Vector)
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Less(t, info.Size(), int64(1536*4+64))
}

func TestWAL_WriteEntries_WritesAllEntriesInOneSync(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := Open(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	entries := []*Entry{
		{Action: ActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple},
		{Action: ActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
		{Action: ActionDelete, ID: "v1"},
	}

	err = wal.WriteEntries(context.Background(), entries)
	require.NoError(t, err)

	// No extra fsync needed to observe the data -- WriteEntries syncs internally,
	// so it should already be on disk without a Close().
	got := readEntries(t, walPath)
	require.Len(t, got, 3, "each entry should be its own record, in order")
	for i, want := range entries {
		assert.Equal(t, *want, got[i])
	}
}

func TestWAL_WriteEntries_EmptySliceIsNoop(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := Open(walPath)
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
	wal, err := Open(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	entries := []*Entry{
		{Action: ActionInsert, ID: "good-entry-before", Vector: fixtures.Vec3dSimple},
		{Action: ActionInsert, ID: "bad-entry", Meta: map[string]any{"unmarshalable": make(chan int)}},
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
	wal, err := Open(walPath)
	require.NoError(t, err)

	// Close the underlying file out from under the WAL to force the write to fail.
	require.NoError(t, wal.file.Close())

	entries := []*Entry{
		{Action: ActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple},
		{Action: ActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
	}

	err = wal.WriteEntries(context.Background(), entries)
	require.Error(t, err)

	data, readErr := os.ReadFile(walPath)
	require.NoError(t, readErr)
	assert.Empty(t, data, "a failed batch write must not leave a partial batch on disk")
}

func TestWAL_Close(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := Open(walPath)
	require.NoError(t, err)

	// Write entry
	entry := Entry{Action: ActionInsert, ID: "test", Vector: fixtures.Vec3dSimple}
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
	wal, err := Open(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	var wg sync.WaitGroup
	numGoroutines := 50
	errs := make(chan error, numGoroutines)
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(n int) {
			defer wg.Done()
			entry := Entry{
				Action: ActionInsert,
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
	entries := readEntries(t, walPath)
	require.Len(t, entries, numGoroutines)

	seen := make(map[string]bool, numGoroutines)
	for _, entry := range entries {
		seen[entry.ID] = true
	}
	for i := 0; i < numGoroutines; i++ {
		assert.True(t, seen[fmt.Sprintf("v%d", i)], "entry v%d is missing", i)
	}
}
