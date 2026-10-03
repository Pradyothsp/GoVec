package index

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

// =============================================================================
// Consolidated WAL Tests
// =============================================================================

func TestNewWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	assert.NotNil(t, wal)
	assert.FileExists(t, walPath)
}

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

// TestWriteEntry_Variations consolidates all WriteEntry tests
func TestWriteEntry_Variations(t *testing.T) {
	tests := []struct {
		name   string
		entry  WALEntry
		verify func(*testing.T, string)
	}{
		{
			name: "insert_action",
			entry: WALEntry{
				Action: WALActionInsert,
				ID:     "vec1",
				Vector: fixtures.Vec3dSimple,
				Meta:   fixtures.MetaSimple,
			},
			verify: func(t *testing.T, path string) {
				data, _ := os.ReadFile(path) //nolint:errcheck // test verification
				assert.Contains(t, string(data), `"action":"INSERT"`)
				assert.Contains(t, string(data), `"id":"vec1"`)
			},
		},
		{
			name: "delete_action",
			entry: WALEntry{
				Action: WALActionDelete,
				ID:     "vec2",
			},
			verify: func(t *testing.T, path string) {
				data, _ := os.ReadFile(path) //nolint:errcheck // test verification
				assert.Contains(t, string(data), `"action":"DELETE"`)
				assert.Contains(t, string(data), `"id":"vec2"`)
			},
		},
		{
			name: "large_vector_1536d",
			entry: WALEntry{
				Action: WALActionInsert,
				ID:     "large",
				Vector: fixtures.Vec1536d,
			},
			verify: func(t *testing.T, path string) {
				info, _ := os.Stat(path) //nolint:errcheck // test verification
				assert.Greater(t, info.Size(), int64(1000), "Large vector should produce substantial entry")
			},
		},
		{
			name: "special_characters_in_metadata",
			entry: WALEntry{
				Action: WALActionInsert,
				ID:     "special",
				Vector: fixtures.Vec3dSimple,
				Meta: map[string]any{
					"unicode": "测试",
					"symbols": "!@#$%^&*()",
					"quotes":  `"nested"quotes"`,
				},
			},
			verify: func(t *testing.T, path string) {
				data, _ := os.ReadFile(path) //nolint:errcheck // test verification
				assert.Contains(t, string(data), "测试")
				// JSON escapes & as \u0026
				assert.Contains(t, string(data), "symbols")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			walPath := filepath.Join(t.TempDir(), "test.wal")
			wal, err := NewWAL(walPath)
			require.NoError(t, err)
			defer wal.Close() //nolint:errcheck // test cleanup

			err = wal.WriteEntry(context.Background(), &tt.entry)
			require.NoError(t, err)

			// Force flush
			_ = wal.Close() //nolint:errcheck // test cleanup

			if tt.verify != nil {
				tt.verify(t, walPath)
			}
		})
	}
}

func TestWriteEntry_MultipleEntries(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	// Write multiple entries
	entries := []WALEntry{
		{Action: WALActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple},
		{Action: WALActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
		{Action: WALActionDelete, ID: "v1"},
	}

	for _, entry := range entries {
		err := wal.WriteEntry(context.Background(), &entry)
		require.NoError(t, err)
	}

	_ = wal.Close() //nolint:errcheck // test cleanup

	// Verify all entries written
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"id":"v1"`)
	assert.Contains(t, string(data), `"id":"v2"`)
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
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	require.Len(t, lines, 3, "each entry should be its own line, in order")
	assert.Contains(t, lines[0], `"id":"v1"`)
	assert.Contains(t, lines[0], `"action":"INSERT"`)
	assert.Contains(t, lines[1], `"id":"v2"`)
	assert.Contains(t, lines[2], `"id":"v1"`)
	assert.Contains(t, lines[2], `"action":"DELETE"`)
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

func TestWAL_Clear(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close() //nolint:errcheck // test cleanup

	// Write entry
	entry := WALEntry{Action: WALActionInsert, ID: "test", Vector: fixtures.Vec3dSimple}
	err = wal.WriteEntry(context.Background(), &entry)
	require.NoError(t, err)

	// Clear
	err = wal.Clear()
	require.NoError(t, err)

	// Verify file is empty
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size())
}

func TestWAL_Close(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	// Write entry
	entry := WALEntry{Action: WALActionInsert, ID: "test", Vector: fixtures.Vec3dSimple}
	err = wal.WriteEntry(context.Background(), &entry)
	require.NoError(t, err)

	// Close
	err = wal.Close()
	require.NoError(t, err)

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
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(n int) {
			defer wg.Done()
			entry := WALEntry{
				Action: WALActionInsert,
				ID:     string(rune('a' + n)),
				Vector: []float32{float32(n), float32(n * 2)},
			}
			_ = wal.WriteEntry(context.Background(), &entry) //nolint:errcheck // test concurrency
		}(i)
	}

	wg.Wait()
	// Test passes if no race conditions occur
}

// TestReplayWAL_Variations consolidates replay test variations
func TestReplayWAL_Variations(t *testing.T) {
	tests := []struct {
		name      string
		setup     func(string) error
		expectLen int
		validate  func(*testing.T, *VectorIndex[[]float32])
	}{
		{
			name: "empty_wal",
			setup: func(path string) error {
				// Create empty WAL
				wal, err := NewWAL(path)
				if err != nil {
					return err
				}
				return wal.Close()
			},
			expectLen: 0,
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name: "missing_wal",
			setup: func(path string) error {
				// Don't create WAL
				return nil
			},
			expectLen: 0,
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name: "single_insert",
			setup: func(path string) error {
				wal, err := NewWAL(path)
				if err != nil {
					return err
				}
				defer wal.Close() //nolint:errcheck // test setup
				return wal.WriteEntry(context.Background(), &WALEntry{
					Action: WALActionInsert,
					ID:     "vec1",
					Vector: fixtures.Vec3dSimple,
				})
			},
			expectLen: 1,
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, err := idx.IDMapper.ToUint32ID("vec1")
				require.NoError(t, err)
				assert.Contains(t, idx.Store, internalID)
			},
		},
		{
			name: "single_delete",
			setup: func(path string) error {
				wal, err := NewWAL(path)
				if err != nil {
					return err
				}
				defer wal.Close() //nolint:errcheck // test setup
				return wal.WriteEntry(context.Background(), &WALEntry{
					Action: WALActionDelete,
					ID:     "nonexistent",
				})
			},
			expectLen: 0,
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name: "multiple_inserts",
			setup: func(path string) error {
				wal, err := NewWAL(path)
				if err != nil {
					return err
				}
				defer wal.Close() //nolint:errcheck // test setup

				entries := []WALEntry{
					{Action: WALActionInsert, ID: "v1", Vector: fixtures.Vec3dSimple},
					{Action: WALActionInsert, ID: "v2", Vector: fixtures.Vec3dAlternate},
					{Action: WALActionInsert, ID: "v3", Vector: fixtures.Vec3dThird},
				}

				for _, e := range entries {
					if err := wal.WriteEntry(context.Background(), &e); err != nil {
						return err
					}
				}
				return nil
			},
			expectLen: 3,
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 3)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			walPath := filepath.Join(t.TempDir(), "test.wal")

			// Setup
			err := tt.setup(walPath)
			require.NoError(t, err)

			// Create fresh index
			idx := newTestIndex(t)

			// Replay
			err = idx.ReplayWAL(walPath)
			require.NoError(t, err)

			// Validate
			assert.Len(t, idx.Store, tt.expectLen)
			if tt.validate != nil {
				tt.validate(t, idx)
			}
		})
	}
}

// TestReplayWAL_ErrorHandling tests error cases
// Note: ReplayWAL logs errors but continues processing (non-fatal)
func TestReplayWAL_ErrorHandling(t *testing.T) {
	tests := []struct {
		name      string
		setupData string
	}{
		{
			name:      "malformed_json_logged",
			setupData: `{"action":"INSERT","id":"test1"`,
		},
		{
			name:      "partial_entry_processed",
			setupData: `{"action":"INSERT","id":"vec1"}` + "\n",
		},
		{
			name:      "unknown_action_logged",
			setupData: `{"action":"UNKNOWN","id":"test"}` + "\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			walPath := filepath.Join(t.TempDir(), "test.wal")

			// Write test data
			err := os.WriteFile(walPath, []byte(tt.setupData), 0o600)
			require.NoError(t, err)

			// Create fresh index
			idx := newTestIndex(t)

			// Replay - errors are logged but not returned
			err = idx.ReplayWAL(walPath)
			assert.NoError(t, err, "ReplayWAL logs errors but doesn't fail")
		})
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
	assert.Len(t, idx.Store, 1, "Should have 1 vector (v1 updated, v2 deleted)")

	internalID, err := idx.IDMapper.ToUint32ID("v1")
	require.NoError(t, err)
	assert.Contains(t, idx.Store, internalID)
	assert.Equal(t, fixtures.Vec3dThird, idx.Store[internalID].Vector, "Should have updated vector")
	// Note: JSON unmarshal changes types (int→float64, []string→[]interface{})
	// Just verify key fields exist
	assert.Contains(t, idx.Store[internalID].Metadata, "category")
	assert.Contains(t, idx.Store[internalID].Metadata, "nested")
	assert.Contains(t, idx.Store[internalID].Metadata, "tags")

	// v2 should be deleted
	_, err = idx.IDMapper.ToUint32ID("v2")
	assert.Error(t, err, "v2 should be deleted")
}

// TestWAL_Integration tests full snapshot + replay workflow
func TestWAL_Integration(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "test.wal")
	snapshotPath := filepath.Join(tmpDir, "snapshot.bin")

	// Create index with WAL
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	idx := newTestIndexWithWAL(t, wal)

	// Insert data
	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple)    //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, fixtures.MetaNested) //nolint:errcheck // test setup

	// Save snapshot
	err = idx.SaveToFile(context.Background(), snapshotPath)
	require.NoError(t, err)

	// Clear WAL after snapshot
	err = wal.Clear()
	require.NoError(t, err)

	// Insert more data after snapshot
	_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, nil) //nolint:errcheck // test setup

	_ = wal.Close() //nolint:errcheck // test cleanup

	// Simulate crash recovery: Load snapshot + Replay WAL
	newIdx := newTestIndex(t)
	err = newIdx.LoadFromFile(context.Background(), snapshotPath)
	require.NoError(t, err)

	err = newIdx.ReplayWAL(walPath)
	require.NoError(t, err)

	// Verify all data recovered
	assert.Len(t, newIdx.Store, 3, "Should have all 3 vectors (2 from snapshot + 1 from WAL)")

	id1, _ := newIdx.IDMapper.ToUint32ID("v1")
	id2, _ := newIdx.IDMapper.ToUint32ID("v2")
	id3, _ := newIdx.IDMapper.ToUint32ID("v3")

	assert.Contains(t, newIdx.Store, id1)
	assert.Contains(t, newIdx.Store, id2)
	assert.Contains(t, newIdx.Store, id3)
}

// Helper: create index with specific WAL
func newTestIndexWithWAL(t testing.TB, wal *WAL) *VectorIndex[[]float32] {
	t.Helper()
	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	return NewVectorIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity, nil)
}
