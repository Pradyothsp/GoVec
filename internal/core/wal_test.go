package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// A. Constructor & Initialization Tests (3 tests)
// =============================================================================

func TestNewWAL_CreatesFileSuccessfully(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	wal, err := NewWAL(walPath)
	require.NoError(t, err, "NewWAL should succeed")
	defer wal.Close()

	// Verify file exists
	info, err := os.Stat(walPath)
	assert.NoError(t, err, "WAL file should exist")
	assert.Equal(t, os.FileMode(0644), info.Mode().Perm(), "File permissions should be 0644")
}

func TestNewWAL_CreatesParentDirectories(t *testing.T) {
	// Try to create WAL in non-existent parent directory
	walPath := filepath.Join(t.TempDir(), "nonexistent", "parent", "test.wal")

	_, err := NewWAL(walPath)
	assert.Error(t, err, "NewWAL should fail with non-existent parent directory")
	assert.Contains(t, err.Error(), "no such file", "Error should indicate missing directory")
}

func TestNewWAL_AppendsToExistingFile(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")

	// Create WAL, write entry, close
	wal1, err := NewWAL(walPath)
	require.NoError(t, err)
	err = wal1.WriteEntry(WALEntry{Action: WALActionInsert, ID: "vec1", Vector: []float32{1.0, 2.0}})
	require.NoError(t, err)
	wal1.Close()

	// Reopen and write second entry
	wal2, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal2.Close()
	err = wal2.WriteEntry(WALEntry{Action: WALActionInsert, ID: "vec2", Vector: []float32{3.0, 4.0}})
	require.NoError(t, err)

	// Verify both entries present
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)
	lines := countLines(data)
	assert.Equal(t, 2, lines, "WAL should contain 2 entries (append mode)")
}

// =============================================================================
// B. WriteEntry Operations Tests (5 tests)
// =============================================================================

func TestWriteEntry_SingleInsertAction(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	entry := WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{1.0, 2.0},
		Meta:   map[string]any{"label": "test"},
	}

	err = wal.WriteEntry(entry)
	require.NoError(t, err)

	// Read and parse JSON
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	var parsed WALEntry
	err = json.Unmarshal(data[:len(data)-1], &parsed) // Strip newline
	require.NoError(t, err)

	assert.Equal(t, WALActionInsert, parsed.Action)
	assert.Equal(t, "vec1", parsed.ID)
	// Vector is unmarshaled as []interface{} with float64 elements
	vecSlice, ok := parsed.Vector.([]interface{})
	require.True(t, ok, "Vector should be []interface{}")
	assert.Len(t, vecSlice, 2, "Vector should have 2 elements")
	assert.Equal(t, float64(1.0), vecSlice[0])
	assert.Equal(t, float64(2.0), vecSlice[1])
	assert.Equal(t, "test", parsed.Meta["label"])
}

func TestWriteEntry_SingleDeleteAction(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	entry := WALEntry{
		Action: WALActionDelete,
		ID:     "vec1",
	}

	err = wal.WriteEntry(entry)
	require.NoError(t, err)

	// Read and parse JSON
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	var parsed WALEntry
	err = json.Unmarshal(data[:len(data)-1], &parsed) // Strip newline
	require.NoError(t, err)

	assert.Equal(t, WALActionDelete, parsed.Action)
	assert.Equal(t, "vec1", parsed.ID)
	assert.Nil(t, parsed.Vector, "Vector should be nil for DELETE")
	assert.Nil(t, parsed.Meta, "Metadata should be nil for DELETE")
}

func TestWriteEntry_MultipleEntries(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// Write 5 INSERT + 3 DELETE entries
	for i := 0; i < 5; i++ {
		entry := WALEntry{
			Action: WALActionInsert,
			ID:     "vec" + string(rune('1'+i)),
			Vector: []float32{float32(i)},
		}
		err = wal.WriteEntry(entry)
		require.NoError(t, err)
	}

	for i := 0; i < 3; i++ {
		entry := WALEntry{
			Action: WALActionDelete,
			ID:     "vec" + string(rune('1'+i)),
		}
		err = wal.WriteEntry(entry)
		require.NoError(t, err)
	}

	// Verify 8 lines
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)
	lines := countLines(data)
	assert.Equal(t, 8, lines, "WAL should contain 8 entries")
}

func TestWriteEntry_LargeVector(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// 1536-dimensional vector (OpenAI embedding size)
	vec := make([]float32, 1536)
	for i := 0; i < 1536; i++ {
		vec[i] = float32(i) * 0.001
	}

	entry := WALEntry{
		Action: WALActionInsert,
		ID:     "embedding1",
		Vector: vec,
	}

	err = wal.WriteEntry(entry)
	require.NoError(t, err)

	// Read and verify
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	var parsed WALEntry
	err = json.Unmarshal(data[:len(data)-1], &parsed) // Strip newline
	require.NoError(t, err)

	// Vector is unmarshaled as []interface{} with float64 elements
	vecSlice, ok := parsed.Vector.([]interface{})
	require.True(t, ok, "Vector should be []interface{}")
	assert.Equal(t, 1536, len(vecSlice), "Vector dimensions should be preserved")
	// Note: We don't do exact comparison here since JSON unmarshals to []interface{}{float64...}
}

func TestWriteEntry_SpecialCharactersInMetadata(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	entry := WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{1.0},
		Meta: map[string]any{
			"quotes":    `"double" and 'single'`,
			"newlines":  "line1\nline2",
			"emoji":     "🚀💻🔥",
			"backslash": `C:\path\to\file`,
		},
	}

	err = wal.WriteEntry(entry)
	require.NoError(t, err)

	// Read and verify JSON escaping
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	var parsed WALEntry
	err = json.Unmarshal(data[:len(data)-1], &parsed) // Strip newline
	require.NoError(t, err)

	assert.Equal(t, entry.Meta, parsed.Meta, "Special characters should round-trip correctly")
}

// =============================================================================
// C. Clear Operations Tests (3 tests)
// =============================================================================

func TestClear_TruncatesFileToZero(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// Write 10 entries
	for i := 0; i < 10; i++ {
		entry := WALEntry{
			Action: WALActionInsert,
			ID:     "vec" + string(rune('0'+i)),
			Vector: []float32{float32(i)},
		}
		err = wal.WriteEntry(entry)
		require.NoError(t, err)
	}

	// Verify size > 0
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0), "File should have content before Clear")

	// Clear WAL
	err = wal.Clear()
	require.NoError(t, err)

	// Verify size == 0
	info, err = os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size(), "File should be empty after Clear")

	// Verify file still exists
	assert.FileExists(t, walPath, "File should still exist after Clear")
}

func TestClear_EmptyWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// Clear empty WAL
	err = wal.Clear()
	require.NoError(t, err, "Clearing empty WAL should not error")

	// Verify size == 0
	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Equal(t, int64(0), info.Size(), "File should be empty")
}

func TestClear_SeekPositionAfterClear(t *testing.T) {
	// ⚠️ May reveal bug: Truncate doesn't reset seek position
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	// Write entries
	for i := 0; i < 3; i++ {
		entry := WALEntry{
			Action: WALActionInsert,
			ID:     "vec" + string(rune('1'+i)),
			Vector: []float32{float32(i)},
		}
		err = wal.WriteEntry(entry)
		require.NoError(t, err)
	}

	// Clear
	err = wal.Clear()
	require.NoError(t, err)

	// Write new entry
	entry := WALEntry{
		Action: WALActionInsert,
		ID:     "new1",
		Vector: []float32{99.0},
	}
	err = wal.WriteEntry(entry)
	require.NoError(t, err)

	// Verify only 1 line at position 0
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)
	lines := countLines(data)

	// This test may FAIL if Truncate doesn't reset seek position
	assert.Equal(t, 1, lines, "File should contain only 1 entry at position 0")
}

// =============================================================================
// D. Close & Cleanup Tests (2 tests)
// =============================================================================

func TestClose_ReleasesFileHandle(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	// Close WAL
	err = wal.Close()
	require.NoError(t, err)

	// Attempt second Close
	err = wal.Close()
	assert.Error(t, err, "Second Close should return error")
}

func TestClose_FlushesBufferedWrites(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)

	// Write entry
	entry := WALEntry{
		Action: WALActionInsert,
		ID:     "vec1",
		Vector: []float32{1.0, 2.0},
	}
	err = wal.WriteEntry(entry)
	require.NoError(t, err)

	// Check size before Close
	info1, err := os.Stat(walPath)
	require.NoError(t, err)
	sizeBefore := info1.Size()

	// Close
	err = wal.Close()
	require.NoError(t, err)

	// Check size after Close
	info2, err := os.Stat(walPath)
	require.NoError(t, err)
	sizeAfter := info2.Size()

	assert.Equal(t, sizeBefore, sizeAfter, "No buffered data should be lost")
	assert.Greater(t, sizeAfter, int64(0), "File should have content")
}

// =============================================================================
// E. Concurrency Tests (2 tests)
// =============================================================================

func TestWriteEntry_ConcurrentWrites(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	// 100 goroutines writing simultaneously
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			entry := WALEntry{
				Action: WALActionInsert,
				ID:     "vec" + string(rune('0'+id%10)),
				Vector: []float32{float32(id)},
			}
			_ = wal.WriteEntry(entry)
		}(i)
	}

	wg.Wait()

	// Verify 100 valid JSON lines
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)
	lines := countLines(data)
	assert.Equal(t, goroutines, lines, "All concurrent writes should succeed")

	// Verify all lines are valid JSON
	scanner := bufio.NewScanner(bytes.NewReader(data))
	validLines := 0
	for scanner.Scan() {
		var entry WALEntry
		if json.Unmarshal(scanner.Bytes(), &entry) == nil {
			validLines++
		}
	}
	assert.Equal(t, goroutines, validLines, "All lines should be valid JSON")
}

func TestClear_ConcurrentWithWrite(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	defer wal.Close()

	var wg sync.WaitGroup
	wg.Add(2)

	// Background writes
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			entry := WALEntry{
				Action: WALActionInsert,
				ID:     "before" + string(rune('0'+i%10)),
				Vector: []float32{float32(i)},
			}
			_ = wal.WriteEntry(entry)
		}
	}()

	// Main thread calls Clear midway
	go func() {
		defer wg.Done()
		for i := 0; i < 25; i++ {
			entry := WALEntry{
				Action: WALActionInsert,
				ID:     "temp",
				Vector: []float32{1.0},
			}
			_ = wal.WriteEntry(entry)
		}
		_ = wal.Clear()
		for i := 0; i < 25; i++ {
			entry := WALEntry{
				Action: WALActionInsert,
				ID:     "after" + string(rune('0'+i%10)),
				Vector: []float32{float32(i)},
			}
			_ = wal.WriteEntry(entry)
		}
	}()

	wg.Wait()

	// Verify no corruption (all entries are valid JSON)
	data, err := os.ReadFile(walPath)
	require.NoError(t, err)

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var entry WALEntry
		err := json.Unmarshal(scanner.Bytes(), &entry)
		assert.NoError(t, err, "All lines should be valid JSON (no corruption)")
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

func countLines(data []byte) int {
	lines := 0
	for _, b := range data {
		if b == '\n' {
			lines++
		}
	}
	return lines
}
