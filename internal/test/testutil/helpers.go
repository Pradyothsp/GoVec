package testutil

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/index"

	"github.com/Pradyothsp/govec/internal/core"
)

// NewTestIndex creates an empty VectorIndex backed by a temp WAL for testing.
// The WAL is closed automatically via t.Cleanup.
func NewTestIndex(t testing.TB) *index.VectorIndex[[]float32] {
	t.Helper()
	wal, err := index.NewWAL(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("NewTestIndex: failed to create WAL: %v", err)
	}
	t.Cleanup(func() { _ = wal.Close() }) //nolint:errcheck // test cleanup

	// Create index with identity encode function and cosine similarity
	identityFunc := func(v []float32) []float32 { return v }
	return index.NewVectorIndex[[]float32](wal, nil, identityFunc, core.CosineSimilarity)
}

// NewTestVectorIndex creates a pre-populated VectorIndex for testing.
func NewTestVectorIndex(t testing.TB) *index.VectorIndex[[]float32] {
	t.Helper()
	idx := NewTestIndex(t)

	_ = idx.Insert("test1", []float32{1.0, 2.0, 3.0}, core.SparseVector{}, map[string]any{ //nolint:errcheck // test helper
		"label": "first",
		"type":  "test",
	})

	_ = idx.Insert("test2", []float32{4.0, 5.0, 6.0}, core.SparseVector{}, map[string]any{ //nolint:errcheck // test helper
		"label": "second",
		"type":  "test",
	})

	_ = idx.Insert("test3", []float32{7.0, 8.0, 9.0}, core.SparseVector{}, nil) //nolint:errcheck // test helper

	return idx
}

// CreateTestVector generates a deterministic test vector
//
//nolint:gocritic // unnamedResult: id parameter name would shadow return name
func CreateTestVector(id string, dimensions int) (string, []float32, map[string]any) {
	vec := make([]float32, dimensions)
	for i := 0; i < dimensions; i++ {
		vec[i] = float32(i) * 0.1
	}

	metadata := map[string]any{
		"id":         id,
		"dimensions": dimensions,
		"type":       "test_vector",
	}

	return id, vec, metadata
}

// AssertHTTPResponse validates HTTP response status and body
func AssertHTTPResponse(t *testing.T, recorder *httptest.ResponseRecorder, expectedStatus int, expectedBody string) {
	t.Helper()

	assert.Equal(t, expectedStatus, recorder.Code, "HTTP status code should match")

	if expectedBody != "" {
		assert.JSONEq(t, expectedBody, recorder.Body.String(), "Response body should match")
	}
}

// MakeJSONRequest creates an HTTP request with JSON body
func MakeJSONRequest(t *testing.T, method, url string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()

	var reqBody *bytes.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		require.NoError(t, err, "Failed to marshal request body")
		reqBody = bytes.NewReader(jsonData)
	} else {
		reqBody = bytes.NewReader([]byte{})
	}

	req := httptest.NewRequest(method, url, reqBody)
	req.Header.Set("Content-Type", "application/json")

	recorder := httptest.NewRecorder()

	return recorder
}

// CreateVectorRequest represents a vector insertion request payload
type CreateVectorRequest struct {
	ID       string         `json:"id"`
	Vector   []float32      `json:"vector"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewCreateVectorRequest creates a standard vector request for testing
func NewCreateVectorRequest(id string, dimensions int, withMetadata bool) CreateVectorRequest {
	vec := make([]float32, dimensions)
	for i := 0; i < dimensions; i++ {
		vec[i] = float32(i)
	}

	req := CreateVectorRequest{
		ID:     id,
		Vector: vec,
	}

	if withMetadata {
		req.Metadata = map[string]any{
			"test":  true,
			"id":    id,
			"label": "test_vector",
		}
	}

	return req
}

// AssertVectorInIndex verifies a vector exists in the index with expected values
func AssertVectorInIndex(t *testing.T, idx *index.VectorIndex[[]float32], id string, expectedVector []float32, expectedMetadata map[string]any) {
	t.Helper()

	require.Contains(t, idx.Store, id, "Vector %s should exist in index", id)

	node := idx.Store[id]
	assert.Equal(t, id, node.ID, "Vector ID should match")
	assert.Equal(t, expectedVector, node.Vector, "Vector data should match")
	assert.Equal(t, expectedMetadata, node.Metadata, "Metadata should match")
}

// AssertVectorCount verifies the number of vectors in the index
func AssertVectorCount(t *testing.T, idx *index.VectorIndex[[]float32], expectedCount int) {
	t.Helper()
	assert.Len(t, idx.Store, expectedCount, "Index should contain %d vectors", expectedCount)
}

// =============================================================================
// Persistence Test Helpers
// =============================================================================

// CreateTempStorageFile creates a temporary file for testing persistence
func CreateTempStorageFile(t *testing.T, filename string) string {
	t.Helper()
	return filepath.Join(t.TempDir(), filename)
}

// CleanupStorageFile removes test storage files
func CleanupStorageFile(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		t.Logf("Warning: Failed to cleanup storage file %s: %v", path, err)
	}
}

// PopulateIndexWithVectors inserts N vectors into an index for testing
func PopulateIndexWithVectors(idx *index.VectorIndex[[]float32], count int) {
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("vec%d", i)
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		meta := map[string]any{
			"index": i,
			"type":  "test_vector",
			"batch": "populated",
		}
		_ = idx.Insert(id, vec, core.SparseVector{}, meta) //nolint:errcheck // test helper
	}
}

// AssertFileExists verifies that a file exists at the given path
func AssertFileExists(t *testing.T, path string) {
	t.Helper()
	_, err := os.Stat(path)
	assert.NoError(t, err, "File should exist at path: %s", path)
}

// AssertFileNotExists verifies that no file exists at the given path
func AssertFileNotExists(t *testing.T, path string) {
	t.Helper()
	_, err := os.Stat(path)
	assert.True(t, os.IsNotExist(err), "File should not exist at path: %s", path)
}

// CorruptFile creates an intentionally corrupted file for testing
func CorruptFile(t *testing.T, path string) {
	t.Helper()
	corruptedData := []byte("This is not valid GOB data!\nIt should fail to decode.")
	err := os.WriteFile(path, corruptedData, 0o600)
	require.NoError(t, err, "Should be able to create corrupted file for testing")
}

// CreateReadOnlyFile creates a file with read-only permissions for testing
func CreateReadOnlyFile(t *testing.T, path string) {
	t.Helper()
	err := os.WriteFile(path, []byte("read-only content"), 0o444) //nolint:gosec // intentionally setting read-only permissions for test
	require.NoError(t, err, "Should be able to create read-only file")
}

// =============================================================================
// WAL Test Helpers
// =============================================================================

// NewWALWithPath creates a WAL at a specific path for testing
func NewWALWithPath(t testing.TB, path string) *index.WAL {
	t.Helper()
	wal, err := index.NewWAL(path)
	require.NoError(t, err, "Failed to create WAL at path: %s", path)
	t.Cleanup(func() { _ = wal.Close() }) //nolint:errcheck // test cleanup
	return wal
}

// WriteWALEntry manually writes entries for test setup
func WriteWALEntry(t testing.TB, path string, entry *index.WALEntry) {
	t.Helper()
	wal := NewWALWithPath(t, path)
	err := wal.WriteEntry(entry)
	require.NoError(t, err, "Failed to write WAL entry")
}

// ReadWALEntries reads all entries from a WAL file
func ReadWALEntries(t testing.TB, path string) []index.WALEntry {
	t.Helper()

	file, err := os.Open(path) //nolint:gosec // path is test-controlled
	require.NoError(t, err, "Failed to open WAL file")
	defer file.Close() //nolint:errcheck // read-only test operation

	var entries []index.WALEntry
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var entry index.WALEntry
		err := json.Unmarshal(scanner.Bytes(), &entry)
		require.NoError(t, err, "Failed to unmarshal WAL entry")
		entries = append(entries, entry)
	}
	require.NoError(t, scanner.Err(), "Scanner error reading WAL")

	return entries
}

// AssertWALContainsEntries verifies entry count
func AssertWALContainsEntries(t testing.TB, path string, expectedCount int) {
	t.Helper()
	entries := ReadWALEntries(t, path)
	assert.Len(t, entries, expectedCount, "WAL should contain %d entries", expectedCount)
}

// AssertWALEmpty verifies WAL file is empty (0 bytes)
func AssertWALEmpty(t testing.TB, path string) {
	t.Helper()
	info, err := os.Stat(path)
	require.NoError(t, err, "Failed to stat WAL file")
	assert.Equal(t, int64(0), info.Size(), "WAL file should be empty")
}

// MakeWALReadOnly sets file permissions for failure testing
func MakeWALReadOnly(t testing.TB, path string) {
	t.Helper()
	err := os.Chmod(path, 0o444) //nolint:gosec // intentionally read-only for test
	require.NoError(t, err, "Failed to make WAL read-only")
}

// CorruptWALFile creates corrupted WAL for error testing
func CorruptWALFile(t testing.TB, path, corruptionType string) {
	t.Helper()

	var data []byte
	switch corruptionType {
	case "binary":
		data = []byte{0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA}
	case "malformed_json":
		data = []byte(`{"action":"INSERT","id":"test1"`)
	case "partial":
		data = []byte(`{"action":"INSERT","id":"vec1"}` + "\n")
	default:
		data = []byte("corrupted data\n")
	}

	err := os.WriteFile(path, data, 0o644) //nolint:gosec // test file with intentional corruption
	require.NoError(t, err, "Failed to create corrupted WAL file")
}
