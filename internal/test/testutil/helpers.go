package testutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// NewTestVectorIndex creates a pre-populated VectorIndex for testing
func NewTestVectorIndex() *core.VectorIndex {
	idx := core.NewVectorIndex()

	idx.Insert("test1", []float32{1.0, 2.0, 3.0}, map[string]any{
		"label": "first",
		"type":  "test",
	})

	idx.Insert("test2", []float32{4.0, 5.0, 6.0}, map[string]any{
		"label": "second",
		"type":  "test",
	})

	idx.Insert("test3", []float32{7.0, 8.0, 9.0}, nil)

	return idx
}

// CreateTestVector generates a deterministic test vector
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
func AssertVectorInIndex(t *testing.T, idx *core.VectorIndex, id string, expectedVector []float32, expectedMetadata map[string]any) {
	t.Helper()

	require.Contains(t, idx.Store, id, "Vector %s should exist in index", id)

	node := idx.Store[id]
	assert.Equal(t, id, node.ID, "Vector ID should match")
	assert.Equal(t, expectedVector, node.Vector, "Vector data should match")
	assert.Equal(t, expectedMetadata, node.Metadata, "Metadata should match")
}

// AssertVectorCount verifies the number of vectors in the index
func AssertVectorCount(t *testing.T, idx *core.VectorIndex, expectedCount int) {
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
func CleanupStorageFile(t *testing.T, filepath string) {
	t.Helper()
	if err := os.Remove(filepath); err != nil && !os.IsNotExist(err) {
		t.Logf("Warning: Failed to cleanup storage file %s: %v", filepath, err)
	}
}

// PopulateIndexWithVectors inserts N vectors into an index for testing
func PopulateIndexWithVectors(idx *core.VectorIndex, count int) {
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("vec%d", i)
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		meta := map[string]any{
			"index": i,
			"type":  "test_vector",
			"batch": "populated",
		}
		idx.Insert(id, vec, meta)
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
	err := os.WriteFile(path, corruptedData, 0644)
	require.NoError(t, err, "Should be able to create corrupted file for testing")
}

// CreateReadOnlyFile creates a file with read-only permissions for testing
func CreateReadOnlyFile(t *testing.T, path string) {
	t.Helper()
	err := os.WriteFile(path, []byte("read-only content"), 0444)
	require.NoError(t, err, "Should be able to create read-only file")
}
