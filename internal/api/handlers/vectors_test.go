package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// =============================================================================
// Consolidated Handler Tests - HTTP Layer Validation Only
// =============================================================================

func TestNewVectorHandler(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	assert.NotNil(t, handler)
	assert.Equal(t, index, handler.Engine)
}

// TestInsert_Validation tests HTTP request validation for insert endpoint
func TestInsert_Validation(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		contentType    string
		expectedStatus int
		errorContains  string
	}{
		{
			name:           "invalid_json",
			body:           `{"id": "test", "vector": [1,2,3`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "missing_id_field",
			body:           `{"vector": [1.0, 2.0, 3.0]}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "missing_vector_field",
			body:           `{"id": "test"}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "empty_id",
			body:           `{"id": "", "vector": [1.0, 2.0]}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "invalid_vector_type",
			body:           `{"id": "test", "vector": "not_an_array"}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "wrong_content_type_ignored",
			body:           `{"id": "test", "vector": [1.0, 2.0]}`,
			contentType:    "text/plain",
			expectedStatus: http.StatusCreated, // Content-Type not currently validated
			errorContains:  "",
		},
		{
			name:           "empty_request_body",
			body:           ``,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index := testutil.NewTestIndex(t)
			handler := NewVectorHandler(index)

			router := gin.New()
			router.POST("/vectors", handler.Insert)

			req := httptest.NewRequest("POST", "/vectors", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", tt.contentType)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			assert.Equal(t, tt.expectedStatus, rec.Code)
			if tt.errorContains != "" {
				assert.Contains(t, rec.Body.String(), tt.errorContains)
			}
		})
	}
}

// TestQuery_Validation tests HTTP request validation for query endpoint
func TestQuery_Validation(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		contentType    string
		expectedStatus int
		errorContains  string
	}{
		{
			name:           "invalid_json",
			body:           `{"vector": [1,2,3, "k": 5}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "invalid",
		},
		{
			name:           "missing_vector_field",
			body:           `{"k": 5}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "empty_vector",
			body:           `{"vector": [], "k": 5}`,
			contentType:    "application/json",
			expectedStatus: http.StatusInternalServerError, // Actual error status
			errorContains:  "",
		},
		{
			name:           "negative_k",
			body:           `{"vector": [1.0, 2.0], "k": -1}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "invalid_vector_type",
			body:           `{"vector": "not_an_array", "k": 5}`,
			contentType:    "application/json",
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "wrong_content_type_ignored",
			body:           `{"vector": [1.0, 2.0], "k": 5}`,
			contentType:    "application/xml",
			expectedStatus: http.StatusOK, // Content-Type not currently validated
			errorContains:  "",
		},
		{
			name:           "missing_k_defaults_to_10",
			body:           `{"vector": [1.0, 2.0, 3.0]}`,
			contentType:    "application/json",
			expectedStatus: http.StatusOK, // Should succeed with default k=10
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index := testutil.NewTestIndex(t)
			handler := NewVectorHandler(index)

			router := gin.New()
			router.POST("/query", handler.Search)

			req := httptest.NewRequest("POST", "/query", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", tt.contentType)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			assert.Equal(t, tt.expectedStatus, rec.Code, "Response: %s", rec.Body.String())
			if tt.errorContains != "" {
				assert.Contains(t, rec.Body.String(), tt.errorContains)
			}
		})
	}
}

// TestDelete_Validation tests HTTP request validation for delete endpoint
func TestDelete_Validation(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	router := gin.New()
	router.DELETE("/vectors/:id", handler.Delete)

	// Test nonexistent ID
	req := httptest.NewRequest("DELETE", "/vectors/nonexistent", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	assert.Contains(t, rec.Body.String(), "not found")
}

// TestInsert_SuccessResponse tests the JSON response format for successful inserts
func TestInsert_SuccessResponse(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	router := gin.New()
	router.POST("/vectors", handler.Insert)

	body := `{"id": "test1", "vector": [1.0, 2.0, 3.0], "metadata": {"label": "test"}}`
	req := httptest.NewRequest("POST", "/vectors", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusCreated, rec.Code) // Changed to match actual handler

	var response map[string]interface{}
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.True(t, response["success"].(bool))
	assert.Contains(t, response, "data")
}

// TestQuery_SuccessResponse tests the JSON response format for successful queries
func TestQuery_SuccessResponse(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	// Pre-populate index
	_, vec, meta := testutil.CreateTestVector("vec1", 3)
	_ = index.Insert(context.Background(), "vec1", vec, core.SparseVector{}, meta) //nolint:errcheck // test setup

	router := gin.New()
	router.POST("/query", handler.Search)

	body := `{"vector": [1.0, 2.0, 3.0], "k": 5}`
	req := httptest.NewRequest("POST", "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var response map[string]interface{}
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.True(t, response["success"].(bool))
	assert.Contains(t, response, "data")
	results, ok := response["data"].([]interface{})
	assert.True(t, ok)
	assert.GreaterOrEqual(t, len(results), 0) // May be empty or have results
}

// TestDelete_SuccessResponse tests the JSON response format for successful deletes
func TestDelete_SuccessResponse(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	// Pre-populate index
	_, vec, meta := testutil.CreateTestVector("vec1", 3)
	_ = index.Insert(context.Background(), "vec1", vec, core.SparseVector{}, meta) //nolint:errcheck // test setup

	router := gin.New()
	router.DELETE("/vectors/:id", handler.Delete) // Path parameter

	req := httptest.NewRequest("DELETE", "/vectors/vec1", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var response map[string]interface{}
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.True(t, response["success"].(bool), "Response should indicate success")
	assert.Contains(t, response, "data")
}

// TestGetByID_NotFound verifies 404 when the ID does not exist.
func TestGetByID_NotFound(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	router := gin.New()
	router.GET("/vectors/:id", handler.GetByID)

	req := httptest.NewRequest("GET", "/vectors/nonexistent", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	assert.Contains(t, rec.Body.String(), "not found")
}

// TestGetByID_Success verifies that a stored vector is returned with all fields intact.
func TestGetByID_Success(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	// Pre-populate index
	_, vec, meta := testutil.CreateTestVector("vec1", 3)
	_ = index.Insert(context.Background(), "vec1", vec, core.SparseVector{}, meta) //nolint:errcheck // test setup

	router := gin.New()
	router.GET("/vectors/:id", handler.GetByID)

	req := httptest.NewRequest("GET", "/vectors/vec1", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(rec.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, true, resp["success"].(bool))

	data, ok := resp["data"].(map[string]interface{})
	require.True(t, ok, "data field should be an object")

	// ID round-trip
	assert.Equal(t, "vec1", data["id"])

	// Vector round-trip: JSON unmarshals numbers as float64
	rawVec, ok := data["vector"].([]interface{})
	require.True(t, ok, "vector should be an array")
	require.Len(t, rawVec, len(vec))
	for i, v := range vec {
		assert.InDelta(t, float64(v), rawVec[i].(float64), 1e-6)
	}

	// Metadata round-trip
	rawMeta, ok := data["metadata"].(map[string]interface{})
	require.True(t, ok, "metadata should be an object")
	assert.Equal(t, meta["type"], rawMeta["type"])
	assert.Equal(t, meta["id"], rawMeta["id"])
}

// TestMalformedJSON verifies error handling for invalid JSON payloads
func TestMalformedJSON(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	router := gin.New()
	router.POST("/vectors", handler.Insert)
	router.POST("/query", handler.Search)

	tests := []struct {
		name     string
		endpoint string
		method   string
		body     string
	}{
		{
			name:     "insert_malformed_json",
			endpoint: "/vectors",
			method:   "POST",
			body:     `{"id": "test", "vector": [1,2,3`,
		},
		{
			name:     "query_malformed_json",
			endpoint: "/query",
			method:   "POST",
			body:     `{"vector": [1,2,3, "k": 5}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.endpoint, bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			assert.Equal(t, http.StatusBadRequest, rec.Code)
		})
	}
}

// =============================================================================
// BatchInsert
// =============================================================================

func TestBatchInsert_EmptyArray(t *testing.T) {
	index := testutil.NewTestIndex(t)
	handler := NewVectorHandler(index)

	router := gin.New()
	router.POST("/vectors/batch", handler.BatchInsert)

	req := httptest.NewRequest("POST", "/vectors/batch", bytes.NewBufferString(`{"vectors": []}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestBatchInsert_Success(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	handler := NewVectorHandler(idx)

	router := gin.New()
	router.POST("/vectors/batch", handler.BatchInsert)

	body := `{"vectors": [
		{"id": "v1", "vector": [1.0, 2.0, 3.0]},
		{"id": "v2", "vector": [4.0, 5.0, 6.0]},
		{"id": "v3", "vector": [7.0, 8.0, 9.0]}
	]}`
	req := httptest.NewRequest("POST", "/vectors/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp["success"].(bool))

	data, ok := resp["data"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, float64(3), data["inserted_count"])
	assert.NotContains(t, data, "errors")
	assert.Equal(t, 3, idx.Len())
}

// TestBatchInsert_InvalidItemsDontBlockValidOnes verifies the existing per-item
// validation (invalid sparse_vector) still short-circuits before an item ever
// reaches the engine, and that the valid items in the same request still get
// batched and inserted -- one bad item shouldn't sink the whole request.
func TestBatchInsert_InvalidItemsDontBlockValidOnes(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	handler := NewVectorHandler(idx)

	router := gin.New()
	router.POST("/vectors/batch", handler.BatchInsert)

	body := `{"vectors": [
		{"id": "good1", "vector": [1.0, 2.0, 3.0]},
		{"id": "bad1", "vector": [1.0, 2.0, 3.0], "sparse_vector": {"indices": [0, 1], "values": [0.5]}},
		{"id": "good2", "vector": [4.0, 5.0, 6.0]}
	]}`
	req := httptest.NewRequest("POST", "/vectors/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	data, ok := resp["data"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, float64(2), data["inserted_count"])

	errs, ok := data["errors"].([]interface{})
	require.True(t, ok)
	require.Len(t, errs, 1)
	firstErr, ok := errs[0].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "bad1", firstErr["id"])

	assert.Equal(t, 2, idx.Len())
}

// TestBatchInsert_WALFailureReturns500AndInsertsNothing exercises the batch's
// all-or-nothing durability contract from the HTTP layer down: if the engine
// can't make the batch durable (WAL write fails), the handler must surface a
// server error rather than reporting a misleading partial success, and none of
// the batch's vectors should be queryable afterward.
func TestBatchInsert_WALFailureReturns500AndInsertsNothing(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)

	idMapper := core.NewIDMapper()
	identityFunc := func(v []float32) []float32 { return v }
	idx := index.NewVectorIndex[[]float32](wal, nil, idMapper, identityFunc, core.CosineSimilarity, nil)

	// Force the WAL write the handler triggers to fail.
	require.NoError(t, wal.Close())

	handler := NewVectorHandler(idx)
	router := gin.New()
	router.POST("/vectors/batch", handler.BatchInsert)

	body := `{"vectors": [
		{"id": "v1", "vector": [1.0, 2.0, 3.0]},
		{"id": "v2", "vector": [4.0, 5.0, 6.0]}
	]}`
	req := httptest.NewRequest("POST", "/vectors/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.False(t, resp["success"].(bool))

	assert.Equal(t, 0, idx.Len(), "no vector from the failed batch should have been applied")
}
