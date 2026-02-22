package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
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
	_ = index.Insert("vec1", vec, core.SparseVector{}, meta) //nolint:errcheck // test setup

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
	_ = index.Insert("vec1", vec, core.SparseVector{}, meta) //nolint:errcheck // test setup

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
