package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestNewVectorHandler(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	require.NotNil(t, handler, "Handler should not be nil")
	require.NotNil(t, handler.Index, "Handler Index should not be nil")
	assert.Equal(t, idx, handler.Index, "Handler should reference the provided index")
}

func TestVectorHandler_Insert(t *testing.T) {
	tests := []struct {
		name           string
		requestBody    interface{}
		expectedStatus int
		expectedBody   string
		checkIndex     func(*testing.T, *core.VectorIndex)
	}{
		{
			name: "valid_request_with_metadata",
			requestBody: CreateVectorRequest{
				ID:       "v1",
				Vector:   []float32{1.0, 2.0, 3.0},
				Metadata: map[string]interface{}{"label": "test", "count": 42},
			},
			expectedStatus: http.StatusCreated,
			expectedBody:   `{"status":"inserted"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				require.Contains(t, idx.Store, "v1")
				assert.Equal(t, "v1", idx.Store["v1"].ID)
				assert.Equal(t, []float32{1.0, 2.0, 3.0}, idx.Store["v1"].Vector)
				assert.Equal(t, "test", idx.Store["v1"].Metadata["label"])
				assert.Equal(t, float64(42), idx.Store["v1"].Metadata["count"])
			},
		},
		{
			name: "valid_request_without_metadata",
			requestBody: CreateVectorRequest{
				ID:     "v2",
				Vector: []float32{4.0, 5.0},
			},
			expectedStatus: http.StatusCreated,
			expectedBody:   `{"status":"inserted"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				require.Contains(t, idx.Store, "v2")
				assert.Equal(t, []float32{4.0, 5.0}, idx.Store["v2"].Vector)
				assert.Nil(t, idx.Store["v2"].Metadata)
			},
		},
		{
			name: "valid_request_with_empty_metadata",
			requestBody: CreateVectorRequest{
				ID:       "v3",
				Vector:   []float32{1.0},
				Metadata: map[string]interface{}{},
			},
			expectedStatus: http.StatusCreated,
			expectedBody:   `{"status":"inserted"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				require.Contains(t, idx.Store, "v3")
				assert.Empty(t, idx.Store["v3"].Metadata)
			},
		},
		{
			name: "missing_id",
			requestBody: map[string]interface{}{
				"vector": []float32{1.0, 2.0},
			},
			expectedStatus: http.StatusBadRequest,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store, "No vector should be inserted")
			},
		},
		{
			name: "missing_vector",
			requestBody: map[string]interface{}{
				"id": "v4",
			},
			expectedStatus: http.StatusBadRequest,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store, "No vector should be inserted")
			},
		},
		{
			name: "missing_both_id_and_vector",
			requestBody: map[string]interface{}{
				"metadata": map[string]interface{}{"test": true},
			},
			expectedStatus: http.StatusBadRequest,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name:           "malformed_json",
			requestBody:    "not a valid json",
			expectedStatus: http.StatusBadRequest,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name: "empty_string_id",
			requestBody: CreateVectorRequest{
				ID:     "",
				Vector: []float32{1.0, 2.0},
			},
			expectedStatus: http.StatusBadRequest,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name: "empty_vector_array",
			requestBody: CreateVectorRequest{
				ID:     "v5",
				Vector: []float32{},
			},
			expectedStatus: http.StatusCreated,
			expectedBody:   `{"status":"inserted"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				require.Contains(t, idx.Store, "v5")
				assert.Empty(t, idx.Store["v5"].Vector)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := core.NewVectorIndex()
			handler := NewVectorHandler(idx)

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			var body []byte
			if str, ok := tt.requestBody.(string); ok {
				body = []byte(str)
			} else {
				body, _ = json.Marshal(tt.requestBody)
			}

			c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			handler.Insert(c)

			assert.Equal(t, tt.expectedStatus, w.Code, "HTTP status should match")

			if tt.expectedBody != "" {
				assert.JSONEq(t, tt.expectedBody, w.Body.String(), "Response body should match")
			}

			if tt.checkIndex != nil {
				tt.checkIndex(t, idx)
			}
		})
	}
}

func TestVectorHandler_Insert_EdgeCases(t *testing.T) {
	t.Run("large_vector", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		largeVec := make([]float32, 1536)
		for i := range largeVec {
			largeVec[i] = float32(i) * 0.1
		}

		req := CreateVectorRequest{
			ID:     "large_vec",
			Vector: largeVec,
			Metadata: map[string]interface{}{
				"model": "text-embedding-ada-002",
			},
		}

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body, _ := json.Marshal(req)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")

		handler.Insert(c)

		assert.Equal(t, http.StatusCreated, w.Code)
		require.Contains(t, idx.Store, "large_vec")
		assert.Len(t, idx.Store["large_vec"].Vector, 1536)
	})

	t.Run("special_characters_in_id", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		specialIDs := []string{
			"vec-with-dashes",
			"vec_with_underscores",
			"vec.with.dots",
			"vec:with:colons",
			"vec with spaces",
			"vec@special!",
		}

		for _, id := range specialIDs {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			req := CreateVectorRequest{
				ID:     id,
				Vector: []float32{1.0, 2.0},
			}

			body, _ := json.Marshal(req)
			c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			handler.Insert(c)

			assert.Equal(t, http.StatusCreated, w.Code, "Should accept ID: %s", id)
			require.Contains(t, idx.Store, id, "Vector with ID %s should be inserted", id)
		}
	})

	t.Run("unicode_in_metadata", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		req := CreateVectorRequest{
			ID:     "unicode_vec",
			Vector: []float32{1.0, 2.0},
			Metadata: map[string]interface{}{
				"label":       "测试",
				"description": "こんにちは",
				"emoji":       "🚀",
			},
		}

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body, _ := json.Marshal(req)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")

		handler.Insert(c)

		assert.Equal(t, http.StatusCreated, w.Code)
		require.Contains(t, idx.Store, "unicode_vec")
		assert.Equal(t, "测试", idx.Store["unicode_vec"].Metadata["label"])
	})

	t.Run("nested_metadata", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		req := CreateVectorRequest{
			ID:     "nested_vec",
			Vector: []float32{1.0, 2.0},
			Metadata: map[string]interface{}{
				"config": map[string]interface{}{
					"version": "1.0",
					"settings": map[string]interface{}{
						"enabled": true,
						"count":   10,
					},
				},
				"tags": []string{"test", "nested"},
			},
		}

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body, _ := json.Marshal(req)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")

		handler.Insert(c)

		assert.Equal(t, http.StatusCreated, w.Code)
		require.Contains(t, idx.Store, "nested_vec")
	})

	t.Run("wrong_content_type", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		req := CreateVectorRequest{
			ID:     "test",
			Vector: []float32{1.0},
		}

		body, _ := json.Marshal(req)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		c.Request.Header.Set("Content-Type", "text/plain")

		handler.Insert(c)

		// Gin is lenient with content-type, so this may still succeed
		// This test documents current behavior rather than enforcing strict validation
		assert.Contains(t, []int{http.StatusBadRequest, http.StatusCreated}, w.Code)
	})
}

func TestVectorHandler_Insert_Overwrite(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	req1 := CreateVectorRequest{
		ID:       "vec1",
		Vector:   []float32{1.0, 2.0},
		Metadata: map[string]interface{}{"version": 1},
	}

	w1 := httptest.NewRecorder()
	c1, _ := gin.CreateTestContext(w1)
	body1, _ := json.Marshal(req1)
	c1.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body1))
	c1.Request.Header.Set("Content-Type", "application/json")
	handler.Insert(c1)

	assert.Equal(t, http.StatusCreated, w1.Code)
	require.Contains(t, idx.Store, "vec1")
	assert.Equal(t, []float32{1.0, 2.0}, idx.Store["vec1"].Vector)

	req2 := CreateVectorRequest{
		ID:       "vec1",
		Vector:   []float32{3.0, 4.0, 5.0},
		Metadata: map[string]interface{}{"version": 2, "updated": true},
	}

	w2 := httptest.NewRecorder()
	c2, _ := gin.CreateTestContext(w2)
	body2, _ := json.Marshal(req2)
	c2.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body2))
	c2.Request.Header.Set("Content-Type", "application/json")
	handler.Insert(c2)

	assert.Equal(t, http.StatusCreated, w2.Code)
	require.Contains(t, idx.Store, "vec1")
	assert.Equal(t, []float32{3.0, 4.0, 5.0}, idx.Store["vec1"].Vector, "Vector should be updated")
	assert.Equal(t, float64(2), idx.Store["vec1"].Metadata["version"])
	assert.Len(t, idx.Store, 1, "Should still have only one vector")
}

func TestVectorHandler_Insert_MultipleVectors(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	vectors := []CreateVectorRequest{
		{ID: "v1", Vector: []float32{1.0, 2.0}, Metadata: map[string]interface{}{"label": "first"}},
		{ID: "v2", Vector: []float32{3.0, 4.0}, Metadata: map[string]interface{}{"label": "second"}},
		{ID: "v3", Vector: []float32{5.0, 6.0}, Metadata: nil},
	}

	for _, vec := range vectors {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body, _ := json.Marshal(vec)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")

		handler.Insert(c)

		assert.Equal(t, http.StatusCreated, w.Code)
		assert.JSONEq(t, `{"status":"inserted"}`, w.Body.String())
	}

	assert.Len(t, idx.Store, len(vectors), "All vectors should be inserted")

	for _, vec := range vectors {
		require.Contains(t, idx.Store, vec.ID, "Vector %s should exist", vec.ID)
		assert.Equal(t, vec.Vector, idx.Store[vec.ID].Vector)
	}
}
