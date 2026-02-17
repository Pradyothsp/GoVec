package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
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

func TestVectorHandler_Search_ValidRequest(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	// Insert test vectors
	idx.Insert("v1", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"label": "first"})
	idx.Insert("v2", []float32{0.9, 0.1, 0.0}, map[string]interface{}{"label": "second"})
	idx.Insert("v3", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"label": "third"})

	req := SearchRequest{
		Vector: []float32{1.0, 0.0, 0.0},
		K:      2,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Len(t, results, 2, "Should return k results")
	assert.Equal(t, "v1", results[0].ID, "First result should be most similar")
	assert.InDelta(t, 1.0, results[0].Score, 0.001)
}

func TestVectorHandler_Search_WithoutK(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	// Insert test vectors
	for i := 1; i <= 5; i++ {
		vec := []float32{float32(i), float32(i * 2)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, nil)
	}

	req := SearchRequest{
		Vector: []float32{2.0, 4.0},
		// K is not set (default 0)
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Len(t, results, 5, "Should return all results when k=0")
}

func TestVectorHandler_Search_InvalidJSON(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader([]byte("invalid json")))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestVectorHandler_Search_MissingVector(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	req := map[string]interface{}{
		"k": 5,
		// vector is missing
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestVectorHandler_Search_EmptyVector(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	idx.Insert("v1", []float32{1.0, 2.0}, nil)

	req := SearchRequest{
		Vector: []float32{},
		K:      5,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Contains(t, response["error"], "empty query vector")
}

func TestVectorHandler_Search_WrongContentType(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	req := SearchRequest{
		Vector: []float32{1.0, 2.0},
		K:      5,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "text/plain")

	handler.Search(c)

	// Gin is lenient with content-type, may still work or return 400
	assert.Contains(t, []int{http.StatusBadRequest, http.StatusOK}, w.Code)
}

func TestVectorHandler_Search_VariousKValues(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	// Insert 10 vectors
	for i := 1; i <= 10; i++ {
		vec := []float32{float32(i), float32(i * 2)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, map[string]interface{}{"index": i})
	}

	tests := []struct {
		name          string
		k             int
		expectedCount int
	}{
		{"k_1", 1, 1},
		{"k_5", 5, 5},
		{"k_10", 10, 10},
		{"k_100", 100, 10}, // More than available
		{"k_0", 0, 10},     // Return all
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := SearchRequest{
				Vector: []float32{5.0, 10.0},
				K:      tt.k,
			}

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			body, _ := json.Marshal(req)
			c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			handler.Search(c)

			assert.Equal(t, http.StatusOK, w.Code)

			var results []core.SearchResult
			err := json.Unmarshal(w.Body.Bytes(), &results)
			require.NoError(t, err)

			assert.Len(t, results, tt.expectedCount)
		})
	}
}

func TestVectorHandler_Search_ReturnsCorrectJSONFormat(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	metadata := map[string]interface{}{
		"category": "test",
		"count":    42,
		"active":   true,
	}

	idx.Insert("v1", []float32{1.0, 2.0, 3.0}, metadata)
	idx.Insert("v2", []float32{2.0, 3.0, 4.0}, map[string]interface{}{"label": "second"})

	req := SearchRequest{
		Vector: []float32{1.0, 2.0, 3.0},
		K:      2,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	// Verify structure
	for _, result := range results {
		assert.NotEmpty(t, result.ID)
		assert.NotNil(t, result.Score)
		// Metadata can be nil or non-nil
	}
}

func TestVectorHandler_Search_Integration(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	// Insert diverse vectors
	idx.Insert("identical", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"type": "identical"})
	idx.Insert("similar", []float32{0.9, 0.1, 0.0}, map[string]interface{}{"type": "similar"})
	idx.Insert("orthogonal", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"type": "orthogonal"})
	idx.Insert("opposite", []float32{-1.0, 0.0, 0.0}, map[string]interface{}{"type": "opposite"})

	req := SearchRequest{
		Vector: []float32{1.0, 0.0, 0.0},
		K:      3,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	require.Len(t, results, 3)

	// Verify results are sorted by score
	assert.Equal(t, "identical", results[0].ID)
	assert.InDelta(t, 1.0, results[0].Score, 0.001)

	assert.Equal(t, "similar", results[1].ID)
	assert.Greater(t, results[1].Score, float32(0.8))

	assert.Equal(t, "orthogonal", results[2].ID)

	// Verify metadata is included
	assert.Equal(t, "identical", results[0].Meta["type"])
	assert.Equal(t, "similar", results[1].Meta["type"])
}

func TestVectorHandler_Search_EmptyIndex(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	req := SearchRequest{
		Vector: []float32{1.0, 2.0, 3.0},
		K:      5,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Empty(t, results, "Empty index should return empty results")
}

func TestVectorHandler_Search_DimensionMismatch(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	// Insert 3D vectors
	idx.Insert("v1", []float32{1.0, 2.0, 3.0}, nil)
	idx.Insert("v2", []float32{4.0, 5.0, 6.0}, nil)

	// Search with 2D vector
	req := SearchRequest{
		Vector: []float32{1.0, 2.0},
		K:      5,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Contains(t, response["error"], "vector dimensions mismatch")
}

func TestVectorHandler_Search_LargeVectors(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	// Insert 1536-dimensional vectors (OpenAI embedding size)
	vec1 := make([]float32, 1536)
	vec2 := make([]float32, 1536)
	for i := 0; i < 1536; i++ {
		vec1[i] = float32(i+1) * 0.001
		// vec2 has different direction (reversed second half)
		if i < 768 {
			vec2[i] = float32(i+1) * 0.001
		} else {
			vec2[i] = float32(1536-i) * 0.001
		}
	}

	idx.Insert("large1", vec1, map[string]interface{}{"model": "text-embedding-ada-002"})
	idx.Insert("large2", vec2, map[string]interface{}{"model": "text-embedding-ada-002"})

	req := SearchRequest{
		Vector: vec1,
		K:      2,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Len(t, results, 2)
	// First result should be large1 (identical to query)
	assert.Equal(t, "large1", results[0].ID)
	assert.InDelta(t, 1.0, results[0].Score, 0.001)
	// Second result should have lower similarity
	assert.Less(t, results[1].Score, results[0].Score)
}

func TestVectorHandler_Search_NegativeK(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	req := SearchRequest{
		Vector: []float32{1.0, 2.0, 3.0},
		K:      -1,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Contains(t, response["error"], "k cannot be negative")
}

func TestVectorHandler_Search_WithFilter(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	idx.Insert("v1", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"label": "a"})
	idx.Insert("v2", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"label": "b"})
	idx.Insert("v3", []float32{0.9, 0.1, 0.0}, map[string]interface{}{"label": "a"})

	req := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"k":      10,
		"filter": map[string]interface{}{"label": "a"},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	require.Len(t, results, 2)
	resultIDs := make([]string, len(results))
	for i, r := range results {
		resultIDs[i] = r.ID
	}
	assert.Contains(t, resultIDs, "v1")
	assert.Contains(t, resultIDs, "v3")
	assert.NotContains(t, resultIDs, "v2")
}

func TestVectorHandler_Search_FilterNoResults(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	idx.Insert("v1", []float32{1.0, 0.0}, map[string]interface{}{"type": "X"})
	idx.Insert("v2", []float32{0.9, 0.1}, map[string]interface{}{"type": "X"})

	req := map[string]interface{}{
		"vector": []float32{1.0, 0.0},
		"k":      10,
		"filter": map[string]interface{}{"type": "Y"},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Empty(t, results)
}

func TestVectorHandler_Search_FilterAndK(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	for i := 1; i <= 5; i++ {
		vec := []float32{float32(i), float32(i * 2)}
		idx.Insert(fmt.Sprintf("v%d", i), vec, map[string]interface{}{"tag": "t"})
	}

	req := map[string]interface{}{
		"vector": []float32{5.0, 10.0},
		"k":      2,
		"filter": map[string]interface{}{"tag": "t"},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Len(t, results, 2)
}

func TestVectorHandler_Search_NoFilterBackwardCompatible(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	idx.Insert("v1", []float32{1.0, 0.0}, map[string]interface{}{"type": "A"})
	idx.Insert("v2", []float32{0.9, 0.1}, map[string]interface{}{"type": "B"})
	idx.Insert("v3", []float32{0.8, 0.2}, map[string]interface{}{"type": "C"})

	// No filter field in request
	req := map[string]interface{}{
		"vector": []float32{1.0, 0.0},
		"k":      10,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	assert.Len(t, results, 3, "No filter should return all results")
}

func TestVectorHandler_Delete(t *testing.T) {
	tests := []struct {
		name           string
		setup          func(*core.VectorIndex)
		id             string
		expectedStatus int
		expectedBody   string
		checkIndex     func(*testing.T, *core.VectorIndex)
	}{
		{
			name: "success",
			setup: func(idx *core.VectorIndex) {
				idx.Insert("v1", []float32{1.0, 2.0}, nil)
			},
			id:             "v1",
			expectedStatus: http.StatusOK,
			expectedBody:   `{"status":"deleted","id":"v1"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.NotContains(t, idx.Store, "v1")
			},
		},
		{
			name:           "not_found",
			setup:          func(idx *core.VectorIndex) {},
			id:             "missing",
			expectedStatus: http.StatusNotFound,
			expectedBody:   `{"error":"vector not found"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store)
			},
		},
		{
			name:           "empty_id",
			setup:          func(idx *core.VectorIndex) {},
			id:             "",
			expectedStatus: http.StatusBadRequest,
			expectedBody:   `{"error":"id is required"}`,
			checkIndex: func(t *testing.T, idx *core.VectorIndex) {
				assert.Empty(t, idx.Store)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := core.NewVectorIndex()
			tt.setup(idx)
			handler := NewVectorHandler(idx)

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			if tt.id != "" {
				c.Params = gin.Params{gin.Param{Key: "id", Value: tt.id}}
			}
			c.Request = httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/"+tt.id, nil)

			handler.Delete(c)

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

func TestVectorHandler_Delete_EdgeCases(t *testing.T) {
	t.Run("special_characters_in_id", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		specialID := "vec/with/slashes"
		idx.Insert(specialID, []float32{1.0, 2.0}, nil)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = gin.Params{gin.Param{Key: "id", Value: specialID}}
		c.Request = httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/"+specialID, nil)

		handler.Delete(c)

		assert.Equal(t, http.StatusOK, w.Code)
		assert.JSONEq(t, `{"status":"deleted","id":"vec/with/slashes"}`, w.Body.String())
		assert.NotContains(t, idx.Store, specialID)
	})

	t.Run("double_delete", func(t *testing.T) {
		idx := core.NewVectorIndex()
		handler := NewVectorHandler(idx)

		idx.Insert("v1", []float32{1.0, 2.0}, nil)

		// First delete — should succeed
		w1 := httptest.NewRecorder()
		c1, _ := gin.CreateTestContext(w1)
		c1.Params = gin.Params{gin.Param{Key: "id", Value: "v1"}}
		c1.Request = httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/v1", nil)
		handler.Delete(c1)
		assert.Equal(t, http.StatusOK, w1.Code)

		// Second delete — must return 404
		w2 := httptest.NewRecorder()
		c2, _ := gin.CreateTestContext(w2)
		c2.Params = gin.Params{gin.Param{Key: "id", Value: "v1"}}
		c2.Request = httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/v1", nil)
		handler.Delete(c2)
		assert.Equal(t, http.StatusNotFound, w2.Code)
		assert.JSONEq(t, `{"error":"vector not found"}`, w2.Body.String())
	})
}

func TestVectorHandler_Search_MultipleFilterKeys(t *testing.T) {
	idx := core.NewVectorIndex()
	handler := NewVectorHandler(idx)

	idx.Insert("v1", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"cat": "A", "active": true})
	idx.Insert("v2", []float32{0.9, 0.1, 0.0}, map[string]interface{}{"cat": "A", "active": false})
	idx.Insert("v3", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"cat": "B", "active": true})

	req := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"k":      10,
		"filter": map[string]interface{}{"cat": "A", "active": true},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body, _ := json.Marshal(req)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.Search(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var results []core.SearchResult
	err := json.Unmarshal(w.Body.Bytes(), &results)
	require.NoError(t, err)

	require.Len(t, results, 1)
	assert.Equal(t, "v1", results[0].ID)
}
