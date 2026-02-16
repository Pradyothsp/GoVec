package api

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

func TestSetupRouter(t *testing.T) {
	idx := core.NewVectorIndex()
	router := SetupRouter(idx)

	require.NotNil(t, router, "Router should not be nil")
}

func TestHealthEndpoint(t *testing.T) {
	idx := core.NewVectorIndex()
	router := SetupRouter(idx)

	t.Run("GET_returns_200", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code, "Health endpoint should return 200")
		assert.JSONEq(t, `{"status":"ok"}`, w.Body.String(), "Health response should be correct")
	})

	t.Run("POST_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/health", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "POST to health should not be allowed")
	})

	t.Run("PUT_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPut, "/health", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "PUT to health should not be allowed")
	})

	t.Run("DELETE_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodDelete, "/health", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "DELETE to health should not be allowed")
	})
}

func TestRouterEndpointRegistration(t *testing.T) {
	idx := core.NewVectorIndex()
	router := SetupRouter(idx)

	t.Run("POST_vectors_endpoint_exists", func(t *testing.T) {
		reqBody := map[string]interface{}{
			"id":     "test",
			"vector": []float32{1.0, 2.0},
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusCreated, w.Code, "POST /api/v1/vectors should exist and return 201")
	})

	t.Run("GET_vectors_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/vectors", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "GET /api/v1/vectors should not be allowed")
	})

	t.Run("PUT_vectors_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPut, "/api/v1/vectors", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "PUT /api/v1/vectors should not be allowed")
	})

	t.Run("DELETE_vectors_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "DELETE /api/v1/vectors should not be allowed")
	})

	t.Run("unknown_route_returns_404", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/unknown", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "Unknown routes should return 404")
	})

	t.Run("root_path_returns_404", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "Root path should return 404")
	})
}

func TestRouterConfiguration(t *testing.T) {
	t.Run("router_uses_injected_index", func(t *testing.T) {
		idx := core.NewVectorIndex()
		idx.Insert("existing", []float32{1.0, 2.0}, map[string]any{"pre-existing": true})

		router := SetupRouter(idx)

		assert.Len(t, idx.Store, 1, "Index should have pre-existing vector")

		reqBody := map[string]interface{}{
			"id":     "new_vector",
			"vector": []float32{3.0, 4.0},
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusCreated, w.Code)
		assert.Len(t, idx.Store, 2, "Index should now have both vectors")
		require.Contains(t, idx.Store, "existing")
		require.Contains(t, idx.Store, "new_vector")
	})

	t.Run("api_v1_group_prefix", func(t *testing.T) {
		idx := core.NewVectorIndex()
		router := SetupRouter(idx)

		reqBody := map[string]interface{}{
			"id":     "test",
			"vector": []float32{1.0},
		}

		body, _ := json.Marshal(reqBody)

		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusCreated, w.Code, "Route should be under /api/v1 prefix")
	})

	t.Run("without_api_v1_prefix_fails", func(t *testing.T) {
		idx := core.NewVectorIndex()
		router := SetupRouter(idx)

		reqBody := map[string]interface{}{
			"id":     "test",
			"vector": []float32{1.0},
		}

		body, _ := json.Marshal(reqBody)

		req := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "Route without /api/v1 prefix should fail")
	})
}

func TestRouterMultipleRequests(t *testing.T) {
	idx := core.NewVectorIndex()
	router := SetupRouter(idx)

	vectors := []map[string]interface{}{
		{"id": "v1", "vector": []float32{1.0, 2.0}},
		{"id": "v2", "vector": []float32{3.0, 4.0}},
		{"id": "v3", "vector": []float32{5.0, 6.0}},
	}

	for _, vec := range vectors {
		body, _ := json.Marshal(vec)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusCreated, w.Code)
		assert.JSONEq(t, `{"status":"inserted"}`, w.Body.String())
	}

	assert.Len(t, idx.Store, len(vectors), "All vectors should be in the index")
}
