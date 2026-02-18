package api

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

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code, "Health endpoint should return 200")
	assert.JSONEq(t, `{"success":true,"data":{"status":"ok"}}`, w.Body.String(), "Health response should be correct")
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

	t.Run("POST_query_endpoint_exists", func(t *testing.T) {
		reqBody := map[string]interface{}{
			"vector": []float32{1.0, 2.0},
			"k":      1,
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code, "POST /api/v1/vectors/search should exist and return 200")
	})

	t.Run("DELETE_vectors_id_endpoint_exists", func(t *testing.T) {
		idx.Insert("delete_test", []float32{1.0, 2.0}, nil)

		req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/delete_test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code, "DELETE /api/v1/vectors/:id should exist and return 200")
		assert.JSONEq(t, `{"success":true,"data":{"status":"deleted","id":"delete_test"}}`, w.Body.String())
	})

	t.Run("DELETE_vectors_id_not_found", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/nonexistent", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code, "DELETE with unknown id should return 404")
		assert.JSONEq(t, `{"success":false,"data":null,"error":"vector not found"}`, w.Body.String())
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

func TestSwaggerEndpoints(t *testing.T) {
	idx := core.NewVectorIndex()
	router := SetupRouter(idx)

	t.Run("swagger_ui_returns_200", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/swagger/index.html", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code, "GET /swagger/index.html should return 200")
	})

	t.Run("swagger_doc_json_returns_200_with_spec", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/swagger/doc.json", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code, "GET /swagger/doc.json should return 200")
		assert.Contains(t, w.Body.String(), `"swagger": "2.0"`, "Response should contain OpenAPI 2.0 spec")
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
		assert.JSONEq(t, `{"success":true,"data":{"status":"inserted"}}`, w.Body.String())
	}

	assert.Len(t, idx.Store, len(vectors), "All vectors should be in the index")
}
