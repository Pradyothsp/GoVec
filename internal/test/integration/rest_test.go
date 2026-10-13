package integration

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

// restClient sends JSON requests to router and returns the status and raw body.
type restClient struct {
	t      *testing.T
	router *gin.Engine
}

func (c restClient) do(method, path string, body any) (int, string) {
	c.t.Helper()
	var payload []byte
	if body != nil {
		var err error
		payload, err = json.Marshal(body)
		require.NoError(c.t, err)
	}
	req := httptest.NewRequest(method, path, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c.router.ServeHTTP(rec, req)
	return rec.Code, rec.Body.String()
}

// TestREST_VectorLifecycle drives one index through every vector endpoint of
// the real router, checking each response body in full: the handler tests
// mount one route at a time, so this is what checks the routes are wired and
// the {success, data, error} envelope reaches the client unchanged.
func TestREST_VectorLifecycle(t *testing.T) {
	// Arrange
	gin.SetMode(gin.TestMode)
	c := restClient{t: t, router: api.SetupRouter(testutil.NewTestIndex(t), "")}

	// Act + Assert, one step at a time: each step depends on the ones before.
	code, body := c.do(http.MethodPost, "/api/v1/vectors", map[string]any{
		"id": "doc1", "vector": []float32{1, 0, 0}, "metadata": map[string]any{"category": "tech", "year": 2024},
	})
	require.Equal(t, http.StatusCreated, code, body)
	assert.JSONEq(t, `{"success":true,"data":{"status":"inserted"}}`, body)

	code, body = c.do(http.MethodPost, "/api/v1/vectors/batch", map[string]any{"vectors": []map[string]any{
		{"id": "doc2", "vector": []float32{0.9, 0.1, 0}, "metadata": map[string]any{"category": "tech", "year": 2023}},
		{"id": "doc3", "vector": []float32{0, 1, 0}, "metadata": map[string]any{"category": "science", "year": 2024}},
	}})
	require.Equal(t, http.StatusOK, code, body)
	assert.JSONEq(t, `{"success":true,"data":{"inserted_count":2}}`, body)

	code, body = c.do(http.MethodGet, "/api/v1/vectors/doc1", nil)
	require.Equal(t, http.StatusOK, code, body)
	assert.JSONEq(t, `{"success":true,"data":{"id":"doc1","vector":[1,0,0],"metadata":{"category":"tech","year":2024}}}`, body)

	// Filters arrive as decoded JSON, so 2024 is a float64 here; it must still
	// match. doc2 is the closer tech match by vector but fails on year.
	code, body = c.do(http.MethodPost, "/api/v1/vectors/search", map[string]any{
		"vector": []float32{1, 0, 0}, "k": 5, "filter": map[string]any{"category": "tech", "year": 2024},
	})
	require.Equal(t, http.StatusOK, code, body)
	assert.JSONEq(t, `{"success":true,"data":[{"id":"doc1","score":1,"meta":{"category":"tech","year":2024}}]}`, body)

	code, body = c.do(http.MethodPost, "/api/v1/vectors/search", map[string]any{"vector": []float32{1, 0, 0}, "k": 2})
	require.Equal(t, http.StatusOK, code, body)
	var ranked struct {
		Data []struct{ ID string } `json:"data"`
	}
	require.NoError(t, json.Unmarshal([]byte(body), &ranked))
	require.Len(t, ranked.Data, 2)
	assert.Equal(t, []string{"doc1", "doc2"}, []string{ranked.Data[0].ID, ranked.Data[1].ID}, "nearest first, cut to k")

	code, body = c.do(http.MethodDelete, "/api/v1/vectors/doc2", nil)
	require.Equal(t, http.StatusOK, code, body)
	assert.JSONEq(t, `{"success":true,"data":{"status":"deleted","id":"doc2"}}`, body)

	code, body = c.do(http.MethodDelete, "/api/v1/vectors/doc2", nil)
	assert.Equal(t, http.StatusNotFound, code)
	assert.JSONEq(t, `{"success":false,"data":null,"error":"vector not found"}`, body)

	code, body = c.do(http.MethodGet, "/api/v1/stats", nil)
	require.Equal(t, http.StatusOK, code, body)
	assert.JSONEq(t, `{"success":true,"data":{"vector_count":2}}`, body)

	code, body = c.do(http.MethodPost, "/api/v1/admin/reset", nil)
	require.Equal(t, http.StatusOK, code, body)
	assert.JSONEq(t, `{"success":true,"data":{"status":"reset"}}`, body)

	code, _ = c.do(http.MethodGet, "/api/v1/vectors/doc1", nil)
	assert.Equal(t, http.StatusNotFound, code, "a reset must remove every vector")

	code, body = c.do(http.MethodPost, "/api/v1/vectors", map[string]any{"id": "doc1", "vector": []float32{0, 1}})
	assert.Equal(t, http.StatusCreated, code, "a reset index takes a new ID and a new width: %s", body)
}
