package handlers

import (
	"bytes"
	"context"
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

// The successful path of every route, through the real router, is in
// internal/test/integration/rest_test.go; the JSON shape of each response is
// pinned in responses_test.go. These tests cover what a handler decides
// itself: which requests are the caller's fault, and which failures are ours.

// newVectorRouter mounts every vector handler over engine, as the real router
// does but without its middleware.
func newVectorRouter(engine index.Engine) *gin.Engine {
	h := NewVectorHandler(engine)
	router := gin.New()
	router.POST("/vectors", h.Insert)
	router.POST("/vectors/batch", h.BatchInsert)
	router.GET("/vectors/:id", h.GetByID)
	router.POST("/vectors/search", h.Search)
	router.DELETE("/vectors/:id", h.Delete)
	return router
}

func serve(router *gin.Engine, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

// TestVectorRoutes_BadRequests: a request the caller got wrong is a 4xx that
// says what is wrong -- never a 500, which would blame the server. Wrong-width
// vectors used to come back as 500s; a missing k used to decode as 0 and
// return every vector.
func TestVectorRoutes_BadRequests(t *testing.T) {
	tests := []struct {
		name         string
		method, path string
		body         string
		wantCode     int
		wantError    string
	}{
		{"insert: malformed JSON", http.MethodPost, "/vectors", `{"id": "x", "vector": [1,2,3`, http.StatusBadRequest, ""},
		{"insert: empty body", http.MethodPost, "/vectors", ``, http.StatusBadRequest, ""},
		{"insert: missing id", http.MethodPost, "/vectors", `{"vector": [1, 2, 3]}`, http.StatusBadRequest, ""},
		{"insert: empty id", http.MethodPost, "/vectors", `{"id": "", "vector": [1, 2, 3]}`, http.StatusBadRequest, ""},
		{"insert: missing vector", http.MethodPost, "/vectors", `{"id": "x"}`, http.StatusBadRequest, ""},
		{"insert: vector not an array", http.MethodPost, "/vectors", `{"id": "x", "vector": "abc"}`, http.StatusBadRequest, ""},
		{"insert: wrong width", http.MethodPost, "/vectors", `{"id": "x", "vector": [1, 2]}`, http.StatusBadRequest, "index requires 3"},
		{"batch: no vectors", http.MethodPost, "/vectors/batch", `{"vectors": []}`, http.StatusBadRequest, ""},
		{"search: malformed JSON", http.MethodPost, "/vectors/search", `{"vector": [1,2,3, "k": 5}`, http.StatusBadRequest, "invalid"},
		{"search: missing vector", http.MethodPost, "/vectors/search", `{"k": 5}`, http.StatusBadRequest, ""},
		{"search: empty vector", http.MethodPost, "/vectors/search", `{"vector": [], "k": 5}`, http.StatusBadRequest, "must not be empty"},
		{"search: wrong width", http.MethodPost, "/vectors/search", `{"vector": [1, 2], "k": 5}`, http.StatusBadRequest, "index requires 3"},
		{"search: missing k", http.MethodPost, "/vectors/search", `{"vector": [1, 2, 3]}`, http.StatusBadRequest, ""},
		{"search: negative k", http.MethodPost, "/vectors/search", `{"vector": [1, 2, 3], "k": -1}`, http.StatusBadRequest, ""},
		{"get: unknown id", http.MethodGet, "/vectors/missing", ``, http.StatusNotFound, "not found"},
		{"delete: unknown id", http.MethodDelete, "/vectors/missing", ``, http.StatusNotFound, "not found"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: an index 3 dimensions wide.
			idx := testutil.NewTestIndex(t)
			require.NoError(t, idx.Insert(context.Background(), "stored", []float32{1, 2, 3}, core.SparseVector{}, nil))
			router := newVectorRouter(idx)

			// Act
			rec := serve(router, tt.method, tt.path, tt.body)

			// Assert
			assert.Equal(t, tt.wantCode, rec.Code, rec.Body.String())
			assert.Contains(t, rec.Body.String(), `"success":false`)
			assert.Contains(t, rec.Body.String(), tt.wantError)
			assert.Equal(t, 1, idx.Len(), "a rejected request must change nothing")
		})
	}
}

// k=0 is a valid request for nothing: an empty list, not an error.
func TestSearch_KZero_ReturnsEmptyList(t *testing.T) {
	// Arrange
	idx := testutil.NewTestIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "stored", []float32{1, 2, 3}, core.SparseVector{}, nil))

	// Act
	rec := serve(newVectorRouter(idx), http.MethodPost, "/vectors/search", `{"vector": [1, 2, 3], "k": 0}`)

	// Assert
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.JSONEq(t, `{"success":true,"data":[]}`, rec.Body.String())
}

// A genuine server fault must still read as one: the 400s above narrow the
// 500 case rather than replace it. A batch the WAL can't record is not a
// partial success either -- none of it may be applied.
func TestVectorRoutes_EngineFailure_Returns500(t *testing.T) {
	tests := []struct {
		name, path, body string
	}{
		{"insert", "/vectors", `{"id": "v1", "vector": [1, 2, 3]}`},
		{"batch", "/vectors/batch", `{"vectors": [{"id": "v1", "vector": [1, 2, 3]}, {"id": "v2", "vector": [4, 5, 6]}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: a closed WAL makes every write fail.
			wal, err := index.NewWAL(filepath.Join(t.TempDir(), "test.wal"))
			require.NoError(t, err)
			idx := testutil.NewIndexOnWAL(t, wal)
			require.NoError(t, wal.Close())

			// Act
			rec := serve(newVectorRouter(idx), http.MethodPost, tt.path, tt.body)

			// Assert
			assert.Equal(t, http.StatusInternalServerError, rec.Code)
			assert.Contains(t, rec.Body.String(), `"success":false`)
			assert.Zero(t, idx.Len())
		})
	}
}

// One bad item in a batch is reported against its ID; the good ones are
// still inserted.
func TestBatchInsert_BadItem_DoesNotBlockTheRest(t *testing.T) {
	// Arrange
	idx := testutil.NewTestIndex(t)
	body := `{"vectors": [
		{"id": "good1", "vector": [1, 2, 3]},
		{"id": "bad1", "vector": [1, 2, 3], "sparse_vector": {"indices": [0, 1], "values": [0.5]}},
		{"id": "good2", "vector": [4, 5, 6]}
	]}`

	// Act
	rec := serve(newVectorRouter(idx), http.MethodPost, "/vectors/batch", body)

	// Assert
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"inserted_count":2`)
	assert.Contains(t, rec.Body.String(), `"id":"bad1"`)
	assert.Equal(t, 2, idx.Len())
}
