package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

func TestNewSystemHandler(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	h := NewSystemHandler(idx, "/tmp/test.bin")

	assert.NotNil(t, h)
	assert.Equal(t, idx, h.Engine)
	assert.Equal(t, "/tmp/test.bin", h.DataPath)
}

func TestStats_EmptyIndex(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	h := NewSystemHandler(idx, "")

	router := gin.New()
	router.GET("/stats", h.Stats)

	req := httptest.NewRequest(http.MethodGet, "/stats", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.JSONEq(t, `{"success":true,"data":{"vector_count":0}}`, rec.Body.String())
}

func TestStats_AfterInsert(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1.0, 2.0}, core.SparseVector{}, nil))

	h := NewSystemHandler(idx, "")

	router := gin.New()
	router.GET("/stats", h.Stats)

	req := httptest.NewRequest(http.MethodGet, "/stats", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.JSONEq(t, `{"success":true,"data":{"vector_count":1}}`, rec.Body.String())
}

func TestInfo_EmptyIndex(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	h := NewSystemHandler(idx, "")

	router := gin.New()
	router.GET("/info", h.Info)

	req := httptest.NewRequest(http.MethodGet, "/info", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var env struct {
		Success bool                   `json:"success"`
		Data    map[string]interface{} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &env))
	assert.True(t, env.Success)
	assert.Equal(t, float64(0), env.Data["dimensions"], "dimensions should be 0 on empty index")
	assert.Equal(t, float64(0), env.Data["vector_count"])
}

func TestInfo_DimensionsSetAfterInsert(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1.0, 2.0, 3.0}, core.SparseVector{}, nil))

	h := NewSystemHandler(idx, "")

	router := gin.New()
	router.GET("/info", h.Info)

	req := httptest.NewRequest(http.MethodGet, "/info", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var env struct {
		Success bool                   `json:"success"`
		Data    map[string]interface{} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &env))
	assert.True(t, env.Success)
	assert.Equal(t, float64(3), env.Data["dimensions"], "dimensions should be set from first insert")
	assert.Equal(t, float64(1), env.Data["vector_count"])
}

func TestFlush_Success(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1.0, 2.0}, core.SparseVector{}, nil))

	path := filepath.Join(t.TempDir(), "test.bin")
	h := NewSystemHandler(idx, path)

	router := gin.New()
	router.POST("/admin/flush", h.Flush)

	req := httptest.NewRequest(http.MethodPost, "/admin/flush", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.JSONEq(t, `{"success":true,"data":{"status":"flushed"}}`, rec.Body.String())

	_, err := os.Stat(path)
	assert.NoError(t, err, "flush should create file at DataPath")
}

func TestFlush_InvalidPath(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	h := NewSystemHandler(idx, "/nonexistent/dir/test.bin")

	router := gin.New()
	router.POST("/admin/flush", h.Flush)

	req := httptest.NewRequest(http.MethodPost, "/admin/flush", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
}
