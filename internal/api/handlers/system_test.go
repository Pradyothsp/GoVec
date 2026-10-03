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
	"github.com/Pradyothsp/govec/internal/release"
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

func TestInfo_ReportsServerVersion(t *testing.T) {
	original := release.Version
	release.Version = "v9.9.9-test"
	t.Cleanup(func() { release.Version = original })

	h := NewSystemHandler(testutil.NewTestIndex(t), "")

	router := gin.New()
	router.GET("/info", h.Info)

	req := httptest.NewRequest(http.MethodGet, "/info", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var env struct {
		Data map[string]interface{} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &env))
	assert.Equal(t, "v9.9.9-test", env.Data["version"])
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
	// A path under a regular file can't be created by anyone, root included.
	notADir := filepath.Join(t.TempDir(), "a-file")
	require.NoError(t, os.WriteFile(notADir, nil, 0o600))
	h := NewSystemHandler(idx, filepath.Join(notADir, "test.bin"))

	router := gin.New()
	router.POST("/admin/flush", h.Flush)

	req := httptest.NewRequest(http.MethodPost, "/admin/flush", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
}

func TestReset_Success(t *testing.T) {
	idx := testutil.NewTestIndex(t)
	require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1.0, 2.0}, core.SparseVector{}, nil))
	require.NoError(t, idx.Insert(context.Background(), "v2", []float32{3.0, 4.0}, core.SparseVector{}, nil))
	require.Equal(t, 2, idx.Len())

	h := NewSystemHandler(idx, "")

	router := gin.New()
	router.POST("/admin/reset", h.Reset)

	req := httptest.NewRequest(http.MethodPost, "/admin/reset", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.JSONEq(t, `{"success":true,"data":{"status":"reset"}}`, rec.Body.String())
	assert.Equal(t, 0, idx.Len(), "index should be empty after reset")
}
