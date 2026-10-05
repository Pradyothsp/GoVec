package handlers

import (
	"context"
	"encoding/json"
	"fmt"
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

// Stats and Info report the vector count, and Info the width, which an
// empty index doesn't have yet and the first insert sets.
func TestStatsAndInfo_EmptyThenAfterInsert(t *testing.T) {
	tests := []struct {
		name     string
		inserted [][]float32
		count    float64
		dims     float64
	}{
		{"empty index", nil, 0, 0},
		{"after one insert", [][]float32{{1, 2, 3}}, 1, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			idx := testutil.NewTestIndex(t)
			for i, vec := range tt.inserted {
				require.NoError(t, idx.Insert(context.Background(), fmt.Sprintf("v%d", i), vec, core.SparseVector{}, nil))
			}
			h := NewSystemHandler(idx, "")
			router := gin.New()
			router.GET("/stats", h.Stats)
			router.GET("/info", h.Info)
			stats, info := httptest.NewRecorder(), httptest.NewRecorder()

			// Act
			router.ServeHTTP(stats, httptest.NewRequest(http.MethodGet, "/stats", nil))
			router.ServeHTTP(info, httptest.NewRequest(http.MethodGet, "/info", nil))

			// Assert
			assert.Equal(t, http.StatusOK, stats.Code)
			assert.JSONEq(t, fmt.Sprintf(`{"success":true,"data":{"vector_count":%v}}`, tt.count), stats.Body.String())
			assert.Equal(t, http.StatusOK, info.Code)
			var env struct {
				Success bool           `json:"success"`
				Data    map[string]any `json:"data"`
			}
			require.NoError(t, json.Unmarshal(info.Body.Bytes(), &env))
			assert.True(t, env.Success)
			assert.Equal(t, tt.dims, env.Data["dimensions"])
			assert.Equal(t, tt.count, env.Data["vector_count"])
		})
	}
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
