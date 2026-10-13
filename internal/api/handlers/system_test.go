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
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

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
			h := NewSystemHandler(idx)
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

// Flush writes the snapshot to the engine's own path; a path it can't write
// is a server fault.
func TestFlush(t *testing.T) {
	tests := []struct {
		name     string
		path     func(t *testing.T) string
		wantCode int
		wantBody string
	}{
		{"writes the snapshot", func(t *testing.T) string { return filepath.Join(t.TempDir(), "test.bin") },
			http.StatusOK, `{"success":true,"data":{"status":"flushed"}}`},
		{"unwritable path", func(t *testing.T) string {
			// A path under a regular file can't be created by anyone, root included.
			notADir := filepath.Join(t.TempDir(), "a-file")
			require.NoError(t, os.WriteFile(notADir, nil, 0o600))
			return filepath.Join(notADir, "test.bin")
		}, http.StatusInternalServerError, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			path := tt.path(t)
			idx := testutil.NewTestIndexAt(t, path)
			require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 2}, core.SparseVector{}, nil))
			router := gin.New()
			router.POST("/admin/flush", NewSystemHandler(idx).Flush)
			rec := httptest.NewRecorder()

			// Act
			router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/admin/flush", nil))

			// Assert
			assert.Equal(t, tt.wantCode, rec.Code)
			if tt.wantBody != "" {
				assert.JSONEq(t, tt.wantBody, rec.Body.String())
				assert.FileExists(t, path)
			}
		})
	}
}
