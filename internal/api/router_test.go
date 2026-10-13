package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"

	"github.com/Pradyothsp/govec/internal/test/testutil"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// Every /api/v1 route through this router is exercised in
// internal/test/integration/rest_test.go. These cover what the router adds
// around them: routes outside /api/v1, unknown paths, and where auth applies.
func TestRouter(t *testing.T) {
	tests := []struct {
		name         string
		apiKey       string
		method, path string
		auth         string
		wantCode     int
		wantBody     string
		wantContains string
	}{
		{name: "health", method: http.MethodGet, path: "/health", wantCode: http.StatusOK, wantBody: `{"success":true,"data":{"status":"ok"}}`},
		{name: "swagger UI", method: http.MethodGet, path: "/swagger/index.html", wantCode: http.StatusOK},
		{name: "swagger spec", method: http.MethodGet, path: "/swagger/doc.json", wantCode: http.StatusOK, wantContains: `"swagger": "2.0"`},
		{name: "unknown route", method: http.MethodGet, path: "/api/v1/unknown", wantCode: http.StatusNotFound},
		{name: "root", method: http.MethodGet, path: "/", wantCode: http.StatusNotFound},
		{name: "route without the /api/v1 prefix", method: http.MethodGet, path: "/stats", wantCode: http.StatusNotFound},
		{name: "api key set: /api/v1 needs the token", apiKey: "secret", method: http.MethodGet, path: "/api/v1/stats", wantCode: http.StatusUnauthorized},
		{name: "api key set: the token opens /api/v1", apiKey: "secret", method: http.MethodGet, path: "/api/v1/stats", auth: "Bearer secret", wantCode: http.StatusOK},
		{name: "api key set: health stays open", apiKey: "secret", method: http.MethodGet, path: "/health", wantCode: http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			router := SetupRouter(testutil.NewTestIndex(t), tt.apiKey)
			req := httptest.NewRequest(tt.method, tt.path, nil)
			if tt.auth != "" {
				req.Header.Set("Authorization", tt.auth)
			}
			rec := httptest.NewRecorder()

			// Act
			router.ServeHTTP(rec, req)

			// Assert
			assert.Equal(t, tt.wantCode, rec.Code)
			if tt.wantBody != "" {
				assert.JSONEq(t, tt.wantBody, rec.Body.String())
			}
			assert.Contains(t, rec.Body.String(), tt.wantContains)
		})
	}
}
