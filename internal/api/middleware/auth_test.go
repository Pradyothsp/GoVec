package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestBearerAuth(t *testing.T) {
	tests := []struct {
		name   string
		apiKey string // "" disables auth
		header string // Authorization; "" sends none
		wantOK bool
	}{
		{"disabled, no header", "", "", true},
		{"disabled, any header", "", "Bearer whatever", true},
		{"correct token", "secret", "Bearer secret", true},
		{"missing header", "secret", "", false},
		{"wrong token", "secret", "Bearer wrongtoken", false},
		{"wrong scheme", "secret", "Token secret", false},
		{"empty bearer token", "secret", "Bearer ", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			called := false
			router := gin.New()
			router.Use(BearerAuth(tt.apiKey))
			router.GET("/protected", func(c *gin.Context) {
				called = true
				c.Status(http.StatusOK)
			})

			req := httptest.NewRequest(http.MethodGet, "/protected", nil)
			if tt.header != "" {
				req.Header.Set("Authorization", tt.header)
			}
			w := httptest.NewRecorder()

			// Act
			router.ServeHTTP(w, req)

			// Assert
			if tt.wantOK {
				assert.Equal(t, http.StatusOK, w.Code)
				assert.True(t, called)
				return
			}
			assert.Equal(t, http.StatusUnauthorized, w.Code)
			assert.JSONEq(t, `{"success":false,"data":null,"error":"unauthorized"}`, w.Body.String())
			assert.False(t, called, "the handler must not run after a 401")
		})
	}
}
