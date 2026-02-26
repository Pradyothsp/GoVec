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

func newTestRouter(apiKey string) *gin.Engine {
	r := gin.New()
	r.Use(BearerAuth(apiKey))
	r.GET("/protected", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})
	return r
}

func TestBearerAuth_Disabled(t *testing.T) {
	// Empty apiKey disables auth — all requests pass
	router := newTestRouter("")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestBearerAuth_Disabled_NoHeader(t *testing.T) {
	// Even without Authorization header, disabled auth lets requests through
	router := newTestRouter("")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestBearerAuth_MissingHeader(t *testing.T) {
	router := newTestRouter("secret")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.JSONEq(t, `{"success":false,"data":null,"error":"unauthorized"}`, w.Body.String())
}

func TestBearerAuth_WrongToken(t *testing.T) {
	router := newTestRouter("secret")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Bearer wrongtoken")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.JSONEq(t, `{"success":false,"data":null,"error":"unauthorized"}`, w.Body.String())
}

func TestBearerAuth_WrongScheme(t *testing.T) {
	// "Token secret" instead of "Bearer secret"
	router := newTestRouter("secret")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Token secret")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestBearerAuth_CorrectToken(t *testing.T) {
	router := newTestRouter("secret")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Bearer secret")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestBearerAuth_EmptyBearerToken(t *testing.T) {
	// "Bearer " with no actual token value
	router := newTestRouter("secret")

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Bearer ")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestBearerAuth_AbortsProperly(t *testing.T) {
	// Verify that downstream handlers do NOT execute after a 401
	called := false
	r := gin.New()
	r.Use(BearerAuth("secret"))
	r.GET("/protected", func(c *gin.Context) {
		called = true
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.False(t, called, "downstream handler must not run after abort")
}
