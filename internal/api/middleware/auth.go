package middleware

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/api/response"
)

// BearerAuth returns middleware enforcing Bearer token auth.
// If apiKey is empty, all requests pass through (auth disabled).
func BearerAuth(apiKey string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if apiKey == "" {
			c.Next()
			return
		}

		token, found := strings.CutPrefix(c.GetHeader("Authorization"), "Bearer ")

		// Secure Constant-Time Comparison
		// subtle.ConstantTimeCompare returns 1 if they match, 0 if they don't.
		if !found || subtle.ConstantTimeCompare([]byte(token), []byte(apiKey)) != 1 {
			log.Warn().
				Str("ip", c.ClientIP()).
				Str("method", c.Request.Method).
				Str("path", c.Request.URL.Path).
				Msg("unauthorized access attempt")
			response.Fail(c, http.StatusUnauthorized, "unauthorized")
			c.Abort()
			return
		}

		c.Next()
	}
}
