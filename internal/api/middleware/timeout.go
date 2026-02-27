package middleware

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/api/response"
)

// Timeout returns middleware that cancels the request context after d.
// Handlers should respect ctx.Err() for cooperative cancellation.
func Timeout(d time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), d)
		defer cancel()

		c.Request = c.Request.WithContext(ctx)
		c.Next()

		if ctx.Err() == context.DeadlineExceeded {
			log.Warn().
				Str("method", c.Request.Method).
				Str("path", c.Request.URL.Path).
				Dur("timeout", d).
				Msg("request timed out")
			response.Fail(c, http.StatusServiceUnavailable, "request timed out")
			c.Abort()
		}
	}
}
