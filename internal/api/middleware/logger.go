package middleware

import (
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
)

// RequestLogger returns a Gin middleware that emits a structured JSON access log
// for every request, compatible with the zerolog pipeline.
func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()

		status := c.Writer.Status()
		latency := time.Since(start)

		logger := zerolog.Ctx(c.Request.Context())
		event := logger.Info()
		if status >= 500 {
			event = logger.Error()
		} else if status >= 400 {
			event = logger.Warn()
		}

		if len(c.Errors) > 0 {
			event.Interface("errors", c.Errors.Errors())
		}

		event.
			Str("method", c.Request.Method).
			Str("path", c.Request.URL.Path).
			Int("status", status).
			Dur("latency_ms", latency).
			Str("ip", c.ClientIP()).
			Msg("request")
	}
}
