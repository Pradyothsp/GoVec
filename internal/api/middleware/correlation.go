package middleware

import (
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/requestid"
)

const correlationHeader = "X-Correlation-ID"

// CorrelationID extracts or generates a correlation ID for each request,
// attaches it to the request context via a zerolog logger, and echoes it
// back in the response header so callers can trace their request.
func CorrelationID() gin.HandlerFunc {
	return func(c *gin.Context) {
		var correlationID string

		if c.GetHeader(correlationHeader) != "" {
			correlationID = c.GetHeader(correlationHeader)
		} else {
			correlationID = requestid.Generate()
		}

		//  Build a scoped logger
		logger := log.With().Str("correlation_id", correlationID).Logger()

		// Attach to context
		ctx := logger.WithContext(c.Request.Context())

		// Update the request
		c.Request = c.Request.WithContext(ctx)

		// Store for handlers
		c.Set(requestid.GinKey, correlationID)

		// Echo back to a client
		c.Header(correlationHeader, correlationID)

		c.Next()
	}
}
