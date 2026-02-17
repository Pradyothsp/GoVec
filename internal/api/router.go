package api

import (
	"github.com/gin-gonic/gin"

	"github.com/Pradyothsp/govec/internal/api/handlers"
	"github.com/Pradyothsp/govec/internal/core"
)

// SetupRouter configures the Gin engine with all routes and injects dependencies.
func SetupRouter(index *core.VectorIndex) *gin.Engine {
	r := gin.Default()

	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	// Initialize handlers, injecting dependencies
	vecHandler := handlers.NewVectorHandler(index)

	v1 := r.Group("api/v1")
	v1.POST("/vectors", vecHandler.Insert)
	v1.POST("/query", vecHandler.Search)

	return r
}
