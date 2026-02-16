package api

import (
	"github.com/Pradyothsp/govec/internal/api/handlers"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/gin-gonic/gin"
)

func SetupRouter(index *core.VectorIndex) *gin.Engine {
	r := gin.Default()

	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	// Initialize handlers, injecting dependencies
	vecHandler := handlers.NewVectorHandler(index)

	v1 := r.Group("api/v1")
	{
		v1.POST("/vectors", vecHandler.Insert)
		v1.POST("/query", vecHandler.Search)
	}

	return r
}
