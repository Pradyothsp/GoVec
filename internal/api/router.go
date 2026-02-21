package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	_ "github.com/Pradyothsp/govec/docs" // generated spec; registers via init()
	"github.com/Pradyothsp/govec/internal/api/handlers"
	"github.com/Pradyothsp/govec/internal/api/response"
	"github.com/Pradyothsp/govec/internal/core"
)

// SetupRouter configures the Gin engine with all routes and injects dependencies.
func SetupRouter(engine core.Engine) *gin.Engine {
	r := gin.Default()

	r.GET("/health", healthCheck)
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Initialize handlers, injecting dependencies
	vecHandler := handlers.NewVectorHandler(engine)

	v1 := r.Group("api/v1")
	v1.POST("/vectors", vecHandler.Insert)
	v1.POST("/vectors/search", vecHandler.Search)
	v1.DELETE("/vectors/:id", vecHandler.Delete)

	return r
}

// HealthResponse is the JSON response for the health check endpoint.
type HealthResponse struct {
	Status string `json:"status" example:"ok"`
}

// healthCheck handles GET /health
//
// @Summary      Health check
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=api.HealthResponse}
// @Router       /health [get]
func healthCheck(c *gin.Context) {
	response.OK(c, http.StatusOK, HealthResponse{Status: "ok"})
}
