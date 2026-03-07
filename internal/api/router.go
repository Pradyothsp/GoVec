package api

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	"github.com/Pradyothsp/govec/internal/index"

	_ "github.com/Pradyothsp/govec/build/swagger" // generated spec; registers via init()
	"github.com/Pradyothsp/govec/internal/api/handlers"
	"github.com/Pradyothsp/govec/internal/api/middleware"
	"github.com/Pradyothsp/govec/internal/api/response"
)

// SetupRouter configures the Gin engine with all routes and injects dependencies.
// If apiKey is non-empty, all /api/v1/* routes require a matching Bearer token.
// dataPath is the persistence path used by the flush endpoint.
func SetupRouter(engine index.Engine, apiKey, dataPath string) *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(middleware.CorrelationID())
	r.Use(middleware.RequestLogger())

	r.GET("/health", healthCheck)
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Initialize handlers, injecting dependencies
	vecHandler := handlers.NewVectorHandler(engine)
	sysHandler := handlers.NewSystemHandler(engine, dataPath)

	v1 := r.Group("api/v1")
	v1.Use(middleware.BearerAuth(apiKey))
	v1.Use(middleware.Timeout(30 * time.Second))

	v1.POST("/vectors", vecHandler.Insert)
	v1.POST("/vectors/batch", vecHandler.BatchInsert)
	v1.GET("/vectors/:id", vecHandler.GetByID)
	v1.POST("/vectors/search", vecHandler.Search)
	v1.DELETE("/vectors/:id", vecHandler.Delete)

	v1.GET("/stats", sysHandler.Stats)
	v1.GET("/info", sysHandler.Info)
	v1.POST("/admin/flush", sysHandler.Flush)

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
