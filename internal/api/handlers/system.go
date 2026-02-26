package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/Pradyothsp/govec/internal/api/response"
	"github.com/Pradyothsp/govec/internal/index"
)

// SystemHandler handles system-level endpoints (stats, info, admin operations).
type SystemHandler struct {
	Engine   index.Engine
	DataPath string
}

// NewSystemHandler creates a SystemHandler with the given engine and data path.
func NewSystemHandler(engine index.Engine, dataPath string) *SystemHandler {
	return &SystemHandler{Engine: engine, DataPath: dataPath}
}

// StatsResponse is the JSON response for the stats endpoint.
type StatsResponse struct {
	VectorCount int `json:"vector_count" example:"42"`
}

// Stats handles GET /api/v1/stats
//
// @Summary      Get index statistics
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=handlers.StatsResponse}
// @Failure      401  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/stats [get]
func (h *SystemHandler) Stats(c *gin.Context) {
	response.OK(c, http.StatusOK, StatsResponse{VectorCount: h.Engine.Len()})
}

// Info handles GET /api/v1/info
//
// @Summary      Get engine configuration info
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=index.EngineInfo}
// @Failure      401  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/info [get]
func (h *SystemHandler) Info(c *gin.Context) {
	response.OK(c, http.StatusOK, h.Engine.Info())
}

// Flush handles POST /api/v1/admin/flush
//
// @Summary      Flush index to disk
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=object}
// @Failure      401  {object}  response.ErrorResponse
// @Failure      500  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/admin/flush [post]
func (h *SystemHandler) Flush(c *gin.Context) {
	if err := h.Engine.SaveToFile(h.DataPath); err != nil {
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}
	response.OK(c, http.StatusOK, gin.H{"status": "flushed"})
}
