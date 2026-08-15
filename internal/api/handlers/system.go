package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"

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
	if err := h.Engine.SaveToFile(c.Request.Context(), h.DataPath); err != nil {
		zerolog.Ctx(c.Request.Context()).Error().Err(err).Str("path", h.DataPath).Msg("failed to flush index to disk")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}
	response.OK(c, http.StatusOK, gin.H{"status": "flushed"})
}

// Reset handles POST /api/v1/admin/reset
//
// Reset wipes all vectors, ID mappings, and the WAL in memory. It does not
// persist the cleared state to disk -- call Flush afterward if the reset
// should survive a restart.
//
// @Summary      Clear all vectors from the index
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=object}
// @Failure      401  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/admin/reset [post]
func (h *SystemHandler) Reset(c *gin.Context) {
	h.Engine.Clear()
	response.OK(c, http.StatusOK, gin.H{"status": "reset"})
}
