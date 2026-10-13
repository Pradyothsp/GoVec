package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/api/response"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/release"
)

// SystemHandler handles system-level endpoints (stats, info, admin operations).
type SystemHandler struct {
	Engine index.Engine
}

// NewSystemHandler creates a SystemHandler with the given engine.
func NewSystemHandler(engine index.Engine) *SystemHandler {
	return &SystemHandler{Engine: engine}
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

// InfoResponse is the JSON response for the info endpoint: the engine's configuration plus the
// server release, so a client can ask a running server which version it is.
type InfoResponse struct {
	index.EngineInfo
	Version string `json:"version" example:"v0.1.0"`
}

// Info handles GET /api/v1/info
//
// @Summary      Get engine configuration info and server version
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=handlers.InfoResponse}
// @Failure      401  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/info [get]
func (h *SystemHandler) Info(c *gin.Context) {
	response.OK(c, http.StatusOK, InfoResponse{EngineInfo: h.Engine.Info(), Version: release.Version})
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
	if err := h.Engine.Flush(c.Request.Context()); err != nil {
		zerolog.Ctx(c.Request.Context()).Error().Err(err).Msg("failed to flush index to disk")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}
	response.OK(c, http.StatusOK, gin.H{"status": "flushed"})
}

// Reset handles POST /api/v1/admin/reset
//
// Reset removes every vector durably: the engine writes an empty snapshot and
// truncates the WAL before clearing memory, so a restart comes back empty.
//
// @Summary      Remove all vectors from the index, durably
// @Tags         system
// @Produce      json
// @Success      200  {object}  response.Response{data=object}
// @Failure      401  {object}  response.ErrorResponse
// @Failure      500  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/admin/reset [post]
func (h *SystemHandler) Reset(c *gin.Context) {
	if err := h.Engine.Reset(c.Request.Context()); err != nil {
		zerolog.Ctx(c.Request.Context()).Error().Err(err).Msg("failed to reset index")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}
	response.OK(c, http.StatusOK, gin.H{"status": "reset"})
}
