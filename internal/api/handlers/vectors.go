package handlers

import (
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"

	"github.com/Pradyothsp/govec/internal/api/response"
)

// CreateVectorRequest is the JSON payload for inserting a vector.
type CreateVectorRequest struct {
	ID           string                 `json:"id" binding:"required" example:"vec-001"`
	Vector       []float32              `json:"vector" binding:"required" swaggertype:"array,number"`
	SparseVector *core.SparseVector     `json:"sparse_vector,omitempty" swaggertype:"object"`
	Metadata     map[string]interface{} `json:"metadata" swaggertype:"object"`
}

// SearchRequest is the JSON payload for a nearest-neighbour query.
type SearchRequest struct {
	Vector       []float32              `json:"vector" binding:"required" swaggertype:"array,number"`
	SparseVector *core.SparseVector     `json:"sparse_vector,omitempty" swaggertype:"object"`
	K            *int                   `json:"k" binding:"required" example:"10"` // required; 0 returns an empty list
	Filters      map[string]interface{} `json:"filter" swaggertype:"object"`
}

// InsertResponse is the JSON response for a successful vector insert.
type InsertResponse struct {
	Status string `json:"status" example:"inserted"`
}

// DeleteResponse is the JSON response for a successful vector deletion.
type DeleteResponse struct {
	Status string `json:"status" example:"deleted"`
	ID     string `json:"id" example:"vec-001"`
}

// VectorHandler holds a reference to the core logic
type VectorHandler struct {
	Engine index.Engine
}

// NewVectorHandler creates a VectorHandler with the given engine dependency.
func NewVectorHandler(engine index.Engine) *VectorHandler {
	return &VectorHandler{Engine: engine}
}

// BatchInsertRequest is the JSON payload for inserting multiple vectors in one request.
type BatchInsertRequest struct {
	Vectors []CreateVectorRequest `json:"vectors" binding:"required"`
}

// BatchInsertResult captures the outcome for a single vector in a batch.
type BatchInsertResult struct {
	ID    string `json:"id"`
	Error string `json:"error,omitempty"`
}

// BatchInsertResponse summarises the result of a batch insert.
type BatchInsertResponse struct {
	InsertedCount int                 `json:"inserted_count"`
	Errors        []BatchInsertResult `json:"errors,omitempty"`
}

// BatchInsert handles POST /api/v1/vectors/batch
//
// @Summary      Batch insert vectors
// @Tags         vectors
// @Accept       json
// @Produce      json
// @Param        body  body      BatchInsertRequest   true  "Batch vector payload"
// @Success      200   {object}  response.Response{data=handlers.BatchInsertResponse}
// @Failure      400   {object}  response.ErrorResponse
// @Failure      401   {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/vectors/batch [post]
func (h *VectorHandler) BatchInsert(c *gin.Context) {
	var req BatchInsertRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, err.Error())
		return
	}

	if len(req.Vectors) == 0 {
		response.Fail(c, http.StatusBadRequest, "vectors array must not be empty")
		return
	}

	var errs []BatchInsertResult
	items := make([]index.BatchInsertItem, 0, len(req.Vectors))

	for _, v := range req.Vectors {
		items = append(items, index.BatchInsertItem{
			ID:     v.ID,
			Vector: v.Vector,
			Sparse: sparseOrNone(v.SparseVector),
			Meta:   v.Metadata,
		})
	}

	inserted := 0
	if len(items) > 0 {
		failures, err := h.Engine.BatchInsert(c.Request.Context(), items)
		if err != nil {
			response.Fail(c, http.StatusInternalServerError, err.Error())
			return
		}

		for _, f := range failures {
			errs = append(errs, BatchInsertResult{ID: f.ID, Error: f.Err.Error()})
		}
		inserted = len(items) - len(failures)
	}

	response.OK(c, http.StatusOK, BatchInsertResponse{
		InsertedCount: inserted,
		Errors:        errs,
	})
}

// Insert handles POST /api/v1/vectors
//
// @Summary      Insert or update a vector
// @Tags         vectors
// @Accept       json
// @Produce      json
// @Param        body  body      CreateVectorRequest  true  "Vector payload"
// @Success      201   {object}  response.Response{data=handlers.InsertResponse}
// @Failure      400   {object}  response.ErrorResponse
// @Failure      401   {object}  response.ErrorResponse
// @Failure      500   {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/vectors [post]
func (h *VectorHandler) Insert(c *gin.Context) {
	var req CreateVectorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, err.Error())
		return
	}

	// Validate sparse vector if provided
	err := h.Engine.Insert(c.Request.Context(), req.ID, req.Vector, sparseOrNone(req.SparseVector), req.Metadata)
	if err != nil {
		// A mis-sized or empty vector is the caller's mistake, not a server
		// fault -- report it as such, and don't log it at error level alongside
		// real failures.
		if index.IsInvalidVectorError(err) {
			response.Fail(c, http.StatusBadRequest, err.Error())
			return
		}

		zerolog.Ctx(c.Request.Context()).Error().Err(err).Str("id", req.ID).Msg("failed to insert vector")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	response.OK(c, http.StatusCreated, InsertResponse{Status: "inserted"})
}

// Search handles POST /api/v1/vectors/search
//
// @Summary      Search for similar vectors
// @Tags         vectors
// @Accept       json
// @Produce      json
// @Param        body  body      SearchRequest     true  "Search payload"
// @Success      200   {object}  response.Response{data=[]index.SearchResult}
// @Failure      400   {object}  response.ErrorResponse
// @Failure      401   {object}  response.ErrorResponse
// @Failure      500   {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/vectors/search [post]
func (h *VectorHandler) Search(c *gin.Context) {
	const maxSearchK = 10000

	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, err.Error())
		return
	}

	if *req.K < 0 {
		response.Fail(c, http.StatusBadRequest, "k cannot be negative")
		return
	}
	if *req.K > maxSearchK {
		response.Fail(c, http.StatusBadRequest, "k exceeds maximum allowed value")
		return
	}

	results, err := h.Engine.Search(c.Request.Context(), req.Vector, sparseOrNone(req.SparseVector), *req.K, req.Filters)
	if err != nil {
		// Same rule as Insert: a query the index can't compare is a bad
		// request, not a server fault. It matters more here -- search is the
		// call clients make in a loop.
		if index.IsInvalidVectorError(err) {
			response.Fail(c, http.StatusBadRequest, err.Error())
			return
		}

		zerolog.Ctx(c.Request.Context()).Error().Err(err).Int("k", *req.K).Msg("failed to search vectors")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	if results == nil {
		results = []index.SearchResult{}
	}

	response.OK(c, http.StatusOK, results)
}

// GetByID handles GET /api/v1/vectors/:id
//
// @Summary      Get a vector by ID
// @Tags         vectors
// @Produce      json
// @Param        id   path      string            true  "Vector ID"
// @Success      200  {object}  response.Response{data=index.VectorRecord}
// @Failure      400  {object}  response.ErrorResponse
// @Failure      401  {object}  response.ErrorResponse
// @Failure      404  {object}  response.ErrorResponse
// @Failure      500  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/vectors/{id} [get]
func (h *VectorHandler) GetByID(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		response.Fail(c, http.StatusBadRequest, "id is required")
		return
	}

	record, err := h.Engine.GetByID(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, core.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "vector not found")
			return
		}
		zerolog.Ctx(c.Request.Context()).Error().Err(err).Str("id", id).Msg("failed to get vector by ID")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	response.OK(c, http.StatusOK, record)
}

// Delete handles DELETE /api/v1/vectors/:id
//
// @Summary      Delete a vector by ID
// @Tags         vectors
// @Produce      json
// @Param        id   path      string            true  "Vector ID"
// @Success      200  {object}  response.Response{data=handlers.DeleteResponse}
// @Failure      400  {object}  response.ErrorResponse
// @Failure      401  {object}  response.ErrorResponse
// @Failure      404  {object}  response.ErrorResponse
// @Failure      500  {object}  response.ErrorResponse
// @Security     BearerAuth
// @Router       /api/v1/vectors/{id} [delete]
func (h *VectorHandler) Delete(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		response.Fail(c, http.StatusBadRequest, "id is required")
		return
	}

	success, err := h.Engine.Delete(c.Request.Context(), id)
	if err != nil {
		zerolog.Ctx(c.Request.Context()).Error().Err(err).Str("id", id).Msg("failed to delete vector")
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	if !success {
		response.Fail(c, http.StatusNotFound, "vector not found")
		return
	}

	response.OK(c, http.StatusOK, DeleteResponse{Status: "deleted", ID: id})
}

// sparseOrNone turns an absent sparse_vector into an empty one, which the
// engine reads as none. The engine checks a present one.
func sparseOrNone(sparse *core.SparseVector) core.SparseVector {
	if sparse == nil {
		return core.SparseVector{}
	}
	return *sparse
}
