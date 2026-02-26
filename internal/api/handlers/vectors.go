package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"

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
	K            int                    `json:"k" example:"10"`
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
	if req.SparseVector != nil && !req.SparseVector.IsValid() {
		response.Fail(c, http.StatusBadRequest, "invalid sparse_vector: indices and values must have the same length")
		return
	}

	// Convert nil pointer to empty SparseVector for cleaner API
	var sparse core.SparseVector
	if req.SparseVector != nil {
		sparse = *req.SparseVector
	}

	err := h.Engine.Insert(req.ID, req.Vector, sparse, req.Metadata)
	if err != nil {
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
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, err.Error())
		return
	}

	if req.K < 0 {
		response.Fail(c, http.StatusBadRequest, "k cannot be negative")
		return
	}

	// Validate sparse vector if provided
	if req.SparseVector != nil && !req.SparseVector.IsValid() {
		response.Fail(c, http.StatusBadRequest, "invalid sparse_vector: indices and values must have the same length")
		return
	}

	// Convert nil pointer to empty SparseVector for cleaner API
	var sparse core.SparseVector
	if req.SparseVector != nil {
		sparse = *req.SparseVector
	}

	results, err := h.Engine.Search(req.Vector, sparse, req.K, req.Filters)
	if err != nil {
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	if results == nil {
		results = []index.SearchResult{}
	}

	response.OK(c, http.StatusOK, results)
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

	success, err := h.Engine.Delete(id)
	if err != nil {
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	if !success {
		response.Fail(c, http.StatusNotFound, "vector not found")
		return
	}

	response.OK(c, http.StatusOK, DeleteResponse{Status: "deleted", ID: id})
}
