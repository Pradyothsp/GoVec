package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/Pradyothsp/govec/internal/api/response"
	"github.com/Pradyothsp/govec/internal/core"
)

// CreateVectorRequest is the JSON payload for inserting a vector.
type CreateVectorRequest struct {
	ID       string                 `json:"id" binding:"required" example:"vec-001"`
	Vector   []float32              `json:"vector" binding:"required" swaggertype:"array,number"`
	Metadata map[string]interface{} `json:"metadata" swaggertype:"object"`
}

// SearchRequest is the JSON payload for a nearest-neighbour query.
type SearchRequest struct {
	Vector  []float32              `json:"vector" binding:"required" swaggertype:"array,number"`
	K       int                    `json:"k" example:"10"`
	Filters map[string]interface{} `json:"filter" swaggertype:"object"`
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
	Index *core.VectorIndex
}

// NewVectorHandler creates a VectorHandler with the given index dependency.
func NewVectorHandler(index *core.VectorIndex) *VectorHandler {
	return &VectorHandler{Index: index}
}

// Insert handles POST /api/v1/vectors
//
// @Summary      Insert or update a vector
// @Tags         vectors
// @Accept       json
// @Produce      json
// @Param        body  body      CreateVectorRequest  true  "Vector payload"
// @Success      201   {object}  response.Response{data=handlers.InsertResponse}
// @Failure      400   {object}  response.Response
// @Router       /api/v1/vectors [post]
func (h *VectorHandler) Insert(c *gin.Context) {
	var req CreateVectorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, err.Error())
		return
	}

	h.Index.Insert(req.ID, req.Vector, req.Metadata)

	response.OK(c, http.StatusCreated, InsertResponse{Status: "inserted"})
}

// Search handles POST /api/v1/vectors/search
//
// @Summary      Search for similar vectors
// @Tags         vectors
// @Accept       json
// @Produce      json
// @Param        body  body      SearchRequest     true  "Search payload"
// @Success      200   {object}  response.Response{data=[]core.SearchResult}
// @Failure      400   {object}  response.Response
// @Failure      500   {object}  response.Response
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

	results, err := h.Index.Search(req.Vector, req.K, req.Filters)
	if err != nil {
		response.Fail(c, http.StatusInternalServerError, err.Error())
		return
	}

	if results == nil {
		results = []core.SearchResult{}
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
// @Failure      400  {object}  response.Response
// @Failure      404  {object}  response.Response
// @Router       /api/v1/vectors/{id} [delete]
func (h *VectorHandler) Delete(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		response.Fail(c, http.StatusBadRequest, "id is required")
		return
	}

	success := h.Index.Delete(id)
	if !success {
		response.Fail(c, http.StatusNotFound, "vector not found")
		return
	}

	response.OK(c, http.StatusOK, DeleteResponse{Status: "deleted", ID: id})
}
