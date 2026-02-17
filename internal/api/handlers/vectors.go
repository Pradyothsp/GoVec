package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/Pradyothsp/govec/internal/core"
)

// CreateVectorRequest is the JSON payload for inserting a vector.
type CreateVectorRequest struct {
	ID       string                 `json:"id" binding:"required"`
	Vector   []float32              `json:"vector" binding:"required"`
	Metadata map[string]interface{} `json:"metadata"`
}

// SearchRequest is the JSON payload for a nearest-neighbour query.
type SearchRequest struct {
	Vector  []float32              `json:"vector" binding:"required"`
	K       int                    `json:"k"`
	Filters map[string]interface{} `json:"filter"`
}

// VectorHandler holds a reference to the core logic
type VectorHandler struct {
	Index *core.VectorIndex
}

// NewVectorHandler creates a VectorHandler with the given index dependency.
func NewVectorHandler(index *core.VectorIndex) *VectorHandler {
	return &VectorHandler{Index: index}
}

// Insert handles POST /vectors`
func (h *VectorHandler) Insert(c *gin.Context) {
	var req CreateVectorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.Index.Insert(req.ID, req.Vector, req.Metadata)

	c.JSON(http.StatusCreated, gin.H{"status": "inserted"})
}

// Search handles POST /vectors/search`
func (h *VectorHandler) Search(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.K < 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "k cannot be negative"})
		return
	}

	results, err := h.Index.Search(req.Vector, req.K, req.Filters)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, results)
}

// Delete handles DELETE /vectors/:id
func (h *VectorHandler) Delete(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "id is required"})
		return
	}

	success := h.Index.Delete(id)
	if !success {
		c.JSON(http.StatusNotFound, gin.H{"error": "vector not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "deleted", "id": id})
}
