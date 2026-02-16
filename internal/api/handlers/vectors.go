package handlers

import (
	"net/http"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/gin-gonic/gin"
)

type CreateVectorRequest struct {
	ID       string                 `json:"id" binding:"required"`
	Vector   []float32              `json:"vector" binding:"required"`
	Metadata map[string]interface{} `json:"metadata"`
}

type SearchRequest struct {
	Vector []float32 `json:"vector" binding:"required"`
	K      int       `json:"k"`
}

// VectorHandler holds a reference to the core logic
type VectorHandler struct {
	Index *core.VectorIndex
}

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

	results, err := h.Index.Search(req.Vector, req.K)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, results)
}
