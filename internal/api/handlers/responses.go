package handlers

import (
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// VectorResponse is the JSON response for fetching one vector by ID.
//
// sparse_vector and metadata are omitted when the vector has none, rather
// than sent as null -- an absent field means absent, and callers can test for
// the key.
//
// SparseVector is a pointer for exactly that reason: encoding/json honours
// omitempty for pointers, maps and slices but silently ignores it on a struct
// value, which this field used to be. Every dense-only record therefore went
// out as {"sparse_vector":{"indices":null,"values":null}}, claiming a sparse
// component that was not there.
type VectorResponse struct {
	ID           string             `json:"id" example:"vec-001"`
	Vector       []float32          `json:"vector" swaggertype:"array,number"`
	SparseVector *core.SparseVector `json:"sparse_vector,omitempty" swaggertype:"object"`
	Metadata     map[string]any     `json:"metadata,omitempty" swaggertype:"object"`
}

// SearchResultResponse is one match in the JSON response for a search.
//
// meta is omitted when the vector carries no metadata, matching
// VectorResponse.Metadata. It previously had no omitempty at all, so every
// result in every response ended with "meta":null.
type SearchResultResponse struct {
	ID    string         `json:"id" example:"vec-001"`
	Score float32        `json:"score" example:"0.97"`
	Meta  map[string]any `json:"meta,omitempty" swaggertype:"object"`
}

// toVectorResponse is record as the JSON response for fetching it.
func toVectorResponse(record *index.VectorRecord) VectorResponse {
	return VectorResponse{
		ID:           record.ID,
		Vector:       record.Vector,
		SparseVector: sparseOrNil(record.Sparse),
		Metadata:     record.Metadata,
	}
}

// sparseOrNil returns a pointer to sv, or nil when sv holds nothing, so that a
// dense-only VectorResponse omits sparse_vector entirely.
func sparseOrNil(sv core.SparseVector) *core.SparseVector {
	if sv.IsEmpty() {
		return nil
	}

	return &sv
}

// toSearchResults is results as the JSON response for a search: an empty
// list, never null, when nothing matched.
func toSearchResults(results []index.SearchResult) []SearchResultResponse {
	out := make([]SearchResultResponse, 0, len(results))
	for _, r := range results {
		out = append(out, SearchResultResponse{ID: r.ID, Score: r.Score, Meta: r.Meta})
	}

	return out
}
