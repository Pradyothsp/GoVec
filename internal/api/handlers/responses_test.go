package handlers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// These pin the serialised shape of the responses built from engine results.
//
// The bug they guard against was invisible to every other test in the repo:
// the record type's sparse field carried `omitempty` while being a struct,
// which encoding/json ignores, so a dense-only record claimed to have a sparse
// component whose contents were null. Nothing failed, because nothing asserted
// on the JSON text -- the Go-level round trip was fine, and only a client
// reading the key noticed. Asserting on the marshalled bytes is the only way
// this class of mistake shows up.

func TestVectorResponse_DenseOnly_OmitsTheSparseField(t *testing.T) {
	// Arrange
	record := &index.VectorRecord{ID: "a", Vector: []float32{1, 2}}

	// Act
	data, err := json.Marshal(toVectorResponse(record))

	// Assert
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"a","vector":[1,2]}`, string(data))
	assert.NotContains(t, string(data), "sparse_vector", "a record with no sparse component must not mention one")
}

func TestVectorResponse_WithSparse_KeepsTheSparseField(t *testing.T) {
	// Arrange
	record := &index.VectorRecord{
		ID:     "a",
		Vector: []float32{1, 2},
		Sparse: core.SparseVector{Indices: []uint32{0, 7}, Values: []float32{0.5, 0.25}},
	}

	// Act
	data, err := json.Marshal(toVectorResponse(record))

	// Assert
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"a","vector":[1,2],"sparse_vector":{"indices":[0,7],"values":[0.5,0.25]}}`, string(data))
}

func TestSearchResults_Meta_OmittedOnlyWhenAbsent(t *testing.T) {
	// Arrange
	results := []index.SearchResult{
		{ID: "a", Score: 0.9},
		{ID: "b", Score: 0.8, Meta: map[string]any{"k": "v"}},
	}

	// Act
	data, err := json.Marshal(toSearchResults(results))

	// Assert
	require.NoError(t, err)
	assert.JSONEq(t, `[{"id":"a","score":0.9},{"id":"b","score":0.8,"meta":{"k":"v"}}]`, string(data))
}

func TestSearchResults_NoMatches_IsAnEmptyList(t *testing.T) {
	// Act
	data, err := json.Marshal(toSearchResults(nil))

	// Assert
	require.NoError(t, err)
	assert.JSONEq(t, `[]`, string(data), "no matches is [], never null")
}

func TestSparseOrNil(t *testing.T) {
	tests := []struct {
		name string
		sv   core.SparseVector
		want *core.SparseVector
	}{
		{"zero value", core.SparseVector{}, nil},
		{"non-nil but empty slices", core.SparseVector{Indices: []uint32{}, Values: []float32{}}, nil},
		{"holds a token", core.SparseVector{Indices: []uint32{3}, Values: []float32{1.5}}, &core.SparseVector{Indices: []uint32{3}, Values: []float32{1.5}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			got := sparseOrNil(tt.sv)

			// Assert
			assert.Equal(t, tt.want, got)
		})
	}
}
