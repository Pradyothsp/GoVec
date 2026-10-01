package index

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// These pin the serialised shape of the two types that cross the REST boundary.
//
// The bug they guard against was invisible to every other test in the repo:
// VectorRecord.SparseVector carried `omitempty` while being a struct, which
// encoding/json ignores, so a dense-only record claimed to have a sparse
// component whose contents were null. Nothing failed, because nothing asserted
// on the JSON text -- the Go-level round trip was fine, and only a client
// reading the key noticed. Asserting on the marshalled bytes is the only way
// this class of mistake shows up.

func TestVectorRecord_DenseOnly_OmitsTheSparseField(t *testing.T) {
	rec := &VectorRecord{ID: "a", Vector: []float32{1, 2}}

	data, err := json.Marshal(rec)
	require.NoError(t, err)

	assert.JSONEq(t, `{"id":"a","vector":[1,2]}`, string(data))
	assert.NotContains(t, string(data), "sparse_vector",
		"a record with no sparse component must not mention one")
}

func TestVectorRecord_WithSparse_KeepsTheSparseField(t *testing.T) {
	rec := &VectorRecord{
		ID:           "a",
		Vector:       []float32{1, 2},
		SparseVector: sparseOrNil(core.SparseVector{Indices: []uint32{0, 7}, Values: []float32{0.5, 0.25}}),
	}

	data, err := json.Marshal(rec)
	require.NoError(t, err)

	assert.JSONEq(t,
		`{"id":"a","vector":[1,2],"sparse_vector":{"indices":[0,7],"values":[0.5,0.25]}}`,
		string(data))
}

func TestSearchResult_NoMetadata_OmitsTheMetaField(t *testing.T) {
	data, err := json.Marshal(SearchResult{ID: "a", Score: 0.9})
	require.NoError(t, err)

	assert.JSONEq(t, `{"id":"a","score":0.9}`, string(data))
	assert.NotContains(t, string(data), "meta")
}

func TestSearchResult_WithMetadata_KeepsTheMetaField(t *testing.T) {
	data, err := json.Marshal(SearchResult{ID: "a", Score: 0.9, Meta: map[string]any{"k": "v"}})
	require.NoError(t, err)

	assert.JSONEq(t, `{"id":"a","score":0.9,"meta":{"k":"v"}}`, string(data))
}

func TestSparseOrNil(t *testing.T) {
	assert.Nil(t, sparseOrNil(core.SparseVector{}), "an empty vector is an absent one")
	assert.Nil(t, sparseOrNil(core.SparseVector{Indices: []uint32{}, Values: []float32{}}),
		"non-nil but empty slices are still nothing to report")

	sv := core.SparseVector{Indices: []uint32{3}, Values: []float32{1.5}}
	got := sparseOrNil(sv)
	require.NotNil(t, got)
	assert.Equal(t, sv, *got)
}

// The pointer must address a copy. Handing back &node.Sparse would let a caller
// mutating the record it received reach into the index's stored state.
func TestSparseOrNil_PointsAtACopy(t *testing.T) {
	stored := core.SparseVector{Indices: []uint32{3}, Values: []float32{1.5}}

	got := sparseOrNil(stored)
	require.NotNil(t, got)
	got.Indices = []uint32{99}

	assert.Equal(t, []uint32{3}, stored.Indices, "the stored vector must be untouched")
}
