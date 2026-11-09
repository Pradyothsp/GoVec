package index

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

func TestValidateVectorDims(t *testing.T) {
	tests := []struct {
		name       string
		vec        []float32
		dimensions int
		wantErr    bool
	}{
		{name: "matching_width", vec: []float32{1, 2, 3}, dimensions: 3},
		{name: "unset_width_accepts_any", vec: []float32{1, 2, 3}, dimensions: 0},
		{name: "too_narrow", vec: []float32{1, 2}, dimensions: 3, wantErr: true},
		{name: "too_wide", vec: []float32{1, 2, 3, 4}, dimensions: 3, wantErr: true},
		{name: "empty_against_set_width", vec: []float32{}, dimensions: 3, wantErr: true},
		{name: "empty_against_unset_width", vec: []float32{}, dimensions: 0, wantErr: true},
		{name: "nil_against_unset_width", vec: nil, dimensions: 0, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateVectorDims(tt.vec, tt.dimensions)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
		})
	}
}

// The transports branch on these sentinels to answer 400 / InvalidArgument
// instead of 500, so the wrapping has to survive fmt.Errorf.
func TestValidateVectorDims_ErrorsAreClassifiable(t *testing.T) {
	mismatch := validateVectorDims([]float32{1, 2}, 3)
	require.Error(t, mismatch)
	assert.ErrorIs(t, mismatch, ErrDimensionMismatch)
	assert.True(t, IsInvalidVectorError(mismatch))

	empty := validateVectorDims(nil, 3)
	require.Error(t, empty)
	assert.ErrorIs(t, empty, ErrEmptyVector)
	assert.True(t, IsInvalidVectorError(empty))

	assert.False(t, IsInvalidVectorError(nil), "no error is not a client error")
	assert.False(t, IsInvalidVectorError(errors.New("disk on fire")),
		"an unrelated failure must still read as a server fault")
}

func TestValidateFinite(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	tests := []struct {
		name    string
		vec     []float32
		sparse  core.SparseVector
		meta    map[string]any
		wantErr string
	}{
		{name: "all_finite", vec: []float32{1, -2}, sparse: core.SparseVector{Indices: []uint32{1}, Values: []float32{0.5}},
			meta: map[string]any{"year": 2024.0, "tags": []any{"a", 1.0}, "range": map[string]any{"$gt": 1.0}}},
		{name: "nan_in_vector", vec: []float32{1, nan}, wantErr: "vector[1]"},
		{name: "negative_infinity_in_vector", vec: []float32{float32(math.Inf(-1))}, wantErr: "vector[0]"},
		{name: "infinity_in_sparse_value", vec: []float32{1}, sparse: core.SparseVector{Indices: []uint32{7}, Values: []float32{inf}}, wantErr: "sparse_vector value 0"},
		{name: "nan_in_metadata", vec: []float32{1}, meta: map[string]any{"score": math.NaN()}, wantErr: `"score"`},
		{name: "infinity_in_a_list", vec: []float32{1}, meta: map[string]any{"tags": []any{"a", math.Inf(1)}}, wantErr: `"tags"`},
		{name: "nan_in_a_filter_operator", vec: []float32{1}, meta: map[string]any{"year": map[string]any{"$gt": math.NaN()}}, wantErr: `"year"`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			err := validateFinite(tt.vec, tt.sparse, tt.meta)

			// Assert
			if tt.wantErr == "" {
				assert.NoError(t, err)
				return
			}
			require.ErrorIs(t, err, ErrNonFinite)
			assert.True(t, IsInvalidVectorError(err), "transports must answer 400 / InvalidArgument")
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// A NaN used to be stored and then score NaN against every query; in metadata
// it failed the WAL write instead, as a server fault. Both are now refused
// before the WAL, on insert, per batch item, and on search.
func TestNonFinite_IsRejectedOnEveryWriteAndQuery(t *testing.T) {
	ctx := context.Background()
	nan := float32(math.NaN())
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			require.NoError(t, idx.Insert(ctx, "good", []float32{1, 2, 3}, core.SparseVector{}, nil))

			// Act
			insertErr := idx.Insert(ctx, "nan", []float32{1, nan, 3}, core.SparseVector{}, nil)
			metaErr := idx.Insert(ctx, "meta", []float32{1, 2, 3}, core.SparseVector{}, map[string]any{"score": math.NaN()})
			failures, batchErr := idx.BatchInsert(ctx, []BatchInsertItem{
				{ID: "a", Vector: []float32{3, 2, 1}},
				{ID: "inf", Vector: []float32{float32(math.Inf(1)), 2, 3}},
			})
			_, searchErr := idx.Search(ctx, []float32{nan, 2, 3}, core.SparseVector{}, 1, nil)
			_, filterErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 1, map[string]any{"score": math.Inf(-1)})

			// Assert
			assert.ErrorIs(t, insertErr, ErrNonFinite)
			assert.ErrorIs(t, metaErr, ErrNonFinite)
			require.NoError(t, batchErr, "one bad item must not abort the whole batch")
			require.Len(t, failures, 1)
			assert.Equal(t, "inf", failures[0].ID)
			assert.ErrorIs(t, failures[0].Err, ErrNonFinite)
			assert.ErrorIs(t, searchErr, ErrNonFinite)
			assert.ErrorIs(t, filterErr, ErrNonFinite)
			assert.Len(t, r.nodes, 2, "only good and a are stored")
		})
	}
}

// bothEngines opens each engine over a float32 cosine index, along with the
// records it keeps, for tests that hold the two to the same rule.
var bothEngines = []struct {
	name string
	open func(t *testing.T) (FileEngine, *records[[]float32])
}{
	{"brute", func(t *testing.T) (FileEngine, *records[[]float32]) {
		idx := newTestIndex(t)
		return idx, &idx.records
	}},
	{"hnsw", func(t *testing.T) (FileEngine, *records[[]float32]) {
		idx := newTestHNSWIndex(t)
		return idx, &idx.records
	}},
}

// TestInsert_DimensionMismatch_DoesNotCorruptIndex is the regression test for
// the bug this guard exists for: a mis-sized vector used to insert fine and
// then make every later search fail, because nothing re-checked widths at
// query time.
func TestInsert_DimensionMismatch_DoesNotCorruptIndex(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			require.NoError(t, idx.Insert(ctx, "good", []float32{1, 2, 3}, core.SparseVector{}, nil))

			// Act
			err := idx.Insert(ctx, "bad", []float32{1, 2}, core.SparseVector{}, nil)

			// Assert
			require.Error(t, err, "a 2-d vector must not enter a 3-d index")
			assert.Len(t, r.nodes, 1, "the rejected vector must not be stored")
			_, idErr := r.ids.ToUint32ID("bad")
			assert.Error(t, idErr, "a rejected insert must not leave an id mapping behind")

			// The whole point: search still works afterwards.
			results, searchErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 1, nil)
			require.NoError(t, searchErr, "search must survive a rejected insert")
			require.Len(t, results, 1)
			assert.Equal(t, "good", results[0].ID)
		})
	}
}

func TestBatchInsert_DimensionMismatch_RejectsOnlyTheBadItem(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			items := []BatchInsertItem{
				{ID: "a", Vector: []float32{1, 2, 3}},
				{ID: "bad", Vector: []float32{9, 9}},
				{ID: "b", Vector: []float32{4, 5, 6}},
			}

			// Act
			failures, err := idx.BatchInsert(ctx, items)

			// Assert
			require.NoError(t, err, "one bad item must not abort the whole batch")
			require.Len(t, failures, 1)
			assert.Equal(t, "bad", failures[0].ID)
			assert.Len(t, r.nodes, 2, "the two well-formed items must still be stored")

			results, searchErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 2, nil)
			require.NoError(t, searchErr, "search must survive a partially-rejected batch")
			assert.Len(t, results, 2)
		})
	}
}

// Search had the same defect Insert did: a wrong-width query reached the
// similarity loop and surfaced a bare "vector dimensions mismatch" that named
// neither the width supplied nor the width required. HNSW short-circuits on
// an empty graph, so the check must come first there too, or the two engines
// disagree on the same bad query.
func TestSearch_DimensionMismatch_IsRejectedWithTheRequiredWidth(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange: one populated 3-d index, one empty one configured for 3.
			idx, _ := e.open(t)
			require.NoError(t, idx.Insert(ctx, "good", []float32{1, 2, 3}, core.SparseVector{}, nil))
			empty, r := e.open(t)
			r.dimensions = 3

			// Act
			_, err := idx.Search(ctx, []float32{1, 2}, core.SparseVector{}, 1, nil)
			_, emptyErr := empty.Search(ctx, []float32{1, 2}, core.SparseVector{}, 1, nil)

			// Assert
			require.ErrorIs(t, err, ErrDimensionMismatch)
			assert.Contains(t, err.Error(), "index requires 3")
			assert.ErrorIs(t, emptyErr, ErrDimensionMismatch,
				"an empty index must not swallow a query it could never have matched")
		})
	}
}

// A query against an index with no width yet is unconstrained -- there is
// nothing to disagree with -- and must not start erroring.
func TestSearch_IndexWithNoWidthYet_AcceptsAnyQuery(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)

			// Act
			results, err := idx.Search(context.Background(), []float32{1, 2, 3, 4, 5}, core.SparseVector{}, 1, nil)

			// Assert
			require.NoError(t, err)
			assert.Empty(t, results)
		})
	}
}

// Reset wipes the index; a width it merely learned from the first insert has
// nothing left to be consistent with. Keeping it left an *empty* index
// rejecting every vector of a different size, so switching embedding models
// after a reset needed a server restart.
func TestReset_LearnedWidth_IsReleased(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			require.NoError(t, idx.Insert(ctx, "three", []float32{1, 2, 3}, core.SparseVector{}, nil))
			require.Equal(t, 3, idx.Info().Dimensions)

			// Act
			require.NoError(t, idx.Reset(ctx))

			// Assert
			assert.Equal(t, 0, idx.Info().Dimensions, "a reset index reports no width")
			assert.NoError(t, idx.Insert(ctx, "five", []float32{1, 2, 3, 4, 5}, core.SparseVector{}, nil),
				"an empty index must accept a vector of any width")
		})
	}
}

// A width the operator set in config is not the index's to forget -- mmap
// sizing depends on it, and it is the whole point of setting it.
func TestReset_ConfiguredWidth_Survives(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			r.dimensions = 3
			r.configuredDimensions = 3

			// Act
			require.NoError(t, idx.Reset(ctx))

			// Assert
			assert.Equal(t, 3, idx.Info().Dimensions)
			assert.Error(t, idx.Insert(ctx, "five", []float32{1, 2, 3, 4, 5}, core.SparseVector{}, nil),
				"a configured width still binds after a reset")
		})
	}
}
