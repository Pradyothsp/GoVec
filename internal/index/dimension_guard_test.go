package index

import (
	"context"
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

// TestInsert_DimensionMismatch_DoesNotCorruptIndex is the regression test for
// the bug this guard exists for: a mis-sized vector used to insert fine and
// then make every later search fail, because nothing re-checked widths at
// query time.
func TestInsert_DimensionMismatch_DoesNotCorruptIndex(t *testing.T) {
	ctx := context.Background()
	idx := newTestIndex(t)

	require.NoError(t, idx.Insert(ctx, "good", []float32{1, 2, 3}, core.SparseVector{}, nil))

	err := idx.Insert(ctx, "bad", []float32{1, 2}, core.SparseVector{}, nil)

	require.Error(t, err, "a 2-d vector must not enter a 3-d index")
	assert.Len(t, idx.Store, 1, "the rejected vector must not be stored")

	_, idErr := idx.IDMapper.ToUint32ID("bad")
	assert.Error(t, idErr, "a rejected insert must not leave an id mapping behind")

	// The whole point: search still works afterwards.
	results, searchErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 1, nil)
	require.NoError(t, searchErr, "search must survive a rejected insert")
	require.Len(t, results, 1)
	assert.Equal(t, "good", results[0].ID)
}

// The HNSW engine tracks dimensions in bookkeepInsert rather than
// insertInternal, so it needs its own coverage -- both Insert and BatchInsert
// route through it.
func TestHNSWInsert_DimensionMismatch_DoesNotCorruptIndex(t *testing.T) {
	ctx := context.Background()
	idx := newTestHNSWIndex(t)

	require.NoError(t, idx.Insert(ctx, "good", []float32{1, 2, 3}, core.SparseVector{}, nil))

	err := idx.Insert(ctx, "bad", []float32{1, 2}, core.SparseVector{}, nil)

	require.Error(t, err, "a 2-d vector must not enter a 3-d HNSW index")

	_, idErr := idx.IDMapper.ToUint32ID("bad")
	assert.Error(t, idErr, "a rejected insert must not leave an id mapping behind")

	results, searchErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 1, nil)
	require.NoError(t, searchErr, "search must survive a rejected insert")
	require.Len(t, results, 1)
	assert.Equal(t, "good", results[0].ID)
}

func TestHNSWBatchInsert_DimensionMismatch_RejectsOnlyTheBadItem(t *testing.T) {
	ctx := context.Background()
	idx := newTestHNSWIndex(t)

	items := []BatchInsertItem{
		{ID: "a", Vector: []float32{1, 2, 3}},
		{ID: "bad", Vector: []float32{9, 9}},
		{ID: "b", Vector: []float32{4, 5, 6}},
	}

	failures, err := idx.BatchInsert(ctx, items)

	require.NoError(t, err, "one bad item must not abort the whole batch")
	require.Len(t, failures, 1)
	assert.Equal(t, "bad", failures[0].ID)

	results, searchErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 2, nil)
	require.NoError(t, searchErr, "search must survive a partially-rejected batch")
	assert.Len(t, results, 2)
}

func TestBatchInsert_DimensionMismatch_RejectsOnlyTheBadItem(t *testing.T) {
	ctx := context.Background()
	idx := newTestIndex(t)

	items := []BatchInsertItem{
		{ID: "a", Vector: []float32{1, 2, 3}},
		{ID: "bad", Vector: []float32{9, 9}},
		{ID: "b", Vector: []float32{4, 5, 6}},
	}

	failures, err := idx.BatchInsert(ctx, items)

	require.NoError(t, err, "one bad item must not abort the whole batch")
	require.Len(t, failures, 1)
	assert.Equal(t, "bad", failures[0].ID)
	assert.Len(t, idx.Store, 2, "the two well-formed items must still be stored")

	results, searchErr := idx.Search(ctx, []float32{1, 2, 3}, core.SparseVector{}, 2, nil)
	require.NoError(t, searchErr, "search must survive a partially-rejected batch")
	assert.Len(t, results, 2)
}
