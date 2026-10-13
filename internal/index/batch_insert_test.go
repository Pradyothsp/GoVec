package index

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A batch is applied whole and logged as one WAL write, so replaying that
// write reproduces the batch exactly. Per-item failures and a failed WAL write
// are covered in dimension_guard_test.go and write_ahead_test.go.
func TestBatchInsert(t *testing.T) {
	ctx := context.Background()
	items := []BatchInsertItem{
		{ID: "v1", Vector: []float32{1, 0, 0}, Meta: map[string]any{"label": "first"}},
		{ID: "v2", Vector: []float32{0, 1, 0}},
		{ID: "v3", Vector: []float32{0, 0, 1}},
	}
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)

			// Act
			emptyFailures, emptyErr := idx.BatchInsert(ctx, nil)
			failures, err := idx.BatchInsert(ctx, items)

			// Assert
			require.NoError(t, emptyErr)
			assert.Empty(t, emptyFailures)
			require.NoError(t, err)
			assert.Empty(t, failures)

			replayed, _ := e.open(t)
			require.NoError(t, replayed.ReplayWAL(r.wal.Path()))
			for _, engine := range []FileEngine{idx, replayed} {
				assert.Equal(t, len(items), engine.Len())
				for _, item := range items {
					got, err := engine.GetByID(ctx, item.ID)
					require.NoError(t, err)
					assert.Equal(t, item.Vector, got.Vector)
					assert.Equal(t, item.Meta, got.Metadata)
				}
			}
		})
	}
}
