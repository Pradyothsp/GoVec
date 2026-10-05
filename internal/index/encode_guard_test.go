package index

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// TestInsert_Float32_StoresExactSizeCopy checks that both engines keep their
// own exact-size copy of a float32 vector, not the caller's slice: decoders
// leave spare capacity (about 17% at 1536 dimensions), and the caller still
// owns that memory.
func TestInsert_Float32_StoresExactSizeCopy(t *testing.T) {
	stored := map[string]func(t *testing.T, vec []float32) []float32{
		"brute": func(t *testing.T, vec []float32) []float32 {
			idx := newTestIndex(t)
			require.NoError(t, idx.Insert(context.Background(), "a", vec, core.SparseVector{}, nil))
			id, err := idx.IDMapper.ToUint32ID("a")
			require.NoError(t, err)
			return idx.Store[id].Vector
		},
		"hnsw": func(t *testing.T, vec []float32) []float32 {
			idx := newTestHNSWIndex(t)
			require.NoError(t, idx.Insert(context.Background(), "a", vec, core.SparseVector{}, nil))
			id, err := idx.IDMapper.ToUint32ID("a")
			require.NoError(t, err)
			return idx.metadata[id].Vector
		},
	}

	for name, storedVector := range stored {
		t.Run(name, func(t *testing.T) {
			// As a JSON decoder leaves it: length 3, room for more.
			vec := make([]float32, 3, 8)
			copy(vec, []float32{0.6, 0.8, 0})

			got := storedVector(t, vec)
			vec[0] = 99

			assert.Equal(t, []float32{0.6, 0.8, 0}, got, "the caller's later writes must not reach the index")
			assert.Equal(t, len(got), cap(got), "the stored copy should carry no spare capacity")
		})
	}
}
