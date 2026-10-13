package index

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

// An insert stores the vector and metadata it was given, as given; an empty
// vector can never be compared against anything, so it is refused.
func TestInsert(t *testing.T) {
	tests := []struct {
		name    string
		vector  []float32
		meta    map[string]any
		wantErr error
	}{
		{name: "with metadata", vector: []float32{1, 2, 3}, meta: map[string]any{"label": "x", "nested": map[string]any{"n": 1}}},
		{name: "without metadata", vector: []float32{1, 2, 3}},
		{name: "empty metadata", vector: []float32{1, 2, 3}, meta: map[string]any{}},
		{name: "empty vector", vector: []float32{}, wantErr: ErrEmptyVector},
	}
	for _, e := range bothEngines {
		for _, tt := range tests {
			t.Run(e.name+"/"+tt.name, func(t *testing.T) {
				// Arrange
				idx, _ := e.open(t)

				// Act
				err := idx.Insert(context.Background(), "v", tt.vector, core.SparseVector{}, tt.meta)

				// Assert
				if tt.wantErr != nil {
					require.ErrorIs(t, err, tt.wantErr)
					assert.Zero(t, idx.Len())
					return
				}
				require.NoError(t, err)
				got, err := idx.GetByID(context.Background(), "v")
				require.NoError(t, err)
				assert.Equal(t, tt.vector, got.Vector)
				assert.Equal(t, tt.meta, got.Metadata)
			})
		}
	}
}

// An insert with an existing ID replaces the record -- vector and metadata --
// and, for HNSW, the graph node, so search finds the new vector.
func TestInsert_Overwrite(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			require.NoError(t, idx.Insert(ctx, "vec1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple))
			require.NoError(t, idx.Insert(ctx, "other", fixtures.Vec3dThird, core.SparseVector{}, nil))

			// Act
			err := idx.Insert(ctx, "vec1", fixtures.Vec3dAlternate, core.SparseVector{}, fixtures.MetaNested)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, 2, idx.Len(), "an overwrite must not add a record")
			internalID, err := r.IDMapper.ToUint32ID("vec1")
			require.NoError(t, err)
			assert.Equal(t, fixtures.Vec3dAlternate, r.Store[internalID].Vector)
			assert.Equal(t, fixtures.MetaNested, r.Store[internalID].Metadata)

			results, err := idx.Search(ctx, fixtures.Vec3dAlternate, core.SparseVector{}, 1, nil)
			require.NoError(t, err)
			require.Len(t, results, 1)
			assert.Equal(t, "vec1", results[0].ID)
			assert.InDelta(t, 1, results[0].Score, 1e-5, "search must see the new vector, not the old one")
		})
	}
}

// TestConcurrentWritesSearchesAndSaves runs inserts from several goroutines
// alongside searches and saves. -race checks the memory accesses; the
// assertions check that every call succeeded, every search saw a full result,
// and the last save holds every insert.
func TestConcurrentWritesSearchesAndSaves(t *testing.T) {
	ctx := context.Background()
	const writers, perWriter, searchers, saves, k = 5, 10, 10, 3, 5
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			for i := range k {
				require.NoError(t, idx.Insert(ctx, fmt.Sprintf("initial%d", i), []float32{1, float32(i), 0}, core.SparseVector{}, nil))
			}
			path := filepath.Join(t.TempDir(), "snapshot.bin")

			// Act
			var wg sync.WaitGroup
			errs := make(chan error, writers*perWriter+searchers+saves)
			for w := range writers {
				wg.Go(func() {
					for i := range perWriter {
						errs <- idx.Insert(ctx, fmt.Sprintf("w%d-%d", w, i), []float32{float32(w), float32(i), 1}, core.SparseVector{}, nil)
					}
				})
			}
			for range searchers {
				wg.Go(func() {
					results, err := idx.Search(ctx, []float32{1, 0, 0}, core.SparseVector{}, k, nil)
					if err == nil && len(results) != k {
						err = fmt.Errorf("search returned %d results, want %d", len(results), k)
					}
					errs <- err
				})
			}
			for range saves {
				wg.Go(func() { errs <- idx.SaveToFile(ctx, path) })
			}
			wg.Wait()
			close(errs)

			// Assert
			for err := range errs {
				require.NoError(t, err)
			}
			want := k + writers*perWriter
			assert.Equal(t, want, idx.Len())
			require.NoError(t, idx.SaveToFile(ctx, path))
			loaded, _ := e.open(t)
			require.NoError(t, loaded.LoadFromFile(ctx, path))
			assert.Equal(t, want, loaded.Len())
		})
	}
}

func TestDelete(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			require.NoError(t, idx.Insert(ctx, "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
			require.NoError(t, idx.Insert(ctx, "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
			internalID, err := r.IDMapper.ToUint32ID("v1")
			require.NoError(t, err)

			// Act
			deleted, err := idx.Delete(ctx, "v1")
			missingDeleted, missingErr := idx.Delete(ctx, "does-not-exist")

			// Assert
			require.NoError(t, err)
			assert.True(t, deleted)
			assert.Equal(t, 1, idx.Len())
			assert.NotContains(t, r.Store, internalID)
			_, err = r.IDMapper.ToUint32ID("v1")
			assert.Error(t, err, "a deleted ID must not resolve")
			assert.True(t, r.IDMapper.IsTombstone(internalID))

			results, err := idx.Search(ctx, fixtures.Vec3dSimple, core.SparseVector{}, 5, nil)
			require.NoError(t, err)
			for _, res := range results {
				assert.NotEqual(t, "v1", res.ID, "search must not return a deleted vector")
			}
			require.NoError(t, missingErr)
			assert.False(t, missingDeleted, "deleting an unknown ID reports false")
		})
	}
}

// TestReset_ResetsIDMapperAndWAL verifies Reset empties the store and also
// wipes ID mappings and truncates the WAL -- separate state that is easy to
// forget to reset -- and that the index is usable afterwards.
func TestReset_ResetsIDMapperAndWAL(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			walPath := r.wal.file.Name()
			require.NoError(t, idx.Insert(ctx, "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
			require.NoError(t, idx.Insert(ctx, "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
			_, err := idx.Delete(ctx, "v2")
			require.NoError(t, err)
			info, err := os.Stat(walPath)
			require.NoError(t, err)
			require.Positive(t, info.Size(), "WAL should have entries before Reset")

			// Act
			require.NoError(t, idx.Reset(ctx))

			// Assert
			assert.Empty(t, r.Store)
			assert.Equal(t, 0, idx.Len())
			assert.Equal(t, 0, r.IDMapper.Count(), "IDMapper should have no active mappings after Reset")
			assert.Equal(t, uint32(0), r.IDMapper.NextID(), "IDMapper's ID counter should restart at 0 after Reset")
			info, err = os.Stat(walPath)
			require.NoError(t, err)
			assert.Zero(t, info.Size(), "WAL should be truncated after Reset")

			// Re-inserting after a full Reset may reuse ID 0 -- safe here because
			// the entire collection was wiped alongside it, unlike a plain Delete.
			require.NoError(t, idx.Insert(ctx, "v3", fixtures.Vec3dSimple, core.SparseVector{}, nil))
			newID, err := r.IDMapper.ToUint32ID("v3")
			require.NoError(t, err)
			assert.Equal(t, uint32(0), newID)
			results, err := idx.Search(ctx, fixtures.Vec3dSimple, core.SparseVector{}, 1, nil)
			require.NoError(t, err)
			require.Len(t, results, 1)
			assert.Equal(t, "v3", results[0].ID)
		})
	}
}
