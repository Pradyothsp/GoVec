package index

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/test/fixtures"
)

// =============================================================================
// Consolidated Index Tests
// =============================================================================

func TestNewVectorIndex(t *testing.T) {
	idx := newTestIndex(t)

	require.NotNil(t, idx, "NewVectorIndex should return non-nil")
	require.NotNil(t, idx.Store, "Store should be initialized")
	assert.Empty(t, idx.Store, "Store should be empty initially")
	assert.Equal(t, 0, len(idx.Store), "Store length should be 0")
}

// TestInsert_Variations consolidates all insert test variations into table-driven tests
func TestInsert_Variations(t *testing.T) {
	tests := []struct {
		name          string
		id            string
		vector        []float32
		sparse        core.SparseVector
		metadata      map[string]any
		expectError   bool
		validateAfter func(*testing.T, *VectorIndex[[]float32])
	}{
		{
			name:     "with_metadata",
			id:       "vec1",
			vector:   fixtures.Vec3dSimple,
			sparse:   core.SparseVector{},
			metadata: fixtures.MetaSimple,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, err := idx.IDMapper.ToUint32ID("vec1")
				require.NoError(t, err)
				assert.Contains(t, idx.Store, internalID)
				assert.Equal(t, fixtures.MetaSimple, idx.Store[internalID].Metadata)
			},
		},
		{
			name:     "without_metadata",
			id:       "vec2",
			vector:   fixtures.Vec3dAlternate,
			sparse:   core.SparseVector{},
			metadata: nil,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, _ := idx.IDMapper.ToUint32ID("vec2")
				assert.Nil(t, idx.Store[internalID].Metadata)
			},
		},
		{
			name:     "empty_metadata",
			id:       "vec3",
			vector:   fixtures.Vec3dThird,
			sparse:   core.SparseVector{},
			metadata: fixtures.MetaEmpty,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, _ := idx.IDMapper.ToUint32ID("vec3")
				assert.Equal(t, fixtures.MetaEmpty, idx.Store[internalID].Metadata)
			},
		},
		{
			name:     "nested_metadata",
			id:       "vec4",
			vector:   fixtures.Vec3dSimple,
			sparse:   core.SparseVector{},
			metadata: fixtures.MetaNested,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, _ := idx.IDMapper.ToUint32ID("vec4")
				assert.Equal(t, fixtures.MetaNested, idx.Store[internalID].Metadata)
			},
		},
		{
			// An empty vector can never be compared against anything, so it
			// would make every later search fail once a real vector set the
			// index's width -- rejected at insert instead.
			name:        "empty_vector_is_rejected",
			id:          "vec5",
			vector:      fixtures.VecEmpty,
			sparse:      core.SparseVector{},
			metadata:    nil,
			expectError: true,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.NotContains(t, idx.Store, uint32(0))
			},
		},
		{
			name:     "large_vector_1536d",
			id:       "vec6",
			vector:   fixtures.Vec1536d,
			sparse:   core.SparseVector{},
			metadata: map[string]any{"dims": 1536},
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, _ := idx.IDMapper.ToUint32ID("vec6")
				assert.Len(t, idx.Store[internalID].Vector, 1536)
			},
		},
		{
			name:     "special_chars_in_id",
			id:       "special-id_!@#$",
			vector:   fixtures.Vec3dSimple,
			sparse:   core.SparseVector{},
			metadata: nil,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, err := idx.IDMapper.ToUint32ID("special-id_!@#$")
				require.NoError(t, err)
				assert.Contains(t, idx.Store, internalID)
			},
		},
		{
			name:     "unicode_in_id",
			id:       "测试-unicode",
			vector:   fixtures.Vec3dAlternate,
			sparse:   core.SparseVector{},
			metadata: nil,
			validateAfter: func(t *testing.T, idx *VectorIndex[[]float32]) {
				internalID, err := idx.IDMapper.ToUint32ID("测试-unicode")
				require.NoError(t, err)
				assert.Contains(t, idx.Store, internalID)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := newTestIndex(t)

			err := idx.Insert(context.Background(), tt.id, tt.vector, tt.sparse, tt.metadata)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Len(t, idx.Store, 1, "Should have 1 vector in store")
				if tt.validateAfter != nil {
					tt.validateAfter(t, idx)
				}
			}
		})
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
func TestInsert_Concurrency(t *testing.T) {
	idx := newTestIndex(t)

	var wg sync.WaitGroup
	numGoroutines := 100
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(n int) {
			defer wg.Done()
			id := fmt.Sprintf("vec%d", n)
			vec := []float32{float32(n), float32(n * 2)}
			_ = idx.Insert(context.Background(), id, vec, core.SparseVector{}, map[string]any{"n": n}) //nolint:errcheck // test concurrency
		}(i)
	}

	wg.Wait()
	assert.Len(t, idx.Store, numGoroutines, "All concurrent inserts should succeed")
}

// TestSearch_Variations consolidates all search test variations
func TestSearch_Variations(t *testing.T) {
	tests := []struct {
		name        string
		setup       func(*VectorIndex[[]float32])
		query       []float32
		sparseQuery core.SparseVector
		limit       int
		filters     map[string]interface{}
		expectCount int
		expectError bool
	}{
		{
			name: "empty_index",
			setup: func(idx *VectorIndex[[]float32]) {
				// No setup
			},
			query:       fixtures.Vec3dSimple,
			sparseQuery: core.SparseVector{},
			limit:       10,
			filters:     nil,
			expectCount: 0,
		},
		{
			name: "single_vector",
			setup: func(idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "vec1", fixtures.Vec3dSimple, core.SparseVector{}, nil) //nolint:errcheck // test setup
			},
			query:       fixtures.Vec3dSimple,
			sparseQuery: core.SparseVector{},
			limit:       10,
			filters:     nil,
			expectCount: 1,
		},
		{
			name: "multiple_vectors",
			setup: func(idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil)    //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, nil)     //nolint:errcheck // test setup
			},
			query:       fixtures.Vec3dSimple,
			sparseQuery: core.SparseVector{},
			limit:       10,
			filters:     nil,
			expectCount: 3,
		},
		{
			name: "with_limit_2",
			setup: func(idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil)    //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, nil)     //nolint:errcheck // test setup
			},
			query:       fixtures.Vec3dSimple,
			sparseQuery: core.SparseVector{},
			limit:       2,
			filters:     nil,
			expectCount: 2,
		},
		{
			name: "with_limit_0",
			setup: func(idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil)    //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, nil)     //nolint:errcheck // test setup
			},
			query:       fixtures.Vec3dSimple,
			sparseQuery: core.SparseVector{},
			limit:       0,
			filters:     nil,
			expectCount: 3, // 0 means no limit
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := newTestIndex(t)
			tt.setup(idx)

			results, err := idx.Search(context.Background(), tt.query, tt.sparseQuery, tt.limit, tt.filters)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Len(t, results, tt.expectCount)
			}
		})
	}
}

func TestSearch_WithFilters(t *testing.T) {
	idx := newTestIndex(t)

	// Insert vectors with filterable metadata
	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, map[string]any{"category": "A", "value": 10})    //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, map[string]any{"category": "B", "value": 20}) //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, map[string]any{"category": "A", "value": 30})     //nolint:errcheck // test setup

	// Filter by category
	results, err := idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 10, map[string]interface{}{"category": "A"})
	require.NoError(t, err)
	assert.Len(t, results, 2, "Should find 2 vectors with category A")

	// Filter by value
	results, err = idx.Search(context.Background(), fixtures.Vec3dSimple, core.SparseVector{}, 10, map[string]interface{}{"value": 20})
	require.NoError(t, err)
	assert.Len(t, results, 1, "Should find 1 vector with value 20")
}

// TestSearch_ConcurrentInsertAndSearch runs searches alongside inserts. -race
// checks the memory accesses; the assertions check that every search
// succeeded with a full result and every insert landed.
func TestSearch_ConcurrentInsertAndSearch(t *testing.T) {
	// Arrange
	ctx := context.Background()
	idx := newTestIndex(t)
	const initial, inserted, searchers, k = 10, 20, 20, 5

	for i := 1; i <= initial; i++ {
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		require.NoError(t, idx.Insert(ctx, fmt.Sprintf("initial%d", i), vec, core.SparseVector{}, nil))
	}

	// Act
	var wg sync.WaitGroup
	errs := make(chan error, inserted+searchers)
	wg.Add(1 + searchers)
	go func() {
		defer wg.Done()
		for i := 1; i <= inserted; i++ {
			vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
			errs <- idx.Insert(ctx, fmt.Sprintf("new%d", i), vec, core.SparseVector{}, nil)
		}
	}()
	for range searchers {
		go func() {
			defer wg.Done()
			results, err := idx.Search(ctx, []float32{5, 10, 15}, core.SparseVector{}, k, nil)
			if err == nil && len(results) != k {
				err = fmt.Errorf("search returned %d results, want %d", len(results), k)
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)

	// Assert
	for err := range errs {
		require.NoError(t, err)
	}
	assert.Equal(t, initial+inserted, idx.Len())
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
			require.NoError(t, idx.Reset(ctx, ""))

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

func TestLen(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			lens := make([]int, 0, 3)

			// Act: record the length after each change.
			lens = append(lens, idx.Len())
			require.NoError(t, idx.Insert(ctx, "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))
			require.NoError(t, idx.Insert(ctx, "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil))
			lens = append(lens, idx.Len())
			_, err := idx.Delete(ctx, "v1")
			require.NoError(t, err)
			lens = append(lens, idx.Len())

			// Assert
			assert.Equal(t, []int{0, 2, 1}, lens, "empty, after two inserts, after one delete")
		})
	}
}
