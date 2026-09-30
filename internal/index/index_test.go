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

func TestInsert_Overwrite(t *testing.T) {
	idx := newTestIndex(t)

	// Initial insert
	err := idx.Insert(context.Background(), "vec1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple)
	require.NoError(t, err)

	internalID, _ := idx.IDMapper.ToUint32ID("vec1")
	assert.Equal(t, fixtures.Vec3dSimple, idx.Store[internalID].Vector)
	assert.Equal(t, fixtures.MetaSimple, idx.Store[internalID].Metadata)

	// Overwrite with new vector and metadata
	newVec := fixtures.Vec3dAlternate
	newMeta := fixtures.MetaNested
	err = idx.Insert(context.Background(), "vec1", newVec, core.SparseVector{}, newMeta)
	require.NoError(t, err)

	assert.Len(t, idx.Store, 1, "Should still have 1 vector (overwritten)")
	assert.Equal(t, newVec, idx.Store[internalID].Vector)
	assert.Equal(t, newMeta, idx.Store[internalID].Metadata)
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

func TestSearch_Concurrency(t *testing.T) {
	idx := newTestIndex(t)

	// Pre-populate
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("vec%d", i)
		vec := []float32{float32(i), float32(i * 2)}
		_ = idx.Insert(context.Background(), id, vec, core.SparseVector{}, nil) //nolint:errcheck // test setup
	}

	query := []float32{5.0, 10.0}

	var wg sync.WaitGroup
	numGoroutines := 50
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_, _ = idx.Search(context.Background(), query, core.SparseVector{}, 5, nil) //nolint:errcheck // test concurrency
		}()
	}

	wg.Wait()
	// No assertion - test passes if no race conditions occur
}

func TestSearch_ConcurrentInsertAndSearch(t *testing.T) {
	idx := newTestIndex(t)

	// Pre-populate with non-zero vectors to avoid edge cases
	for i := 1; i <= 10; i++ {
		id := fmt.Sprintf("initial%d", i)
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		_ = idx.Insert(context.Background(), id, vec, core.SparseVector{}, nil) //nolint:errcheck // test setup
	}

	var wg sync.WaitGroup
	wg.Add(2)

	// Concurrent inserts
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			id := fmt.Sprintf("new%d", i)
			vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
			_ = idx.Insert(context.Background(), id, vec, core.SparseVector{}, nil) //nolint:errcheck // test concurrency
		}
	}()

	// Concurrent searches
	go func() {
		defer wg.Done()
		query := []float32{5.0, 10.0, 15.0}
		for i := 0; i < 20; i++ {
			_, _ = idx.Search(context.Background(), query, core.SparseVector{}, 5, nil) //nolint:errcheck // test concurrency
		}
	}()

	wg.Wait()
	// Test passes if no race conditions occur
}

func TestDelete(t *testing.T) {
	idx := newTestIndex(t)

	// Insert vectors
	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil)    //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
	assert.Len(t, idx.Store, 2)

	// Get the internal ID before deletion
	internalID, err := idx.IDMapper.ToUint32ID("v1")
	require.NoError(t, err)

	// Delete one vector
	deleted, err := idx.Delete(context.Background(), "v1")
	require.NoError(t, err)
	assert.True(t, deleted, "Delete should return true for successful deletion")
	assert.Len(t, idx.Store, 1)

	// Verify it's deleted
	_, err = idx.IDMapper.ToUint32ID("v1")
	assert.Error(t, err, "Deleted ID should not be found")
	assert.NotContains(t, idx.Store, internalID)

	// Verify tombstone
	assert.True(t, idx.IDMapper.IsTombstone(internalID))
}

func TestClear(t *testing.T) {
	idx := newTestIndex(t)

	// Insert vectors
	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil)    //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
	assert.Len(t, idx.Store, 2)

	// Clear
	idx.Clear()
	assert.Empty(t, idx.Store, "Store should be empty after clear")
}

// TestClear_ResetsIDMapperAndWAL verifies Clear wipes ID mappings/tombstones
// and truncates the WAL, not just the Store -- these are easy to miss since
// they're separate fields the caller could forget to reset.
func TestClear_ResetsIDMapperAndWAL(t *testing.T) {
	walPath := filepath.Join(t.TempDir(), "test.wal")
	wal, err := NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	idx := newTestIndexWithWAL(t, wal)

	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil)    //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
	_, err = idx.Delete(context.Background(), "v2")
	require.NoError(t, err)

	info, err := os.Stat(walPath)
	require.NoError(t, err)
	assert.Positive(t, info.Size(), "WAL should have entries before Clear")

	idx.Clear()

	assert.Empty(t, idx.Store, "Store should be empty after Clear")
	assert.Equal(t, 0, idx.IDMapper.Count(), "IDMapper should have no active mappings after Clear")
	assert.Equal(t, uint32(0), idx.IDMapper.NextID(), "IDMapper's ID counter should restart at 0 after Clear")

	info, err = os.Stat(walPath)
	require.NoError(t, err)
	assert.Zero(t, info.Size(), "WAL should be truncated after Clear")

	// Re-inserting after a full Clear may reuse ID 0 -- safe here because the
	// entire collection was wiped alongside it, unlike a plain Delete.
	_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dSimple, core.SparseVector{}, nil) //nolint:errcheck // test setup
	newID, err := idx.IDMapper.ToUint32ID("v3")
	require.NoError(t, err)
	assert.Equal(t, uint32(0), newID)
}

func TestLen(t *testing.T) {
	idx := newTestIndex(t)
	assert.Equal(t, 0, idx.Len())

	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil) //nolint:errcheck // test setup
	assert.Equal(t, 1, idx.Len())

	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil) //nolint:errcheck // test setup
	assert.Equal(t, 2, idx.Len())

	_, _ = idx.Delete(context.Background(), "v1") //nolint:errcheck // test setup
	assert.Equal(t, 1, idx.Len())
}
