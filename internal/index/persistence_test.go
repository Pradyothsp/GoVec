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
// Consolidated Persistence Tests
// =============================================================================

// TestSaveAndLoad_Variations uses table-driven tests to cover:
// - Empty index
// - Basic vectors (3D)
// - Large vectors (1536D, 4096D)
// - Special characters in IDs
// - Various metadata types
// - Hybrid search (with inverted index)
func TestSaveAndLoad_Variations(t *testing.T) {
	tests := []struct {
		name      string
		setup     func(*testing.T) *VectorIndex[[]float32]
		populate  func(*testing.T, *VectorIndex[[]float32])
		validate  func(*testing.T, *VectorIndex[[]float32])
		expectLen int
	}{
		{
			name: "empty_index",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				// No population
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Empty(t, idx.Store)
			},
			expectLen: 0,
		},
		{
			name: "basic_vectors",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil)              //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, fixtures.MetaEmpty)   //nolint:errcheck // test setup
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 3)
				// Translate external IDs to internal IDs
				id1, err1 := idx.IDMapper.ToUint32ID("v1")
				id2, err2 := idx.IDMapper.ToUint32ID("v2")
				id3, err3 := idx.IDMapper.ToUint32ID("v3")
				require.NoError(t, err1)
				require.NoError(t, err2)
				require.NoError(t, err3)

				assert.Contains(t, idx.Store, id1)
				assert.Contains(t, idx.Store, id2)
				assert.Contains(t, idx.Store, id3)
				assert.Equal(t, fixtures.Vec3dSimple, idx.Store[id1].Vector)
				assert.Equal(t, fixtures.Vec3dAlternate, idx.Store[id2].Vector)
				assert.Equal(t, fixtures.Vec3dThird, idx.Store[id3].Vector)
			},
			expectLen: 3,
		},
		{
			// Two 1536-d vectors, not one 1536-d and one 4096-d: an index has a
			// single width, and mixing them is now rejected at insert. 4096-d
			// persistence gets its own index in the next case.
			name: "large_vectors_1536d",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "large1", fixtures.Vec1536d, core.SparseVector{}, map[string]any{"size": 1536}) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "large2", fixtures.Vec1536d, core.SparseVector{}, map[string]any{"size": 1536}) //nolint:errcheck // test setup
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 2)
				id1, err1 := idx.IDMapper.ToUint32ID("large1")
				id2, err2 := idx.IDMapper.ToUint32ID("large2")
				require.NoError(t, err1)
				require.NoError(t, err2)

				assert.Contains(t, idx.Store, id1)
				assert.Contains(t, idx.Store, id2)
				assert.Len(t, idx.Store[id1].Vector, 1536)
				assert.Len(t, idx.Store[id2].Vector, 1536)
			},
			expectLen: 2,
		},
		{
			name: "large_vectors_4096d",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "large4096", fixtures.Vec4096d, core.SparseVector{}, map[string]any{"size": 4096}) //nolint:errcheck // test setup
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 1)
				id, err := idx.IDMapper.ToUint32ID("large4096")
				require.NoError(t, err)

				assert.Contains(t, idx.Store, id)
				assert.Len(t, idx.Store[id].Vector, 4096)
			},
			expectLen: 1,
		},
		{
			name: "special_characters_in_ids",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				// All 3-d: this case is about the ids, not the vectors, and an
				// empty vector is no longer a valid insert.
				_ = idx.Insert(context.Background(), "plain_vec", fixtures.Vec3dSimple, core.SparseVector{}, nil)                         //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "special-chars_!@#", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "unicode-测试", fixtures.Vec3dAlternate, core.SparseVector{}, nil)                     //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "spaces in id", fixtures.Vec3dThird, core.SparseVector{}, fixtures.MetaEmpty)        //nolint:errcheck // test setup
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 4)
				// Verify special IDs can be loaded
				_, err1 := idx.IDMapper.ToUint32ID("special-chars_!@#")
				_, err2 := idx.IDMapper.ToUint32ID("unicode-测试")
				_, err3 := idx.IDMapper.ToUint32ID("spaces in id")
				require.NoError(t, err1)
				require.NoError(t, err2)
				require.NoError(t, err3)
			},
			expectLen: 4,
		},
		{
			name: "various_metadata_types",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "meta1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple)    //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "meta2", fixtures.Vec3dAlternate, core.SparseVector{}, fixtures.MetaNested) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "meta3", fixtures.Vec3dThird, core.SparseVector{}, fixtures.MetaComplex)    //nolint:errcheck // test setup
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 3)
				id1, _ := idx.IDMapper.ToUint32ID("meta1")
				id2, _ := idx.IDMapper.ToUint32ID("meta2")
				id3, _ := idx.IDMapper.ToUint32ID("meta3")

				assert.Equal(t, fixtures.MetaSimple, idx.Store[id1].Metadata)
				assert.Equal(t, fixtures.MetaNested, idx.Store[id2].Metadata)
				assert.Equal(t, fixtures.MetaComplex, idx.Store[id3].Metadata)
			},
			expectLen: 3,
		},
		{
			name: "hybrid_search_with_inverted_index",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndexWithHybrid(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "doc1", fixtures.Vec3dSimple, fixtures.SparseBM25, map[string]any{ //nolint:errcheck // test setup
					"title": "First Document",
					"type":  "article",
				})
				_ = idx.Insert(context.Background(), "doc2", fixtures.Vec3dAlternate, fixtures.SparseTfIdf, map[string]any{ //nolint:errcheck // test setup
					"title": "Second Document",
					"type":  "article",
				})
				_ = idx.Insert(context.Background(), "doc3", fixtures.Vec3dThird, fixtures.SparseLarge, map[string]any{ //nolint:errcheck // test setup
					"title": "Third Document",
					"type":  "blog",
				})
			},
			validate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				assert.Len(t, idx.Store, 3)
				// The three fixtures cover 9 distinct tokens with 10 postings
				// (token 100 appears in both SparseTfIdf and SparseLarge).
				// NotNil alone passed while every posting was being lost on load.
				assert.Len(t, idx.InvertedIndex, 9, "every token must have its posting list back")
				assert.Equal(t, 10, countPostings(idx.InvertedIndex), "every posting must survive the reload")
			},
			expectLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup index
			idx := tt.setup(t)
			tt.populate(t, idx)

			// Save to file
			tmpFile := filepath.Join(t.TempDir(), "test.bin")
			err := idx.SaveToFile(context.Background(), tmpFile)
			require.NoError(t, err, "SaveToFile should succeed")

			// Verify file exists
			_, err = os.Stat(tmpFile)
			require.NoError(t, err, "File should exist after save")

			// Load into new index
			newIdx := tt.setup(t) // Use same setup (with/without hybrid)
			err = newIdx.LoadFromFile(context.Background(), tmpFile)
			require.NoError(t, err, "LoadFromFile should succeed")

			// Validate
			assert.Len(t, newIdx.Store, tt.expectLen)
			tt.validate(t, newIdx)

			// Verify IDMapper count is preserved
			assert.Equal(t, idx.IDMapper.Count(), newIdx.IDMapper.Count(), "IDMapper count should be preserved")
			assert.Equal(t, idx.IDMapper.NextID(), newIdx.IDMapper.NextID(), "IDMapper NextID should be preserved")
		})
	}
}

// TestLoadFromFile_MissingFile verifies graceful handling when file doesn't exist
func TestLoadFromFile_MissingFile(t *testing.T) {
	idx := newTestIndex(t)

	err := idx.LoadFromFile(context.Background(), "/nonexistent/path/file.bin")

	// Should return nil error (graceful handling for missing files)
	assert.NoError(t, err, "Missing file should be handled gracefully")
	assert.Empty(t, idx.Store, "Index should remain empty")
}

// TestLoadFromFile_CorruptedFile verifies error handling for corrupted data
func TestLoadFromFile_CorruptedFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "corrupted.bin")

	// Create corrupted file
	corruptedData := []byte("This is not valid GOB data!\nIt should fail to decode.")
	err := os.WriteFile(tmpFile, corruptedData, 0o600)
	require.NoError(t, err)

	idx := newTestIndex(t)
	err = idx.LoadFromFile(context.Background(), tmpFile)

	assert.Error(t, err, "Should return error for corrupted file")
	assert.Contains(t, err.Error(), "failed to decode snapshot", "Error should mention decoding failure")
}

// TestSaveToFile_InvalidPath verifies error handling for invalid paths
func TestSaveToFile_InvalidPath(t *testing.T) {
	idx := newTestIndex(t)
	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple) //nolint:errcheck // test setup

	err := idx.SaveToFile(context.Background(), filepath.Join(uncreatableDir(t), "test.bin"))

	assert.Error(t, err, "Should return error for invalid path")
}

// A snapshot path whose directory doesn't exist yet must not make every save
// fail -- auto-save would log the same error every interval.
func TestSaveToFile_CreatesParentDirectories(t *testing.T) {
	engines := []struct {
		name string
		open func(t *testing.T) Engine
	}{
		{"brute", func(t *testing.T) Engine { return newTestIndex(t) }},
		{"hnsw", func(t *testing.T) Engine { return newTestHNSWIndex(t) }},
	}
	for _, e := range engines {
		t.Run(e.name, func(t *testing.T) {
			ctx := context.Background()
			idx := e.open(t)
			require.NoError(t, idx.Insert(ctx, "v1", fixtures.Vec3dSimple, core.SparseVector{}, nil))

			path := filepath.Join(t.TempDir(), "nested", "deep", "snapshot.bin")
			require.NoError(t, idx.SaveToFile(ctx, path))
			assert.FileExists(t, path)
		})
	}
}

// TestAtomicWrite_Verification verifies temp file cleanup after successful write
func TestAtomicWrite_Verification(t *testing.T) {
	idx := newTestIndex(t)
	_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple) //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil)              //nolint:errcheck // test setup
	_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, fixtures.MetaEmpty)   //nolint:errcheck // test setup

	tmpFile := filepath.Join(t.TempDir(), "atomic_test.bin")
	err := idx.SaveToFile(context.Background(), tmpFile)
	require.NoError(t, err)

	// Verify main file exists
	_, err = os.Stat(tmpFile)
	require.NoError(t, err, "Main file should exist")

	// Verify temp file is cleaned up
	tmpTempFile := tmpFile + ".tmp"
	_, err = os.Stat(tmpTempFile)
	assert.True(t, os.IsNotExist(err), "Temp file should be cleaned up")
}

// TestConcurrentOperations verifies save operations work during concurrent access
func TestConcurrentOperations(t *testing.T) {
	tests := []struct {
		name string
		op   func(*VectorIndex[[]float32], *sync.WaitGroup)
	}{
		{
			name: "concurrent_insert_and_save",
			op: func(idx *VectorIndex[[]float32], wg *sync.WaitGroup) {
				defer wg.Done()
				for i := 0; i < 10; i++ {
					id := fmt.Sprintf("concurrent_%d", i)
					vec := []float32{float32(i), float32(i * 2)}
					_ = idx.Insert(context.Background(), id, vec, core.SparseVector{}, nil) //nolint:errcheck // test concurrency
				}
			},
		},
		{
			name: "concurrent_search_and_save",
			op: func(idx *VectorIndex[[]float32], wg *sync.WaitGroup) {
				defer wg.Done()
				query := []float32{1.0, 2.0, 3.0}
				for i := 0; i < 10; i++ {
					_, _ = idx.Search(context.Background(), query, core.SparseVector{}, 5, nil) //nolint:errcheck // test concurrency
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := newTestIndex(t)
			_ = idx.Insert(context.Background(), "v1", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple) //nolint:errcheck // test setup
			_ = idx.Insert(context.Background(), "v2", fixtures.Vec3dAlternate, core.SparseVector{}, nil)              //nolint:errcheck // test setup
			_ = idx.Insert(context.Background(), "v3", fixtures.Vec3dThird, core.SparseVector{}, fixtures.MetaEmpty)   //nolint:errcheck // test setup

			tmpFile := filepath.Join(t.TempDir(), "concurrent.bin")

			var wg sync.WaitGroup
			wg.Add(2)

			// Start concurrent operation
			go tt.op(idx, &wg)

			// Start concurrent save
			go func() {
				defer wg.Done()
				for i := 0; i < 5; i++ {
					_ = idx.SaveToFile(context.Background(), tmpFile) //nolint:errcheck // test concurrency
				}
			}()

			wg.Wait()

			// Verify final save works
			err := idx.SaveToFile(context.Background(), tmpFile)
			require.NoError(t, err)

			// Verify load works
			newIdx := newTestIndex(t)
			err = newIdx.LoadFromFile(context.Background(), tmpFile)
			require.NoError(t, err)
			assert.NotEmpty(t, newIdx.Store, "Should have loaded some data")
		})
	}
}

// TestRoundTrip_ComplexMetadata verifies complex nested metadata survives save/load
func TestRoundTrip_ComplexMetadata(t *testing.T) {
	idx := newTestIndex(t)

	// Insert with complex nested metadata
	complexMeta := map[string]any{
		"string": "value",
		"int":    123,
		"float":  3.14,
		"bool":   true,
		"array":  []string{"a", "b", "c"},
		"nested": map[string]any{
			"level2": map[string]any{
				"level3": "deep",
				"number": 42,
			},
			"array": []int{1, 2, 3},
		},
		"nil_value": nil,
	}

	err := idx.Insert(context.Background(), "complex", fixtures.Vec3dSimple, core.SparseVector{}, complexMeta)
	require.NoError(t, err)

	// Save and load
	tmpFile := filepath.Join(t.TempDir(), "complex_meta.bin")
	err = idx.SaveToFile(context.Background(), tmpFile)
	require.NoError(t, err)

	newIdx := newTestIndex(t)
	err = newIdx.LoadFromFile(context.Background(), tmpFile)
	require.NoError(t, err)

	// Verify complex metadata is preserved
	id, err := newIdx.IDMapper.ToUint32ID("complex")
	require.NoError(t, err)
	assert.Contains(t, newIdx.Store, id)

	loadedMeta := newIdx.Store[id].Metadata
	assert.Equal(t, complexMeta, loadedMeta, "Complex metadata should be preserved exactly")
}

// TestIDMapper_Persistence verifies IDMapper survives save/load cycles
func TestIDMapper_Persistence(t *testing.T) {
	idx := newTestIndex(t)

	// Insert vectors with string IDs
	testIDs := []string{"id-1", "id-2", "id-3", "special!@#", "unicode-测试"}
	for _, id := range testIDs {
		err := idx.Insert(context.Background(), id, fixtures.Vec3dSimple, core.SparseVector{}, nil)
		require.NoError(t, err)
	}

	// Capture original count and nextID
	originalCount := idx.IDMapper.Count()
	originalNextID := idx.IDMapper.NextID()

	// Save and load
	tmpFile := filepath.Join(t.TempDir(), "idmapper_test.bin")
	err := idx.SaveToFile(context.Background(), tmpFile)
	require.NoError(t, err)

	newIdx := newTestIndex(t)
	err = newIdx.LoadFromFile(context.Background(), tmpFile)
	require.NoError(t, err)

	// Verify count and nextID are preserved
	assert.Equal(t, originalCount, newIdx.IDMapper.Count(), "IDMapper count should be preserved")
	assert.Equal(t, originalNextID, newIdx.IDMapper.NextID(), "IDMapper NextID should be preserved")

	// Verify we can still look up all IDs (this proves mappings are preserved)
	for _, id := range testIDs {
		internalID, err := newIdx.IDMapper.ToUint32ID(id)
		require.NoError(t, err, "Should find mapping for %s", id)
		assert.Contains(t, newIdx.Store, internalID, "Vector should exist for %s", id)
	}
}

// TestLoadFromFile_RebuildsInvertedIndex is the regression test for reloaded
// hybrid indexes losing every sparse posting: the load path used to look for a
// saved copy behind a version check that could never pass, and kept the empty
// map from the constructor. Snapshots no longer store the inverted index at
// all; LoadFromFile rebuilds it from the nodes' sparse vectors.
func TestLoadFromFile_RebuildsInvertedIndex(t *testing.T) {
	ctx := context.Background()

	// doc2 is the closer dense match; only a working sparse boost on token 10
	// can put doc1 above it.
	populate := func(t *testing.T, idx *VectorIndex[[]float32]) {
		t.Helper()
		require.NoError(t, idx.Insert(ctx, "doc1", []float32{0.9, 0.1, 0}, core.SparseVector{
			Indices: []uint32{10}, Values: []float32{10.0},
		}, nil))
		require.NoError(t, idx.Insert(ctx, "doc2", []float32{1, 0, 0}, core.SparseVector{}, nil))
		require.NoError(t, idx.Insert(ctx, "doc3", []float32{0, 1, 0}, core.SparseVector{
			Indices: []uint32{99}, Values: []float32{9.0},
		}, nil))
	}

	assertSparseBoostWorks := func(t *testing.T, idx *VectorIndex[[]float32]) {
		t.Helper()
		assert.Len(t, idx.InvertedIndex, 2, "tokens 10 and 99 must both have posting lists")
		assert.Equal(t, 2, countPostings(idx.InvertedIndex))

		sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{1.0}}
		results, err := idx.Search(ctx, []float32{1, 0, 0}, sparseQuery, 2, nil)
		require.NoError(t, err)
		require.Len(t, results, 2)
		assert.Equal(t, "doc1", results[0].ID, "the sparse boost must outrank doc2's better dense score")
	}

	t.Run("hybrid_enabled", func(t *testing.T) {
		idx := newTestIndexWithHybrid(t)
		populate(t, idx)

		// Sanity check: the ranking really does depend on the sparse boost.
		assertSparseBoostWorks(t, idx)

		path := filepath.Join(t.TempDir(), "hybrid.bin")
		require.NoError(t, idx.SaveToFile(ctx, path))

		loaded := newTestIndexWithHybrid(t)
		require.NoError(t, loaded.LoadFromFile(ctx, path))
		assertSparseBoostWorks(t, loaded)
	})

	t.Run("hybrid_disabled_loads_hybrid_snapshot", func(t *testing.T) {
		idx := newTestIndexWithHybrid(t)
		populate(t, idx)

		path := filepath.Join(t.TempDir(), "hybrid.bin")
		require.NoError(t, idx.SaveToFile(ctx, path))

		loaded := newTestIndex(t)
		require.NoError(t, loaded.LoadFromFile(ctx, path))
		assert.Nil(t, loaded.InvertedIndex, "an index without hybrid search must not grow an inverted index")
		assert.Len(t, loaded.Store, 3)
	})
}

func countPostings(inverted map[uint32][]core.Posting) int {
	n := 0
	for _, postings := range inverted {
		n += len(postings)
	}
	return n
}
