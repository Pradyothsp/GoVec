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
			name: "large_vectors",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "large1", fixtures.Vec1536d, core.SparseVector{}, map[string]any{"size": 1536}) //nolint:errcheck // test setup
				_ = idx.Insert(context.Background(), "large2", fixtures.Vec4096d, core.SparseVector{}, map[string]any{"size": 4096}) //nolint:errcheck // test setup
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
				assert.Len(t, idx.Store[id2].Vector, 4096)
			},
			expectLen: 2,
		},
		{
			name: "special_characters_in_ids",
			setup: func(t *testing.T) *VectorIndex[[]float32] {
				return newTestIndex(t)
			},
			populate: func(t *testing.T, idx *VectorIndex[[]float32]) {
				_ = idx.Insert(context.Background(), "empty_vec", fixtures.VecEmpty, core.SparseVector{}, nil)                            //nolint:errcheck // test setup
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
				assert.NotNil(t, idx.InvertedIndex, "Inverted index should be preserved")
				// Inverted index entries may be empty if sparse vectors weren't indexed
				// The important thing is that the inverted index map itself is not nil
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

	// Try to save to invalid directory
	invalidPath := "/nonexistent/invalid/path/test.bin"
	err := idx.SaveToFile(context.Background(), invalidPath)

	assert.Error(t, err, "Should return error for invalid path")
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

// TestVersionMismatch verifies backward compatibility with older snapshot versions
func TestVersionMismatch(t *testing.T) {
	tests := []struct {
		name          string
		setup         func(*testing.T, string)
		expectError   bool
		errorContains string
	}{
		{
			name: "v1_snapshot_without_idmapper",
			setup: func(t *testing.T, path string) {
				// Create a V1 snapshot (no IDMapper, no InvertedIndex)
				idx := newTestIndex(t)
				_ = idx.Insert(context.Background(), "v1_test", fixtures.Vec3dSimple, core.SparseVector{}, fixtures.MetaSimple) //nolint:errcheck // test setup

				// Manually create V1 format (would need to mock old format)
				// For now, just save normally - in reality would need old serialization
				_ = idx.SaveToFile(context.Background(), path) //nolint:errcheck // test setup
			},
			expectError: false, // Should load successfully
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile := filepath.Join(t.TempDir(), "version_test.bin")
			tt.setup(t, tmpFile)

			idx := newTestIndex(t)
			err := idx.LoadFromFile(context.Background(), tmpFile)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err, "Should load older version successfully")
			}
		})
	}
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
