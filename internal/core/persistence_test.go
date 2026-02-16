package core

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// A. Basic Save/Load Functionality Tests
// =============================================================================

func TestSaveToFile_CreatesFileSuccessfully(t *testing.T) {
	index := NewVectorIndex()
	index.Insert("test1", []float32{1.0, 2.0, 3.0}, map[string]any{"label": "test"})

	tmpFile := filepath.Join(t.TempDir(), "test.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err, "SaveToFile should succeed")

	_, err = os.Stat(tmpFile)
	assert.NoError(t, err, "File should exist after save")
}

func TestSaveToFile_EmptyIndex(t *testing.T) {
	index := NewVectorIndex()
	tmpFile := filepath.Join(t.TempDir(), "empty.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err, "Should save empty index without error")

	_, err = os.Stat(tmpFile)
	assert.NoError(t, err, "File should exist even for empty index")
}

func TestSaveToFile_PopulatedIndex(t *testing.T) {
	index := NewVectorIndex()

	// Insert multiple vectors
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("vec%d", i)
		vec := []float32{float32(i), float32(i * 2)}
		meta := map[string]any{"index": i, "batch": "test"}
		index.Insert(id, vec, meta)
	}

	tmpFile := filepath.Join(t.TempDir(), "populated.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err, "SaveToFile should succeed")

	info, err := os.Stat(tmpFile)
	require.NoError(t, err, "File should exist")
	assert.Greater(t, info.Size(), int64(0), "File should have content")
}

func TestLoadFromFile_ReadsSavedDataCorrectly(t *testing.T) {
	index := NewVectorIndex()
	index.Insert("v1", []float32{1.0, 2.0}, map[string]any{"label": "first"})
	index.Insert("v2", []float32{3.0, 4.0}, nil)

	tmpFile := filepath.Join(t.TempDir(), "load_test.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Create new index and load
	newIndex := NewVectorIndex()
	err = newIndex.LoadFromFile(tmpFile)
	require.NoError(t, err, "LoadFromFile should succeed")

	assert.Len(t, newIndex.Store, 2, "Should load all vectors")
	assert.Contains(t, newIndex.Store, "v1")
	assert.Contains(t, newIndex.Store, "v2")
	assert.Equal(t, []float32{1.0, 2.0}, newIndex.Store["v1"].Vector)
	assert.Equal(t, []float32{3.0, 4.0}, newIndex.Store["v2"].Vector)
	assert.Equal(t, map[string]any{"label": "first"}, newIndex.Store["v1"].Metadata)
	assert.Nil(t, newIndex.Store["v2"].Metadata)
}

func TestLoadFromFile_MissingFile(t *testing.T) {
	index := NewVectorIndex()

	err := index.LoadFromFile("/nonexistent/path/file.bin")

	// Should return nil error (graceful handling for missing files)
	assert.NoError(t, err, "Missing file should be handled gracefully")
	assert.Empty(t, index.Store, "Index should remain empty")
}

func TestRoundTrip_Preservation(t *testing.T) {
	// Create index with diverse data
	original := NewVectorIndex()
	original.Insert("vec1", []float32{1.0, 2.0, 3.0}, map[string]any{
		"string": "value",
		"number": 42,
		"bool":   true,
	})
	original.Insert("vec2", []float32{4.0, 5.0}, nil)
	original.Insert("vec3", []float32{6.0}, map[string]any{
		"nested": map[string]any{
			"level1": "deep",
		},
	})

	tmpFile := filepath.Join(t.TempDir(), "roundtrip.bin")

	// Save
	err := original.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Load into new index
	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	// Verify exact match
	assert.Equal(t, len(original.Store), len(loaded.Store), "Vector count should match")

	for id, originalNode := range original.Store {
		loadedNode, exists := loaded.Store[id]
		require.True(t, exists, "Vector %s should exist", id)
		assert.Equal(t, originalNode.ID, loadedNode.ID, "ID should match for %s", id)
		assert.Equal(t, originalNode.Vector, loadedNode.Vector, "Vector should match for %s", id)
		assert.Equal(t, originalNode.Metadata, loadedNode.Metadata, "Metadata should match for %s", id)
	}
}

// =============================================================================
// B. Data Integrity Tests
// =============================================================================

func TestComplexNestedMetadata_Persistence(t *testing.T) {
	index := NewVectorIndex()

	complexMeta := map[string]any{
		"nested": map[string]any{
			"level1": map[string]any{
				"level2": map[string]any{
					"level3": "deep_value",
					"number": 123,
				},
			},
		},
		"array":       []any{1, "two", true, 3.14},
		"mixed_types": []any{"string", 42, false},
		"empty_map":   map[string]any{},
		// Note: empty_array is omitted because GOB encodes empty slices as nil
	}

	index.Insert("complex", []float32{1.0, 2.0}, complexMeta)

	tmpFile := filepath.Join(t.TempDir(), "complex_meta.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	assert.Contains(t, loaded.Store, "complex")

	// Verify nested structure
	loadedMeta := loaded.Store["complex"].Metadata
	assert.Equal(t, complexMeta["array"], loadedMeta["array"])
	assert.Equal(t, complexMeta["mixed_types"], loadedMeta["mixed_types"])
	assert.Equal(t, complexMeta["empty_map"], loadedMeta["empty_map"])

	// Verify nested map structure
	nested := loadedMeta["nested"].(map[string]any)
	level1 := nested["level1"].(map[string]any)
	level2 := level1["level2"].(map[string]any)
	assert.Equal(t, "deep_value", level2["level3"])
	assert.Equal(t, 123, level2["number"])
}

func TestLargeVectors_PersistCorrectly(t *testing.T) {
	testCases := []struct {
		name       string
		dimensions int
	}{
		{"1536D_embedding", 1536},
		{"4096D_embedding", 4096},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			index := NewVectorIndex()

			vec := make([]float32, tc.dimensions)
			for i := 0; i < tc.dimensions; i++ {
				vec[i] = float32(i) * 0.001
			}

			index.Insert("large_vec", vec, map[string]any{"dimensions": tc.dimensions})

			tmpFile := filepath.Join(t.TempDir(), fmt.Sprintf("%s.bin", tc.name))

			err := index.SaveToFile(tmpFile)
			require.NoError(t, err)

			loaded := NewVectorIndex()
			err = loaded.LoadFromFile(tmpFile)
			require.NoError(t, err)

			assert.Contains(t, loaded.Store, "large_vec")
			assert.Equal(t, vec, loaded.Store["large_vec"].Vector)
			assert.Len(t, loaded.Store["large_vec"].Vector, tc.dimensions)
		})
	}
}

func TestEmptyVectors_Persist(t *testing.T) {
	index := NewVectorIndex()
	index.Insert("empty", []float32{}, map[string]any{"type": "empty"})

	tmpFile := filepath.Join(t.TempDir(), "empty_vec.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	assert.Contains(t, loaded.Store, "empty")
	assert.Empty(t, loaded.Store["empty"].Vector)
	assert.Equal(t, map[string]any{"type": "empty"}, loaded.Store["empty"].Metadata)
}

func TestSpecialCharactersInIDs_Persist(t *testing.T) {
	testIDs := []string{
		"vec-with-dash",
		"vec_with_underscore",
		"vec.with.dots",
		"vec/with/slashes",
		"vec with spaces",
		"vec🔥emoji",
		"vec中文",
		"vec@special#chars$%",
	}

	index := NewVectorIndex()
	for _, id := range testIDs {
		index.Insert(id, []float32{1.0, 2.0}, map[string]any{"id": id})
	}

	tmpFile := filepath.Join(t.TempDir(), "special_ids.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	for _, id := range testIDs {
		assert.Contains(t, loaded.Store, id, "ID %s should persist", id)
		assert.Equal(t, id, loaded.Store[id].ID)
	}
}

func TestVariousMetadataTypes_Persist(t *testing.T) {
	index := NewVectorIndex()

	testCases := []struct {
		id   string
		meta map[string]any
	}{
		{"string_meta", map[string]any{"type": "string", "value": "test"}},
		{"int_meta", map[string]any{"type": "int", "value": 42}},
		{"float_meta", map[string]any{"type": "float", "value": 3.14}},
		{"bool_meta", map[string]any{"type": "bool", "value": true}},
		{"array_meta", map[string]any{"type": "array", "value": []any{1, 2, 3}}},
		{"nested_meta", map[string]any{"type": "nested", "value": map[string]any{"inner": "data"}}},
		{"nil_meta", nil},
	}

	for _, tc := range testCases {
		index.Insert(tc.id, []float32{1.0}, tc.meta)
	}

	tmpFile := filepath.Join(t.TempDir(), "various_meta.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	for _, tc := range testCases {
		assert.Contains(t, loaded.Store, tc.id)
		assert.Equal(t, tc.meta, loaded.Store[tc.id].Metadata)
	}
}

func TestLargeNumberOfVectors_Performance(t *testing.T) {
	index := NewVectorIndex()

	// Insert 1000 vectors
	numVectors := 1000
	for i := 0; i < numVectors; i++ {
		id := fmt.Sprintf("vec%d", i)
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		meta := map[string]any{"index": i, "batch": "performance_test"}
		index.Insert(id, vec, meta)
	}

	tmpFile := filepath.Join(t.TempDir(), "large_dataset.bin")

	// Test save
	err := index.SaveToFile(tmpFile)
	require.NoError(t, err, "Should save large dataset")

	// Test load
	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err, "Should load large dataset")

	assert.Len(t, loaded.Store, numVectors, "All vectors should be loaded")

	// Spot check some vectors
	assert.Equal(t, []float32{0.0, 0.0, 0.0}, loaded.Store["vec0"].Vector)
	assert.Equal(t, []float32{500.0, 1000.0, 1500.0}, loaded.Store["vec500"].Vector)
	assert.Equal(t, []float32{999.0, 1998.0, 2997.0}, loaded.Store["vec999"].Vector)
}

func TestNilMetadata_Handling(t *testing.T) {
	index := NewVectorIndex()
	index.Insert("no_meta", []float32{1.0, 2.0, 3.0}, nil)

	tmpFile := filepath.Join(t.TempDir(), "nil_meta.bin")

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	assert.Contains(t, loaded.Store, "no_meta")
	assert.Nil(t, loaded.Store["no_meta"].Metadata)
}

// =============================================================================
// C. File Operation Tests
// =============================================================================

func TestSaveToFile_OverwritesExistingFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "overwrite.bin")

	// Save first version
	index1 := NewVectorIndex()
	index1.Insert("v1", []float32{1.0, 2.0}, map[string]any{"version": 1})
	err := index1.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Save second version (overwrite)
	index2 := NewVectorIndex()
	index2.Insert("v2", []float32{3.0, 4.0}, map[string]any{"version": 2})
	err = index2.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Load and verify only second version exists
	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	assert.Len(t, loaded.Store, 1, "Should only have second version")
	assert.Contains(t, loaded.Store, "v2")
	assert.NotContains(t, loaded.Store, "v1")
}

func TestSaveToFile_InvalidPath(t *testing.T) {
	index := NewVectorIndex()
	index.Insert("test", []float32{1.0}, nil)

	// Try to save to invalid path
	err := index.SaveToFile("/invalid/nonexistent/path/file.bin")
	assert.Error(t, err, "Should return error for invalid path")
}

func TestLoadFromFile_CorruptedFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "corrupted.bin")

	// Create corrupted file (not valid GOB data)
	err := os.WriteFile(tmpFile, []byte("this is not GOB data"), 0644)
	require.NoError(t, err)

	index := NewVectorIndex()
	err = index.LoadFromFile(tmpFile)

	assert.Error(t, err, "Should return error for corrupted file")
}

func TestLoadFromFile_NonGOBFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "text.bin")

	// Create text file
	err := os.WriteFile(tmpFile, []byte("Hello, this is a text file!\nWith multiple lines."), 0644)
	require.NoError(t, err)

	index := NewVectorIndex()
	err = index.LoadFromFile(tmpFile)

	assert.Error(t, err, "Should fail gracefully for non-GOB file")
}

func TestTempFileCleanup_OnSuccess(t *testing.T) {
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "cleanup_test.bin")
	tempFile := tmpFile + ".tmp"

	index := NewVectorIndex()
	index.Insert("test", []float32{1.0}, nil)

	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Verify .tmp file is removed
	_, err = os.Stat(tempFile)
	assert.True(t, os.IsNotExist(err), "Temp file should be cleaned up")

	// Verify main file exists
	_, err = os.Stat(tmpFile)
	assert.NoError(t, err, "Main file should exist")
}

// =============================================================================
// D. Concurrency & Thread Safety Tests
// =============================================================================

func TestConcurrentInsert_AndSave(t *testing.T) {
	index := NewVectorIndex()
	tmpFile := filepath.Join(t.TempDir(), "concurrent_insert_save.bin")

	var wg sync.WaitGroup
	numInserts := 50

	// Start concurrent inserts
	for i := 0; i < numInserts; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vecID := fmt.Sprintf("vec%d", id)
			vec := []float32{float32(id), float32(id * 2)}
			meta := map[string]any{"index": id}
			index.Insert(vecID, vec, meta)
		}(i)
	}

	// Concurrent save operations
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := index.SaveToFile(tmpFile)
		assert.NoError(t, err, "Save should succeed during concurrent inserts")
	}()

	wg.Wait()

	// Verify all inserts completed
	assert.LessOrEqual(t, numInserts, len(index.Store), "Most or all inserts should succeed")
}

func TestConcurrentSearch_AndSave(t *testing.T) {
	index := NewVectorIndex()

	// Pre-populate index with non-zero vectors
	for i := 1; i <= 100; i++ { // Start from 1 to avoid zero vectors
		id := fmt.Sprintf("vec%d", i)
		vec := []float32{float32(i), float32(i * 2), float32(i * 3)}
		index.Insert(id, vec, nil)
	}

	tmpFile := filepath.Join(t.TempDir(), "concurrent_search_save.bin")

	var wg sync.WaitGroup
	numSearches := 50

	// Start concurrent searches (reads)
	for i := 1; i <= numSearches; i++ { // Start from 1 to avoid zero query
		wg.Add(1)
		go func(queryID int) {
			defer wg.Done()
			query := []float32{float32(queryID), float32(queryID * 2), float32(queryID * 3)}
			_, err := index.Search(query, 5)
			assert.NoError(t, err, "Search should succeed during save")
		}(i)
	}

	// Concurrent save
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := index.SaveToFile(tmpFile)
		assert.NoError(t, err, "Save should succeed during concurrent searches")
	}()

	wg.Wait()

	// Verify file was created
	_, err := os.Stat(tmpFile)
	assert.NoError(t, err, "Save file should exist")
}

func TestMultipleSaveToFile_Calls(t *testing.T) {
	index := NewVectorIndex()
	index.Insert("test", []float32{1.0, 2.0}, nil)

	tmpFile := filepath.Join(t.TempDir(), "multi_save.bin")

	var wg sync.WaitGroup
	numSaves := 5

	// Multiple concurrent saves
	for i := 0; i < numSaves; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := index.SaveToFile(tmpFile)
			assert.NoError(t, err, "Concurrent saves should not fail")
		}()
	}

	wg.Wait()

	// Verify final file is valid
	loaded := NewVectorIndex()
	err := loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)
	assert.Contains(t, loaded.Store, "test")
}

// =============================================================================
// E. Edge Cases
// =============================================================================

func TestSaveThenLoad_EmptyIndex(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "empty_roundtrip.bin")

	// Save empty index
	index := NewVectorIndex()
	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Load into new index
	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	assert.Empty(t, loaded.Store, "Loaded index should be empty")
}

func TestOverwriteThenPersist(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "overwrite_persist.bin")

	index := NewVectorIndex()

	// Insert initial value
	index.Insert("vec", []float32{1.0, 2.0}, map[string]any{"version": 1.0})

	// Overwrite with updated value
	index.Insert("vec", []float32{3.0, 4.0, 5.0}, map[string]any{"version": 2.0})

	// Save
	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Load and verify updated value
	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)

	assert.Len(t, loaded.Store, 1)
	assert.Equal(t, []float32{3.0, 4.0, 5.0}, loaded.Store["vec"].Vector)
	assert.Equal(t, map[string]any{"version": 2.0}, loaded.Store["vec"].Metadata)
}

func TestVeryLongFilePaths(t *testing.T) {
	// Create nested directory structure
	tmpDir := t.TempDir()
	longPath := tmpDir
	for i := 0; i < 10; i++ {
		longPath = filepath.Join(longPath, fmt.Sprintf("very_long_directory_name_%d", i))
	}

	err := os.MkdirAll(longPath, 0755)
	require.NoError(t, err, "Should create nested directories")

	tmpFile := filepath.Join(longPath, "test.bin")

	index := NewVectorIndex()
	index.Insert("test", []float32{1.0}, nil)

	err = index.SaveToFile(tmpFile)
	require.NoError(t, err, "Should handle long paths")

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)
	assert.Contains(t, loaded.Store, "test")
}

func TestSpecialCharactersInFilePath(t *testing.T) {
	tmpDir := t.TempDir()

	// Test spaces in directory name
	dirWithSpaces := filepath.Join(tmpDir, "dir with spaces")
	err := os.MkdirAll(dirWithSpaces, 0755)
	require.NoError(t, err)

	tmpFile := filepath.Join(dirWithSpaces, "test file.bin")

	index := NewVectorIndex()
	index.Insert("test", []float32{1.0}, nil)

	err = index.SaveToFile(tmpFile)
	require.NoError(t, err, "Should handle spaces in path")

	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)
	assert.Contains(t, loaded.Store, "test")
}

func TestAtomicWrite_Verification(t *testing.T) {
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "atomic.bin")

	index := NewVectorIndex()
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("vec%d", i)
		index.Insert(id, []float32{float32(i)}, nil)
	}

	// Save should use atomic write (.tmp file then rename)
	err := index.SaveToFile(tmpFile)
	require.NoError(t, err)

	// Main file should exist
	_, err = os.Stat(tmpFile)
	assert.NoError(t, err, "Main file should exist")

	// Temp file should be cleaned up
	tempFile := tmpFile + ".tmp"
	_, err = os.Stat(tempFile)
	assert.True(t, os.IsNotExist(err), "Temp file should not exist after successful save")

	// Verify data integrity
	loaded := NewVectorIndex()
	err = loaded.LoadFromFile(tmpFile)
	require.NoError(t, err)
	assert.Len(t, loaded.Store, 100)
}
