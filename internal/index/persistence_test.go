package index

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// TestSaveAndLoad_RoundTrip checks everything a snapshot must bring back:
// every record with its vector and metadata (of any type gob can carry), the
// ID mapper's counter and tombstones -- so an ID freed by a delete is never
// handed out again -- and indexes that search and filter as before.
func TestSaveAndLoad_RoundTrip(t *testing.T) {
	ctx := context.Background()
	records := []searchDoc{
		{id: "plain", vector: []float32{1, 0, 0}},
		{id: "special-chars_!@#", vector: []float32{0, 1, 0}, meta: map[string]any{"label": "alpha", "count": 42, "score": 3.14, "active": true}},
		{id: "unicode-测试", vector: []float32{0, 0, 1}, meta: map[string]any{
			"array":     []string{"a", "b"},
			"numbers":   []int{1, 2, 3},
			"nested":    map[string]any{"level2": map[string]any{"level3": "deep", "number": 42}},
			"nil_value": nil,
		}},
		{id: "spaces in id", vector: []float32{0.5, 0.5, 0.7}, meta: map[string]any{}},
	}
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, r := e.open(t)
			insertDocs(t, idx, records)
			require.NoError(t, idx.Insert(ctx, "deleted", []float32{1, 1, 0}, core.SparseVector{}, nil))
			_, err := idx.Delete(ctx, "deleted")
			require.NoError(t, err)
			path := filepath.Join(t.TempDir(), "snapshot.bin")
			require.NoError(t, idx.SaveToFile(ctx, path))
			loaded, lr := e.open(t)

			// Act
			err = loaded.LoadFromFile(ctx, path)

			// Assert
			require.NoError(t, err)
			assert.NoFileExists(t, path+".tmp", "the temporary file is renamed into place")
			assert.Equal(t, len(records), loaded.Len())
			assert.Equal(t, r.ids.Count(), lr.ids.Count())
			assert.Equal(t, r.ids.NextID(), lr.ids.NextID(), "a freed ID must not be handed out again")

			for _, want := range records {
				got, err := loaded.GetByID(ctx, want.id)
				require.NoError(t, err)
				assert.Equal(t, want.vector, got.Vector, want.id)
				assert.Equal(t, want.meta, got.Metadata, want.id)

				results, err := loaded.Search(ctx, want.vector, core.SparseVector{}, 1, nil)
				require.NoError(t, err)
				assert.Equal(t, want.id, results[0].ID, "each vector must still find itself")
			}
			_, err = loaded.GetByID(ctx, "deleted")
			assert.ErrorIs(t, err, core.ErrNotFound)

			for key, value := range records[1].meta {
				results, err := loaded.Search(ctx, []float32{1, 0, 0}, core.SparseVector{}, 5, map[string]any{key: value})
				require.NoError(t, err)
				assert.Equal(t, []string{records[1].id}, resultIDs(results), "filter on %s", key)
			}
		})
	}
}

func TestSaveAndLoad_EmptyIndex(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			path := filepath.Join(t.TempDir(), "snapshot.bin")
			require.NoError(t, idx.SaveToFile(context.Background(), path))
			loaded, _ := e.open(t)

			// Act
			err := loaded.LoadFromFile(context.Background(), path)

			// Assert
			require.NoError(t, err)
			assert.Zero(t, loaded.Len())
		})
	}
}

func TestLoadFromFile_Failures(t *testing.T) {
	tests := []struct {
		name    string
		path    func(t *testing.T) string
		wantErr string
	}{
		// A first start has no snapshot yet: that's an empty index, not an error.
		{"missing file starts empty", func(t *testing.T) string { return filepath.Join(t.TempDir(), "none.bin") }, ""},
		{"corrupt file", func(t *testing.T) string {
			path := filepath.Join(t.TempDir(), "corrupt.bin")
			require.NoError(t, os.WriteFile(path, []byte("not a snapshot"), 0o600))
			return path
		}, "failed to decode snapshot"},
	}
	for _, e := range bothEngines {
		for _, tt := range tests {
			t.Run(e.name+"/"+tt.name, func(t *testing.T) {
				// Arrange
				idx, _ := e.open(t)

				// Act
				err := idx.LoadFromFile(context.Background(), tt.path(t))

				// Assert
				if tt.wantErr == "" {
					require.NoError(t, err)
				} else {
					require.ErrorContains(t, err, tt.wantErr)
				}
				assert.Zero(t, idx.Len())
			})
		}
	}
}

func TestSaveToFile_UncreatablePath_Fails(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))

			// Act
			err := idx.SaveToFile(context.Background(), filepath.Join(uncreatableDir(t), "snapshot.bin"))

			// Assert
			assert.Error(t, err)
		})
	}
}

// A snapshot path whose directory doesn't exist yet must not make every save
// fail -- auto-save would log the same error every interval.
func TestSaveToFile_CreatesParentDirectories(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))
			path := filepath.Join(t.TempDir(), "nested", "deep", "snapshot.bin")

			// Act
			err := idx.SaveToFile(context.Background(), path)

			// Assert
			require.NoError(t, err)
			assert.FileExists(t, path)
		})
	}
}
