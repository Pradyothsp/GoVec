package integration

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// quantEngine builds an engine of indexType storing vectors as quant, logging
// to the WAL at walPath.
func quantEngine(t *testing.T, indexType config.IndexType, quant config.Quantization, walPath string) index.FileEngine {
	t.Helper()
	wal, err := index.NewWAL(walPath)
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })
	engine, err := index.NewEngine(config.EngineConfig{
		IndexType:      indexType,
		Quantization:   quant,
		DistanceMetric: config.DistanceMetricCosine,
	}, config.StorageConfig{}, wal)
	require.NoError(t, err)
	return engine
}

func insertAll(t *testing.T, engine index.FileEngine, vectors [][]float32) {
	t.Helper()
	for i, v := range vectors {
		require.NoError(t, engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
}

var indexTypes = []config.IndexType{config.IndexTypeBrute, config.IndexTypeHNSW}

// TestScalarQuantization_MatchesFloat32 checks that int8 storage approximates
// float32 storage on the same data. k covers the whole index, so HNSW returns
// every vector and the comparison is of quantization alone, not graph recall
// (the quantization benchmarks measure that).
func TestScalarQuantization_MatchesFloat32(t *testing.T) {
	for _, indexType := range indexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			// Arrange
			ctx := context.Background()
			vectors := generateTestVectors(100, 16)
			float32Engine := quantEngine(t, indexType, config.QuantizationNone, filepath.Join(t.TempDir(), "f32.wal"))
			scalarEngine := quantEngine(t, indexType, config.QuantizationScalar, filepath.Join(t.TempDir(), "int8.wal"))
			insertAll(t, float32Engine, vectors)
			insertAll(t, scalarEngine, vectors)

			// Act
			float32Results, err := float32Engine.Search(ctx, vectors[0], core.SparseVector{}, len(vectors), nil)
			require.NoError(t, err)
			scalarResults, err := scalarEngine.Search(ctx, vectors[0], core.SparseVector{}, len(vectors), nil)
			require.NoError(t, err)

			// Assert
			assertScalarMatchesFloat32(t, float32Results, scalarResults)
		})
	}
}

// scalarScoreTolerance is how far an int8 score may drift from its float32
// score. Observed drift on these low-dimension vectors is about 0.005; a broken
// quantizer is off by far more.
const scalarScoreTolerance = 0.02

// assertScalarMatchesFloat32 checks that int8 search approximates float32
// search: the same IDs, each scored within scalarScoreTolerance, in the same
// order wherever float32 separates them by more than that drift could flip.
func assertScalarMatchesFloat32(t *testing.T, float32Results, scalarResults []index.SearchResult) {
	t.Helper()
	require.Len(t, scalarResults, len(float32Results))

	// Match by ID, not rank, so a reordered near-tie can't skip a comparison.
	scalarScore := make(map[string]float32, len(scalarResults))
	for _, r := range scalarResults {
		scalarScore[r.ID] = r.Score
	}
	for _, r := range float32Results {
		s, ok := scalarScore[r.ID]
		if !ok {
			t.Errorf("%s is in the float32 results but not the scalar results", r.ID)
			continue
		}
		if diff := math.Abs(float64(r.Score - s)); diff > scalarScoreTolerance {
			t.Errorf("%s: scalar score %.4f is %.4f from float32 score %.4f", r.ID, s, diff, r.Score)
		}
	}

	// Two errors of up to the tolerance each can flip a pair closer than twice it.
	for i := 0; i+1 < len(float32Results); i++ {
		hi, lo := float32Results[i], float32Results[i+1]
		if hi.Score-lo.Score > 2*scalarScoreTolerance && scalarScore[hi.ID] <= scalarScore[lo.ID] {
			t.Errorf("scalar ranks %s above %s, but float32 separates them by %.4f", lo.ID, hi.ID, hi.Score-lo.Score)
		}
	}
}

// TestScalarQuantization_SurvivesRestart: int8 vectors come back from a
// snapshot, and the WAL -- which holds the float32 originals -- re-encodes
// them on replay.
func TestScalarQuantization_SurvivesRestart(t *testing.T) {
	ctx := context.Background()
	restarts := map[string]func(t *testing.T, before, after index.FileEngine, walPath string){
		"from_snapshot": func(t *testing.T, before, after index.FileEngine, _ string) {
			snapPath := filepath.Join(t.TempDir(), "snapshot.bin")
			require.NoError(t, before.SaveToFile(ctx, snapPath))
			require.NoError(t, after.LoadFromFile(ctx, snapPath))
		},
		"from_wal": func(t *testing.T, _, after index.FileEngine, walPath string) {
			require.NoError(t, after.ReplayWAL(walPath))
		},
	}
	for _, indexType := range indexTypes {
		for name, restart := range restarts {
			t.Run(string(indexType)+"/"+name, func(t *testing.T) {
				// Arrange
				vectors := generateTestVectors(20, 16)
				walPath := filepath.Join(t.TempDir(), "before.wal")
				before := quantEngine(t, indexType, config.QuantizationScalar, walPath)
				insertAll(t, before, vectors)
				after := quantEngine(t, indexType, config.QuantizationScalar, filepath.Join(t.TempDir(), "after.wal"))

				// Act
				restart(t, before, after, walPath)

				// Assert
				require.Equal(t, len(vectors), after.Len())
				for _, i := range []int{0, 7, 19} {
					results, err := after.Search(ctx, vectors[i], core.SparseVector{}, 1, nil)
					require.NoError(t, err)
					require.Len(t, results, 1)
					assert.Equal(t, fmt.Sprintf("v%d", i), results[0].ID, "each vector must still find itself")
				}
			})
		}
	}
}

// TestScalarQuantization_SnapshotIsSmaller: int8 is the point of scalar
// quantization, so its snapshot must hold about a byte per dimension where
// float32 holds four.
func TestScalarQuantization_SnapshotIsSmaller(t *testing.T) {
	for _, indexType := range indexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			// Arrange
			vectors := generateTestVectors(100, 256)
			sizes := map[config.Quantization]int64{}

			// Act
			for _, quant := range []config.Quantization{config.QuantizationNone, config.QuantizationScalar} {
				engine := quantEngine(t, indexType, quant, filepath.Join(t.TempDir(), "test.wal"))
				insertAll(t, engine, vectors)
				snapPath := filepath.Join(t.TempDir(), "snapshot.bin")
				require.NoError(t, engine.SaveToFile(context.Background(), snapPath))
				info, err := os.Stat(snapPath)
				require.NoError(t, err)
				sizes[quant] = info.Size()
			}

			// Assert
			ratio := float64(sizes[config.QuantizationScalar]) / float64(sizes[config.QuantizationNone])
			assert.Less(t, ratio, 0.35, "int8 snapshot is %d bytes, float32 %d", sizes[config.QuantizationScalar], sizes[config.QuantizationNone])
		})
	}
}

// TestLoadFromFile_QuantizationMismatch_Rejected: a snapshot's vectors are
// stored in its quantization's format, so loading it under the other must be
// refused by the header check -- with an error that says what is wrong --
// rather than fail later in decoding or load garbage.
func TestLoadFromFile_QuantizationMismatch_Rejected(t *testing.T) {
	ctx := context.Background()
	pairs := []struct{ saved, loaded config.Quantization }{
		{config.QuantizationNone, config.QuantizationScalar},
		{config.QuantizationScalar, config.QuantizationNone},
	}
	for _, indexType := range indexTypes {
		for _, p := range pairs {
			t.Run(fmt.Sprintf("%s/%s_into_%s", indexType, p.saved, p.loaded), func(t *testing.T) {
				// Arrange
				saved := quantEngine(t, indexType, p.saved, filepath.Join(t.TempDir(), "saved.wal"))
				require.NoError(t, saved.Insert(ctx, "v1", []float32{0.5, 0.3, 0.8}, core.SparseVector{}, nil))
				snapPath := filepath.Join(t.TempDir(), "snapshot.bin")
				require.NoError(t, saved.SaveToFile(ctx, snapPath))
				loaded := quantEngine(t, indexType, p.loaded, filepath.Join(t.TempDir(), "loaded.wal"))

				// Act
				err := loaded.LoadFromFile(ctx, snapPath)

				// Assert
				require.ErrorContains(t, err, "quantization mismatch")
				assert.Zero(t, loaded.Len(), "a refused snapshot must leave the engine empty")
			})
		}
	}
}
