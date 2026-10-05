package index

import (
	"context"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
)

// The two engines share every rule about what a record is and how it's kept;
// they differ only in how they search. These tests hold both to the same
// behaviour, for rules that once drifted apart between two copies of the code.

var parityIndexTypes = []config.IndexType{config.IndexTypeBrute, config.IndexTypeHNSW}

func newParityEngine(t *testing.T, engineCfg config.EngineConfig, storageCfg config.StorageConfig) Engine {
	t.Helper()
	wal, err := NewWAL(filepath.Join(t.TempDir(), "parity.wal"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = wal.Close() })

	engine, err := NewEngine(engineCfg, storageCfg, wal)
	require.NoError(t, err)
	return engine
}

// TestInfo_ReportsMmap: HNSW's Info once left EnableMmap out, so /info said
// mmap was off for an HNSW index that had it on.
func TestInfo_ReportsMmap(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("mmap store is unix-only")
	}
	for _, indexType := range parityIndexTypes {
		for _, mmap := range []bool{false, true} {
			t.Run(string(indexType)+map[bool]string{false: "/heap", true: "/mmap"}[mmap], func(t *testing.T) {
				engine := newParityEngine(t,
					config.EngineConfig{IndexType: indexType, Quantization: config.QuantizationNone, DistanceMetric: config.DistanceMetricCosine, Dimensions: 3},
					config.StorageConfig{EnableMmap: mmap, MmapStorePath: filepath.Join(t.TempDir(), "mmap")})

				assert.Equal(t, mmap, engine.Info().EnableMmap)
			})
		}
	}
}

// TestInsert_Rejected_LeavesNoTrace: an insert that fails must change nothing.
// HNSW once learned the index's width from a vector it then rejected, so an
// empty index refused every other width; both engines left the rejected
// record's metadata and sparse postings indexed.
func TestInsert_Rejected_LeavesNoTrace(t *testing.T) {
	ctx := context.Background()
	for _, indexType := range parityIndexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			// Scalar quantization under Euclidean rejects values outside [-1, 1].
			engine := newParityEngine(t, config.EngineConfig{
				IndexType:           indexType,
				Quantization:        config.QuantizationScalar,
				DistanceMetric:      config.DistanceMetricEuclidean,
				EnableMetadataIndex: true,
				EnableHybridSearch:  true,
			}, config.StorageConfig{})
			tagged := map[string]any{"tag": "x"}
			sparse := core.SparseVector{Indices: []uint32{7}, Values: []float32{1}}

			err := engine.Insert(ctx, "rejected", []float32{5, 5, 5}, sparse, tagged)

			require.Error(t, err)
			assert.Zero(t, engine.Info().Dimensions, "a rejected vector must not set the width")
			assert.Zero(t, engine.Len())
			r := recordsOf(t, engine)
			assert.Empty(t, r.metaIndex.Allowlist(tagged), "a rejected record must not be in the metadata index")
			assert.Empty(t, r.InvertedIndex, "a rejected record must not be in the inverted index")

			require.NoError(t, engine.Insert(ctx, "accepted", []float32{0.5, 0.5}, sparse, tagged), "the empty index must take any width")
			results, err := engine.Search(ctx, []float32{0.5, 0.5}, core.SparseVector{}, 5, tagged)
			require.NoError(t, err)
			require.Len(t, results, 1)
			assert.Equal(t, "accepted", results[0].ID)
		})
	}
}

func recordsOf(t *testing.T, engine Engine) *records[[]int8] {
	t.Helper()
	switch e := engine.(type) {
	case *VectorIndex[[]int8]:
		return &e.records
	case *HNSWIndex[[]int8]:
		return &e.records
	default:
		t.Fatalf("unexpected engine %T", engine)
		return nil
	}
}

// TestSparseVector_LengthMismatch_Rejected: a sparse vector pairs each index
// with a value, so mismatched lengths are the caller's mistake, for an insert
// and for a query alike. The transports used to check this themselves, four
// times and with two different messages.
func TestSparseVector_LengthMismatch_Rejected(t *testing.T) {
	ctx := context.Background()
	mismatched := core.SparseVector{Indices: []uint32{0, 1}, Values: []float32{0.5}}
	for _, indexType := range parityIndexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			engine := newParityEngine(t, config.EngineConfig{
				IndexType: indexType, Quantization: config.QuantizationNone, DistanceMetric: config.DistanceMetricCosine, EnableHybridSearch: true,
			}, config.StorageConfig{})
			require.NoError(t, engine.Insert(ctx, "stored", []float32{1, 0, 0}, core.SparseVector{}, nil))

			insertErr := engine.Insert(ctx, "bad", []float32{0, 1, 0}, mismatched, nil)
			_, searchErr := engine.Search(ctx, []float32{1, 0, 0}, mismatched, 5, nil)

			for _, err := range []error{insertErr, searchErr} {
				require.ErrorIs(t, err, ErrInvalidSparseVector)
				assert.True(t, IsInvalidVectorError(err), "a caller error, answered with 400 / InvalidArgument")
			}
			assert.Equal(t, 1, engine.Len())
		})
	}
}

// TestSparseVector_Empty_MeansNone: an empty sparse vector is no sparse part,
// on every transport. REST used to reject one with 400 where gRPC took it.
func TestSparseVector_Empty_MeansNone(t *testing.T) {
	ctx := context.Background()
	empty := core.SparseVector{Indices: []uint32{}, Values: []float32{}}
	for _, indexType := range parityIndexTypes {
		t.Run(string(indexType), func(t *testing.T) {
			engine := newParityEngine(t, config.EngineConfig{
				IndexType: indexType, Quantization: config.QuantizationNone, DistanceMetric: config.DistanceMetricCosine, EnableHybridSearch: true,
			}, config.StorageConfig{})

			require.NoError(t, engine.Insert(ctx, "dense-only", []float32{1, 0, 0}, empty, nil))
			_, err := engine.Search(ctx, []float32{1, 0, 0}, empty, 5, nil)
			require.NoError(t, err)

			record, err := engine.GetByID(ctx, "dense-only")
			require.NoError(t, err)
			assert.Nil(t, record.SparseVector)
		})
	}
}

// TestRejectedWrite_NeverReachesTheWAL: an insert the index rejects must not
// be logged. Replay treats an intact record it can't apply as corruption, so a
// single rejected insert -- answered with a 400 -- used to stop the next
// restart until a snapshot happened to clear the WAL.
func TestRejectedWrite_NeverReachesTheWAL(t *testing.T) {
	ctx := context.Background()
	writes := map[string]func(Engine) error{
		"wrong width": func(e Engine) error {
			return e.Insert(ctx, "bad", []float32{1, 2}, core.SparseVector{}, nil)
		},
		"mismatched sparse vector": func(e Engine) error {
			return e.Insert(ctx, "bad", []float32{0, 1, 0}, core.SparseVector{Indices: []uint32{1, 2}, Values: []float32{1}}, nil)
		},
		"batch with one bad item": func(e Engine) error {
			failures, err := e.BatchInsert(ctx, []BatchInsertItem{{ID: "ok", Vector: []float32{0, 1, 0}}, {ID: "bad", Vector: []float32{1, 2}}})
			if err == nil && len(failures) != 1 {
				t.Errorf("want 1 failed item, got %d", len(failures))
			}
			return err
		},
		// The first item sets an empty index's width, so the second is too
		// narrow even though the index had no width when the batch began.
		"batch whose first item sets the width": func(e Engine) error {
			e.Clear()
			_, err := e.BatchInsert(ctx, []BatchInsertItem{{ID: "first", Vector: []float32{1, 0, 0}}, {ID: "bad", Vector: []float32{1, 2}}})
			return err
		},
	}
	for _, e := range bothEngines {
		for name, write := range writes {
			t.Run(e.name+"/"+name, func(t *testing.T) {
				// Arrange
				idx, r := e.open(t)
				require.NoError(t, idx.Insert(ctx, "good", []float32{1, 0, 0}, core.SparseVector{}, nil))

				// Act
				_ = write(idx) //nolint:errcheck // single inserts are expected to fail
				restarted, _ := e.open(t)
				replayErr := restarted.ReplayWAL(r.wal.file.Name())

				// Assert
				require.NoError(t, replayErr, "the WAL must replay after a rejected write")
				assert.Equal(t, idx.Len(), restarted.Len(), "replay must rebuild exactly what was accepted")
			})
		}
	}
}
