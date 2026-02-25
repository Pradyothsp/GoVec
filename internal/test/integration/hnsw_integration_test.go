package integration

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// =============================================================================
// Package-level helper
// =============================================================================

// newHNSWEngine creates a fresh HNSW engine backed by a new WAL at dir/test.wal.
// The caller is responsible for closing the returned WAL.
func newHNSWEngine(t testing.TB, dir string, q config.Quantization) (index.Engine, *index.WAL) {
	t.Helper()
	walPath := dir + "/test.wal"
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("newHNSWEngine: failed to create WAL: %v", err)
	}
	cfg := config.EngineConfig{
		Quantization:        q,
		DistanceMetric:      "cosine",
		IndexType:           config.IndexTypeHNSW,
		EnableMetadataIndex: true,
	}
	engine, err := index.NewEngine(cfg, wal)
	if err != nil {
		t.Fatalf("newHNSWEngine: failed to create engine: %v", err)
	}
	return engine, wal
}

// =============================================================================
// Suite base (shared test methods for both quant variants)
// =============================================================================

type hnswSuiteBase struct {
	suite.Suite
	engine   index.Engine
	wal      *index.WAL
	walPath  string
	snapPath string
	tmpDir   string
	quant    config.Quantization
}

func (s *hnswSuiteBase) TearDownTest() {
	if s.wal != nil {
		_ = s.wal.Close() //nolint:errcheck // test cleanup
		s.wal = nil
	}
}

// =============================================================================
// HNSWFloat32IntegrationSuite  (QuantizationNone)
// =============================================================================

// HNSWFloat32IntegrationSuite tests the full HNSW engine wired with float32 storage.
type HNSWFloat32IntegrationSuite struct {
	hnswSuiteBase
}

func (s *HNSWFloat32IntegrationSuite) SetupTest() {
	s.tmpDir = s.T().TempDir()
	s.walPath = s.tmpDir + "/test.wal"
	s.snapPath = s.tmpDir + "/test.bin"
	s.quant = config.QuantizationNone
	s.engine, s.wal = newHNSWEngine(s.T(), s.tmpDir, config.QuantizationNone)
}

// =============================================================================
// HNSWScalarIntegrationSuite  (QuantizationScalar)
// =============================================================================

// HNSWScalarIntegrationSuite tests the full HNSW engine wired with int8 scalar quantization.
type HNSWScalarIntegrationSuite struct {
	hnswSuiteBase
}

func (s *HNSWScalarIntegrationSuite) SetupTest() {
	s.tmpDir = s.T().TempDir()
	s.walPath = s.tmpDir + "/test.wal"
	s.snapPath = s.tmpDir + "/test.bin"
	s.quant = config.QuantizationScalar
	s.engine, s.wal = newHNSWEngine(s.T(), s.tmpDir, config.QuantizationScalar)
}

// =============================================================================
// Suite runners
// =============================================================================

func TestHNSWFloat32Integration(t *testing.T) {
	suite.Run(t, new(HNSWFloat32IntegrationSuite))
}

func TestHNSWScalarIntegration(t *testing.T) {
	suite.Run(t, new(HNSWScalarIntegrationSuite))
}

// =============================================================================
// Group A — Insert / Search / Delete (4 tests)
// =============================================================================

func (s *hnswSuiteBase) TestHNSW_InsertSearchDelete_BasicFlow() {
	vectors := generateTestVectors(5, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	s.Equal(5, s.engine.Len())

	results, err := s.engine.Search(vectors[0], core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 1)
	s.Equal("v0", results[0].ID)

	deleted, err := s.engine.Delete("v0")
	s.Require().NoError(err)
	s.True(deleted)
	s.Equal(4, s.engine.Len())

	results2, err := s.engine.Search(vectors[0], core.SparseVector{}, 5, nil)
	s.Require().NoError(err)
	for _, r := range results2 {
		s.NotEqual("v0", r.ID, "deleted vector must not appear in search results")
	}
}

func (s *hnswSuiteBase) TestHNSW_Insert_Overwrite_ViaEngine() {
	v1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}

	s.Require().NoError(s.engine.Insert("doc1", v1, core.SparseVector{}, map[string]any{"version": "v1"}))
	s.Equal(1, s.engine.Len())

	// Overwrite same ID — count must stay at 1
	s.Require().NoError(s.engine.Insert("doc1", v2, core.SparseVector{}, map[string]any{"version": "v2"}))
	s.Equal(1, s.engine.Len(), "overwrite must not increase Len")

	// Metadata must reflect the latest insert
	results, err := s.engine.Search(v2, core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 1)
	s.Equal("doc1", results[0].ID)
	s.Equal("v2", results[0].Meta["version"])
}

func (s *hnswSuiteBase) TestHNSW_Delete_NonExistentID() {
	deleted, err := s.engine.Delete("ghost-id")
	s.Require().NoError(err, "deleting a non-existent ID must not error")
	s.False(deleted, "must return false for missing ID")
	s.Equal(0, s.engine.Len(), "Len must remain unchanged")
}

func (s *hnswSuiteBase) TestHNSW_Search_EmptyIndex_ReturnsNil() {
	results, err := s.engine.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, 5, nil)
	s.Require().NoError(err)
	s.Nil(results, "empty index must return nil")
}

// =============================================================================
// Group B — Metadata Filtering (3 tests)
// =============================================================================

func (s *hnswSuiteBase) TestHNSW_MetadataFilter_SelectiveCategory() {
	vectors := generateTestVectors(50, 8)
	rareIndices := map[int]bool{5: true, 15: true, 25: true, 35: true, 45: true}

	for i, v := range vectors {
		cat := "common"
		if rareIndices[i] {
			cat = "rare"
		}
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{},
			map[string]any{"category": cat}))
	}

	results, err := s.engine.Search(vectors[5], core.SparseVector{}, 10, map[string]any{"category": "rare"})
	s.Require().NoError(err)
	for _, r := range results {
		s.Equal("rare", r.Meta["category"], "filter must exclude common-category docs")
	}
	s.LessOrEqual(len(results), 5, "at most 5 rare docs exist")
}

func (s *hnswSuiteBase) TestHNSW_MetadataFilter_NoMatchReturnsEmpty() {
	vectors := generateTestVectors(5, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{},
			map[string]any{"category": "common"}))
	}

	results, err := s.engine.Search(vectors[0], core.SparseVector{}, 5, map[string]any{"category": "nonexistent"})
	s.Require().NoError(err)
	s.Nil(results, "filter matching nothing must return nil (short-circuit)")
}

func (s *hnswSuiteBase) TestHNSW_MetadataFilter_MultipleFields() {
	type doc struct {
		id       string
		vec      []float32
		docType  string
		priority string
	}
	docs := []doc{
		{"v0", []float32{1, 0, 0, 0, 0, 0, 0, 0}, "A", "high"},
		{"v1", []float32{0.9, 0.1, 0, 0, 0, 0, 0, 0}, "A", "high"},
		{"v2", []float32{0, 1, 0, 0, 0, 0, 0, 0}, "A", "low"},
		{"v3", []float32{0, 0, 1, 0, 0, 0, 0, 0}, "B", "high"},
		{"v4", []float32{0, 0, 0, 1, 0, 0, 0, 0}, "B", "low"},
		{"v5", []float32{0.8, 0.2, 0, 0, 0, 0, 0, 0}, "A", "high"},
	}
	for _, d := range docs {
		s.Require().NoError(s.engine.Insert(d.id, d.vec, core.SparseVector{},
			map[string]any{"type": d.docType, "priority": d.priority}))
	}

	// Filter on both fields: type=A AND priority=high → v0, v1, v5
	results, err := s.engine.Search(
		[]float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, 10,
		map[string]any{"type": "A", "priority": "high"},
	)
	s.Require().NoError(err)
	for _, r := range results {
		s.Equal("A", r.Meta["type"], "type filter must be applied")
		s.Equal("high", r.Meta["priority"], "priority filter must be applied")
	}
	s.LessOrEqual(len(results), 3, "at most 3 docs satisfy both filters")
}

// =============================================================================
// Group C — Persistence (3 tests)
// =============================================================================

func (s *hnswSuiteBase) TestHNSW_Persistence_SaveAndLoad() {
	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	vectors := generateTestVectors(10, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}

	before, err := s.engine.Search(query, core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(before, 1)
	expectedTopID := before[0].ID

	s.Require().NoError(s.engine.SaveToFile(s.snapPath))

	// Atomic rename must have cleaned up the .tmp file
	_, statErr := os.Stat(s.snapPath + ".tmp")
	s.True(os.IsNotExist(statErr), ".tmp file must be removed after successful save")

	// Load into a fresh engine and verify top result is preserved
	freshDir := s.T().TempDir()
	freshEngine, freshWal := newHNSWEngine(s.T(), freshDir, s.quant)
	defer freshWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(freshEngine.LoadFromFile(s.snapPath))
	s.Equal(10, freshEngine.Len())

	after, err := freshEngine.Search(query, core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(after, 1)
	s.Equal(expectedTopID, after[0].ID, "top result must be the same after reload")
}

func (s *hnswSuiteBase) TestHNSW_Persistence_MetadataRestoredAfterLoad() {
	vectors := generateTestVectors(10, 8)
	for i, v := range vectors {
		parity := "even"
		if i%2 != 0 {
			parity = "odd"
		}
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{},
			map[string]any{"parity": parity}))
	}

	s.Require().NoError(s.engine.SaveToFile(s.snapPath))

	// Load into fresh engine — rebuildMetaIndex() is exercised during LoadFromFile
	freshDir := s.T().TempDir()
	freshEngine, freshWal := newHNSWEngine(s.T(), freshDir, s.quant)
	defer freshWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(freshEngine.LoadFromFile(s.snapPath))
	s.Equal(10, freshEngine.Len())

	// Filtered search must work, proving metaIndex was rebuilt from the loaded metadata
	results, err := freshEngine.Search(vectors[0], core.SparseVector{}, 10, map[string]any{"parity": "even"})
	s.Require().NoError(err)
	for _, r := range results {
		s.Equal("even", r.Meta["parity"], "metadata filter must work after reload (metaIndex rebuilt)")
	}
}

func (s *hnswSuiteBase) TestHNSW_Persistence_QuantizationMismatchError() {
	s.Require().NoError(s.engine.Insert("v1", []float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, nil))
	s.Require().NoError(s.engine.SaveToFile(s.snapPath))

	oppositeQuant := config.QuantizationScalar
	if s.quant == config.QuantizationScalar {
		oppositeQuant = config.QuantizationNone
	}

	oppDir := s.T().TempDir()
	oppositeEngine, oppositeWal := newHNSWEngine(s.T(), oppDir, oppositeQuant)
	defer oppositeWal.Close() //nolint:errcheck // test cleanup

	err := oppositeEngine.LoadFromFile(s.snapPath)
	s.Require().Error(err, "loading with mismatched quantization must return an error")
	s.Contains(err.Error(), "quantization", "error must mention quantization mismatch")
}

// =============================================================================
// Group D — WAL Recovery (4 tests)
// =============================================================================

func (s *hnswSuiteBase) TestHNSW_WALRecovery_CrashAfterInserts() {
	vectors := generateTestVectors(10, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}

	// Simulate crash: close WAL without saving snapshot
	s.Require().NoError(s.wal.Close())
	s.wal = nil // TearDownTest skips nil WAL

	// Recovery engine with a fresh WAL at a different path
	recoveryDir := s.T().TempDir()
	recoveryEngine, recoveryWal := newHNSWEngine(s.T(), recoveryDir, s.quant)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	// No snapshot was written — LoadFromFile on missing path is a no-op
	s.Require().NoError(recoveryEngine.LoadFromFile(s.snapPath))
	s.Equal(0, recoveryEngine.Len(), "index is empty before WAL replay")

	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(10, recoveryEngine.Len(), "all 10 inserts must be recovered from WAL")
}

func (s *hnswSuiteBase) TestHNSW_WALRecovery_SnapshotPlusWAL() {
	// Insert 20 vectors, then checkpoint (SaveToFile clears the WAL)
	base := generateTestVectors(20, 8)
	for i, v := range base {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("base%d", i), v, core.SparseVector{}, nil))
	}
	s.Require().NoError(s.engine.SaveToFile(s.snapPath)) // clears WAL internally

	// Insert 5 more — these go only to WAL (post-checkpoint)
	extra := generateTestVectors(5, 8)
	for i, v := range extra {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("extra%d", i), v, core.SparseVector{}, nil))
	}

	// Simulate crash
	s.Require().NoError(s.wal.Close())
	s.wal = nil

	// Recovery: snapshot (20) + WAL replay (5) = 25
	recoveryDir := s.T().TempDir()
	recoveryEngine, recoveryWal := newHNSWEngine(s.T(), recoveryDir, s.quant)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(recoveryEngine.LoadFromFile(s.snapPath))
	s.Equal(20, recoveryEngine.Len(), "20 vectors loaded from snapshot")

	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(25, recoveryEngine.Len(), "snapshot(20) + WAL(5) = 25 after full recovery")
}

func (s *hnswSuiteBase) TestHNSW_WALRecovery_DeleteIsPreserved() {
	v1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	v3 := []float32{0, 0, 1, 0, 0, 0, 0, 0}

	s.Require().NoError(s.engine.Insert("v1", v1, core.SparseVector{}, nil))
	s.Require().NoError(s.engine.Insert("v2", v2, core.SparseVector{}, nil))
	s.Require().NoError(s.engine.Insert("v3", v3, core.SparseVector{}, nil))

	deleted, err := s.engine.Delete("v2")
	s.Require().NoError(err)
	s.True(deleted)

	// Simulate crash
	s.Require().NoError(s.wal.Close())
	s.wal = nil

	recoveryDir := s.T().TempDir()
	recoveryEngine, recoveryWal := newHNSWEngine(s.T(), recoveryDir, s.quant)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(2, recoveryEngine.Len(), "v2 was deleted in WAL; only v1 and v3 must survive")

	results, err := recoveryEngine.Search(v2, core.SparseVector{}, 5, nil)
	s.Require().NoError(err)
	for _, r := range results {
		s.NotEqual("v2", r.ID, "deleted vector must not appear after WAL replay")
	}
}

func (s *hnswSuiteBase) TestHNSW_WALRecovery_CorruptWALSkipsEntry() {
	vectors := generateTestVectors(5, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}

	// Close WAL to flush writes, then append a corrupt JSON line
	s.Require().NoError(s.wal.Close())
	s.wal = nil

	f, err := os.OpenFile(s.walPath, os.O_APPEND|os.O_WRONLY, 0o644) //nolint:gosec // test-controlled path
	s.Require().NoError(err)
	_, writeErr := f.Write([]byte("{this is: not valid json\n"))
	s.Require().NoError(writeErr)
	s.Require().NoError(f.Close())

	recoveryDir := s.T().TempDir()
	recoveryEngine, recoveryWal := newHNSWEngine(s.T(), recoveryDir, s.quant)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	// Corrupt line is skipped with a warning; valid entries are recovered
	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(5, recoveryEngine.Len(), "valid entries must be recovered despite corrupt WAL line")
}

// =============================================================================
// Group E — Scalar-specific (HNSWScalarIntegrationSuite only)
// =============================================================================

func (s *HNSWScalarIntegrationSuite) TestHNSW_Scalar_SearchAccuracy_VsFloat32() {
	dim := 64
	numVecs := 100
	vectors := generateTestVectors(numVecs, dim)

	float32Dir := s.T().TempDir()
	float32Engine, float32Wal := newHNSWEngine(s.T(), float32Dir, config.QuantizationNone)
	defer float32Wal.Close() //nolint:errcheck // test cleanup

	for i, v := range vectors {
		id := fmt.Sprintf("v%d", i)
		s.Require().NoError(s.engine.Insert(id, v, core.SparseVector{}, nil))
		s.Require().NoError(float32Engine.Insert(id, v, core.SparseVector{}, nil))
	}

	query := vectors[0]
	topK := 3

	scalarResults, err := s.engine.Search(query, core.SparseVector{}, topK, nil)
	s.Require().NoError(err)
	s.Require().Len(scalarResults, topK)

	float32Results, err := float32Engine.Search(query, core.SparseVector{}, topK, nil)
	s.Require().NoError(err)
	s.Require().Len(float32Results, topK)

	float32Set := make(map[string]float32, topK)
	for _, r := range float32Results {
		float32Set[r.ID] = r.Score
	}

	overlap := 0
	for _, r := range scalarResults {
		if f32Score, ok := float32Set[r.ID]; ok {
			overlap++
			scoreDiff := math.Abs(float64(r.Score - f32Score))
			s.Less(scoreDiff, 0.05, "score diff for %s must be < 0.05 (scalar precision loss)", r.ID)
		}
	}
	s.GreaterOrEqual(overlap, 2, "top-%d overlap between scalar and float32 must be >= 2", topK)
}

// =============================================================================
// Group F — InvertedIndex + Sparse + Metadata type safety (3 tests)
// These run on both suites through base-struct promotion.
// =============================================================================

// TestHNSW_InvertedIndex_SaveLoad_HybridSearchRestored verifies that SaveToFile
// correctly serialises the InvertedIndex (when EnableHybridSearch=true) and
// LoadFromFile correctly restores it, so hybrid search produces the expected
// ranking after a reload.
func (s *hnswSuiteBase) TestHNSW_InvertedIndex_SaveLoad_HybridSearchRestored() {
	dir := s.T().TempDir()
	walPath := dir + "/hybrid.wal"
	snapPath := dir + "/hybrid.bin"

	wal, err := index.NewWAL(walPath)
	s.Require().NoError(err)

	cfg := config.EngineConfig{
		Quantization:       s.quant,
		DistanceMetric:     "cosine",
		IndexType:          config.IndexTypeHNSW,
		EnableHybridSearch: true,
	}
	engine, err := index.NewEngine(cfg, wal)
	s.Require().NoError(err)

	// doc1: strong dense AND strong sparse on token 10 → must rank first in hybrid search
	s.Require().NoError(engine.Insert("doc1", []float32{1, 0, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{5.0},
	}, nil))
	// doc2: slightly weaker dense, no sparse
	s.Require().NoError(engine.Insert("doc2", []float32{0.9, 0.1, 0, 0}, core.SparseVector{}, nil))
	// doc3: orthogonal, no sparse
	s.Require().NoError(engine.Insert("doc3", []float32{0, 1, 0, 0}, core.SparseVector{}, nil))

	s.Require().NoError(engine.SaveToFile(snapPath))
	s.Require().NoError(wal.Close())

	// Load into fresh engine that also has hybrid search enabled
	recoveryWal, err := index.NewWAL(dir + "/recovery.wal")
	s.Require().NoError(err)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	recoveryEngine, err := index.NewEngine(cfg, recoveryWal)
	s.Require().NoError(err)

	s.Require().NoError(recoveryEngine.LoadFromFile(snapPath))
	s.Equal(3, recoveryEngine.Len())

	// Hybrid search: sparse query on token 10 should boost doc1 to the top
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{3.0}}
	results, err := recoveryEngine.Search([]float32{1, 0, 0, 0}, sparseQuery, 3, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 3)
	s.Equal("doc1", results[0].ID,
		"doc1 must rank first after reload: InvertedIndex correctly persisted and restored")
}

// TestHNSW_WALReplay_SparseVectors_RebuildInvertedIndex verifies that ReplayWAL
// correctly rebuilds the InvertedIndex from sparse-vector WAL entries, so hybrid
// search works correctly on an engine recovered from WAL alone (no snapshot).
func (s *hnswSuiteBase) TestHNSW_WALReplay_SparseVectors_RebuildInvertedIndex() {
	dir := s.T().TempDir()
	walPath := dir + "/sparse.wal"

	wal, err := index.NewWAL(walPath)
	s.Require().NoError(err)

	cfg := config.EngineConfig{
		Quantization:       s.quant,
		DistanceMetric:     "cosine",
		IndexType:          config.IndexTypeHNSW,
		EnableHybridSearch: true,
	}
	engine, err := index.NewEngine(cfg, wal)
	s.Require().NoError(err)

	// Insert with sparse vectors — these are written to WAL
	s.Require().NoError(engine.Insert("doc1", []float32{1, 0, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{5.0},
	}, nil))
	s.Require().NoError(engine.Insert("doc2", []float32{0.9, 0.1, 0, 0}, core.SparseVector{}, nil))
	s.Require().NoError(engine.Insert("doc3", []float32{0, 1, 0, 0}, core.SparseVector{}, nil))

	// Simulate crash: close WAL without SaveToFile
	s.Require().NoError(wal.Close())

	// Recovery engine with hybrid search enabled
	recoveryDir := s.T().TempDir()
	recoveryWal, err := index.NewWAL(recoveryDir + "/recovery.wal")
	s.Require().NoError(err)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	recoveryEngine, err := index.NewEngine(cfg, recoveryWal)
	s.Require().NoError(err)

	s.Require().NoError(recoveryEngine.ReplayWAL(walPath))
	s.Equal(3, recoveryEngine.Len())

	// Hybrid search must work — InvertedIndex rebuilt by insertInternal during replay
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{3.0}}
	results, err := recoveryEngine.Search([]float32{1, 0, 0, 0}, sparseQuery, 3, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 3)
	s.Equal("doc1", results[0].ID,
		"doc1 must rank first: sparse InvertedIndex rebuilt correctly during WAL replay")
}

// TestHNSW_Metadata_GOBRoundTrip_PreservesTypes verifies that all common metadata
// value types (string, float64, bool, int, nested map) survive the GOB
// encode → decode cycle without type corruption.
func (s *hnswSuiteBase) TestHNSW_Metadata_GOBRoundTrip_PreservesTypes() {
	meta := map[string]any{
		"label":  "hello world",
		"score":  float64(42.5),
		"active": true,
		"count":  int(7),
		"nested": map[string]any{"key": "value"},
	}
	s.Require().NoError(s.engine.Insert("doc1",
		[]float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, meta))

	s.Require().NoError(s.engine.SaveToFile(s.snapPath))

	freshDir := s.T().TempDir()
	freshEngine, freshWal := newHNSWEngine(s.T(), freshDir, s.quant)
	defer freshWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(freshEngine.LoadFromFile(s.snapPath))
	s.Equal(1, freshEngine.Len())

	results, err := freshEngine.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 1)
	loaded := results[0].Meta

	s.Equal("hello world", loaded["label"])
	s.Equal(float64(42.5), loaded["score"])
	s.Equal(true, loaded["active"])

	// Nested map must decode back as map[string]any (requires gob.Register(map[string]any{}))
	nestedRaw, ok := loaded["nested"]
	s.Require().True(ok, "nested key must survive round-trip")
	nested, ok := nestedRaw.(map[string]any)
	s.Require().True(ok, "nested value must decode as map[string]any, got %T", nestedRaw)
	s.Equal("value", nested["key"])

	// int in interface{} — gob encodes built-in numeric types natively;
	// if this fails it means we need gob.Register(int(0)) in persistence.go init()
	s.Equal(int(7), loaded["count"],
		"int metadata must round-trip without type change via GOB")
}

func (s *HNSWScalarIntegrationSuite) TestHNSW_Scalar_PersistenceRoundTrip() {
	dim := 64
	numVecs := 100
	vectors := generateTestVectors(numVecs, dim)

	// Save scalar snapshot
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	s.Require().NoError(s.engine.SaveToFile(s.snapPath))

	// Build equivalent float32 snapshot
	float32Dir := s.T().TempDir()
	float32Engine, float32Wal := newHNSWEngine(s.T(), float32Dir, config.QuantizationNone)
	defer float32Wal.Close() //nolint:errcheck // test cleanup

	for i, v := range vectors {
		s.Require().NoError(float32Engine.Insert(fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	float32SnapPath := float32Dir + "/float32.bin"
	s.Require().NoError(float32Engine.SaveToFile(float32SnapPath))

	scalarInfo, err := os.Stat(s.snapPath)
	s.Require().NoError(err)
	float32Info, err := os.Stat(float32SnapPath)
	s.Require().NoError(err)

	s.Less(scalarInfo.Size(), float32Info.Size(),
		"scalar snapshot (%d bytes) must be smaller than float32 snapshot (%d bytes)",
		scalarInfo.Size(), float32Info.Size())
}
