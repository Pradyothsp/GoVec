package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"

	"github.com/Pradyothsp/govec/internal/api"
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
	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
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
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	s.Equal(5, s.engine.Len())

	results, err := s.engine.Search(context.Background(), vectors[0], core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 1)
	s.Equal("v0", results[0].ID)

	deleted, err := s.engine.Delete(context.Background(), "v0")
	s.Require().NoError(err)
	s.True(deleted)
	s.Equal(4, s.engine.Len())

	results2, err := s.engine.Search(context.Background(), vectors[0], core.SparseVector{}, 5, nil)
	s.Require().NoError(err)
	for _, r := range results2 {
		s.NotEqual("v0", r.ID, "deleted vector must not appear in search results")
	}
}

func (s *hnswSuiteBase) TestHNSW_Insert_Overwrite_ViaEngine() {
	v1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}

	s.Require().NoError(s.engine.Insert(context.Background(), "doc1", v1, core.SparseVector{}, map[string]any{"version": "v1"}))
	s.Equal(1, s.engine.Len())

	// Overwrite same ID — count must stay at 1
	s.Require().NoError(s.engine.Insert(context.Background(), "doc1", v2, core.SparseVector{}, map[string]any{"version": "v2"}))
	s.Equal(1, s.engine.Len(), "overwrite must not increase Len")

	// Metadata must reflect the latest insert
	results, err := s.engine.Search(context.Background(), v2, core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 1)
	s.Equal("doc1", results[0].ID)
	s.Equal("v2", results[0].Meta["version"])
}

func (s *hnswSuiteBase) TestHNSW_Delete_NonExistentID() {
	deleted, err := s.engine.Delete(context.Background(), "ghost-id")
	s.Require().NoError(err, "deleting a non-existent ID must not error")
	s.False(deleted, "must return false for missing ID")
	s.Equal(0, s.engine.Len(), "Len must remain unchanged")
}

func (s *hnswSuiteBase) TestHNSW_Search_EmptyIndex_ReturnsNil() {
	results, err := s.engine.Search(context.Background(), []float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, 5, nil)
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
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{},
			map[string]any{"category": cat}))
	}

	results, err := s.engine.Search(context.Background(), vectors[5], core.SparseVector{}, 10, map[string]any{"category": "rare"})
	s.Require().NoError(err)
	for _, r := range results {
		s.Equal("rare", r.Meta["category"], "filter must exclude common-category docs")
	}
	s.LessOrEqual(len(results), 5, "at most 5 rare docs exist")
}

func (s *hnswSuiteBase) TestHNSW_MetadataFilter_NoMatchReturnsEmpty() {
	vectors := generateTestVectors(5, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{},
			map[string]any{"category": "common"}))
	}

	results, err := s.engine.Search(context.Background(), vectors[0], core.SparseVector{}, 5, map[string]any{"category": "nonexistent"})
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
		s.Require().NoError(s.engine.Insert(context.Background(), d.id, d.vec, core.SparseVector{},
			map[string]any{"type": d.docType, "priority": d.priority}))
	}

	// Filter on both fields: type=A AND priority=high → v0, v1, v5
	results, err := s.engine.Search(context.Background(),
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
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}

	before, err := s.engine.Search(context.Background(), query, core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().Len(before, 1)
	expectedTopID := before[0].ID

	s.Require().NoError(s.engine.SaveToFile(context.Background(), s.snapPath))

	// Atomic rename must have cleaned up the .tmp file
	_, statErr := os.Stat(s.snapPath + ".tmp")
	s.True(os.IsNotExist(statErr), ".tmp file must be removed after successful save")

	// Load into a fresh engine and verify top result is preserved
	freshDir := s.T().TempDir()
	freshEngine, freshWal := newHNSWEngine(s.T(), freshDir, s.quant)
	defer freshWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(freshEngine.LoadFromFile(context.Background(), s.snapPath))
	s.Equal(10, freshEngine.Len())

	after, err := freshEngine.Search(context.Background(), query, core.SparseVector{}, 1, nil)
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
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{},
			map[string]any{"parity": parity}))
	}

	s.Require().NoError(s.engine.SaveToFile(context.Background(), s.snapPath))

	// Load into fresh engine — rebuildMetaIndex() is exercised during LoadFromFile
	freshDir := s.T().TempDir()
	freshEngine, freshWal := newHNSWEngine(s.T(), freshDir, s.quant)
	defer freshWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(freshEngine.LoadFromFile(context.Background(), s.snapPath))
	s.Equal(10, freshEngine.Len())

	// Filtered search must work, proving metaIndex was rebuilt from the loaded metadata
	results, err := freshEngine.Search(context.Background(), vectors[0], core.SparseVector{}, 10, map[string]any{"parity": "even"})
	s.Require().NoError(err)
	for _, r := range results {
		s.Equal("even", r.Meta["parity"], "metadata filter must work after reload (metaIndex rebuilt)")
	}
}

func (s *hnswSuiteBase) TestHNSW_Persistence_QuantizationMismatchError() {
	s.Require().NoError(s.engine.Insert(context.Background(), "v1", []float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, nil))
	s.Require().NoError(s.engine.SaveToFile(context.Background(), s.snapPath))

	oppositeQuant := config.QuantizationScalar
	if s.quant == config.QuantizationScalar {
		oppositeQuant = config.QuantizationNone
	}

	oppDir := s.T().TempDir()
	oppositeEngine, oppositeWal := newHNSWEngine(s.T(), oppDir, oppositeQuant)
	defer oppositeWal.Close() //nolint:errcheck // test cleanup

	err := oppositeEngine.LoadFromFile(context.Background(), s.snapPath)
	s.Require().Error(err, "loading with mismatched quantization must return an error")
	s.Contains(err.Error(), "quantization", "error must mention quantization mismatch")
}

// =============================================================================
// Group D — WAL Recovery (4 tests)
// =============================================================================

func (s *hnswSuiteBase) TestHNSW_WALRecovery_CrashAfterInserts() {
	vectors := generateTestVectors(10, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}

	// Simulate crash: close WAL without saving snapshot
	s.Require().NoError(s.wal.Close())
	s.wal = nil // TearDownTest skips nil WAL

	// Recovery engine with a fresh WAL at a different path
	recoveryDir := s.T().TempDir()
	recoveryEngine, recoveryWal := newHNSWEngine(s.T(), recoveryDir, s.quant)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	// No snapshot was written — LoadFromFile on missing path is a no-op
	s.Require().NoError(recoveryEngine.LoadFromFile(context.Background(), s.snapPath))
	s.Equal(0, recoveryEngine.Len(), "index is empty before WAL replay")

	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(10, recoveryEngine.Len(), "all 10 inserts must be recovered from WAL")
}

func (s *hnswSuiteBase) TestHNSW_WALRecovery_SnapshotPlusWAL() {
	// Insert 20 vectors, then checkpoint (SaveToFile clears the WAL)
	base := generateTestVectors(20, 8)
	for i, v := range base {
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("base%d", i), v, core.SparseVector{}, nil))
	}
	s.Require().NoError(s.engine.SaveToFile(context.Background(), s.snapPath)) // clears WAL internally

	// Insert 5 more — these go only to WAL (post-checkpoint)
	extra := generateTestVectors(5, 8)
	for i, v := range extra {
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("extra%d", i), v, core.SparseVector{}, nil))
	}

	// Simulate crash
	s.Require().NoError(s.wal.Close())
	s.wal = nil

	// Recovery: snapshot (20) + WAL replay (5) = 25
	recoveryDir := s.T().TempDir()
	recoveryEngine, recoveryWal := newHNSWEngine(s.T(), recoveryDir, s.quant)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(recoveryEngine.LoadFromFile(context.Background(), s.snapPath))
	s.Equal(20, recoveryEngine.Len(), "20 vectors loaded from snapshot")

	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(25, recoveryEngine.Len(), "snapshot(20) + WAL(5) = 25 after full recovery")
}

func (s *hnswSuiteBase) TestHNSW_WALRecovery_DeleteIsPreserved() {
	v1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	v3 := []float32{0, 0, 1, 0, 0, 0, 0, 0}

	s.Require().NoError(s.engine.Insert(context.Background(), "v1", v1, core.SparseVector{}, nil))
	s.Require().NoError(s.engine.Insert(context.Background(), "v2", v2, core.SparseVector{}, nil))
	s.Require().NoError(s.engine.Insert(context.Background(), "v3", v3, core.SparseVector{}, nil))

	deleted, err := s.engine.Delete(context.Background(), "v2")
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

	results, err := recoveryEngine.Search(context.Background(), v2, core.SparseVector{}, 5, nil)
	s.Require().NoError(err)
	for _, r := range results {
		s.NotEqual("v2", r.ID, "deleted vector must not appear after WAL replay")
	}
}

// A crash mid-write leaves a bad last line. Recovery cuts it off and keeps
// every entry before it; a bad line anywhere else stops recovery (see
// internal/index/wal_replay_policy_test.go).
func (s *hnswSuiteBase) TestHNSW_WALRecovery_TornLastLineIsCutOff() {
	vectors := generateTestVectors(5, 8)
	for i, v := range vectors {
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}

	// Close WAL to flush writes, then append a torn last line
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

	s.Require().NoError(recoveryEngine.ReplayWAL(s.walPath))
	s.Equal(5, recoveryEngine.Len(), "every entry before the torn line must be recovered")
	data, err := os.ReadFile(s.walPath)
	s.Require().NoError(err)
	s.NotContains(string(data), "not valid json", "the torn line must be cut off the WAL")
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
		s.Require().NoError(s.engine.Insert(context.Background(), id, v, core.SparseVector{}, nil))
		s.Require().NoError(float32Engine.Insert(context.Background(), id, v, core.SparseVector{}, nil))
	}

	query := vectors[0]
	topK := 3

	scalarResults, err := s.engine.Search(context.Background(), query, core.SparseVector{}, topK, nil)
	s.Require().NoError(err)
	s.Require().Len(scalarResults, topK)

	float32Results, err := float32Engine.Search(context.Background(), query, core.SparseVector{}, topK, nil)
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

// TestHNSW_SaveLoad_HybridRankingSurvives verifies that each document's sparse
// vector survives SaveToFile/LoadFromFile, so hybrid search ranks the same
// after a reload. Sparse scores come from the stored sparse vectors, not the
// InvertedIndex postings, which nothing reads yet.
func (s *hnswSuiteBase) TestHNSW_SaveLoad_HybridRankingSurvives() {
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
	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	s.Require().NoError(err)

	// doc1: close dense match with a strong sparse hit on token 10
	s.Require().NoError(engine.Insert(context.Background(), "doc1", []float32{1, 0, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{5.0},
	}, nil))
	// doc2: the exact dense match for the query below, no sparse
	s.Require().NoError(engine.Insert(context.Background(), "doc2", []float32{0.9, 0.1, 0, 0}, core.SparseVector{}, nil))
	// doc3: orthogonal, no sparse
	s.Require().NoError(engine.Insert(context.Background(), "doc3", []float32{0, 1, 0, 0}, core.SparseVector{}, nil))

	s.Require().NoError(engine.SaveToFile(context.Background(), snapPath))
	s.Require().NoError(wal.Close())

	// Load into fresh engine that also has hybrid search enabled
	recoveryWal, err := index.NewWAL(dir + "/recovery.wal")
	s.Require().NoError(err)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	recoveryEngine, err := index.NewEngine(cfg, config.StorageConfig{}, recoveryWal)
	s.Require().NoError(err)

	s.Require().NoError(recoveryEngine.LoadFromFile(context.Background(), snapPath))
	s.Equal(3, recoveryEngine.Len())

	// The dense query is doc2's own vector, so doc2 wins on dense score alone.
	// doc1 can only rank first if its sparse vector survived the reload.
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{3.0}}
	results, err := recoveryEngine.Search(context.Background(), []float32{0.9, 0.1, 0, 0}, sparseQuery, 3, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 3)
	s.Equal("doc1", results[0].ID,
		"doc1's sparse vector must survive the reload and outrank doc2's dense match")
}

// TestHNSW_WALReplay_HybridRankingSurvives verifies that ReplayWAL restores
// each document's sparse vector from its WAL entry, so hybrid search ranks the
// same on an engine recovered from the WAL alone (no snapshot).
func (s *hnswSuiteBase) TestHNSW_WALReplay_HybridRankingSurvives() {
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
	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	s.Require().NoError(err)

	// Insert with sparse vectors — these are written to WAL
	s.Require().NoError(engine.Insert(context.Background(), "doc1", []float32{1, 0, 0, 0}, core.SparseVector{
		Indices: []uint32{10}, Values: []float32{5.0},
	}, nil))
	s.Require().NoError(engine.Insert(context.Background(), "doc2", []float32{0.9, 0.1, 0, 0}, core.SparseVector{}, nil))
	s.Require().NoError(engine.Insert(context.Background(), "doc3", []float32{0, 1, 0, 0}, core.SparseVector{}, nil))

	// Simulate crash: close WAL without SaveToFile
	s.Require().NoError(wal.Close())

	// Recovery engine with hybrid search enabled
	recoveryDir := s.T().TempDir()
	recoveryWal, err := index.NewWAL(recoveryDir + "/recovery.wal")
	s.Require().NoError(err)
	defer recoveryWal.Close() //nolint:errcheck // test cleanup

	recoveryEngine, err := index.NewEngine(cfg, config.StorageConfig{}, recoveryWal)
	s.Require().NoError(err)

	s.Require().NoError(recoveryEngine.ReplayWAL(walPath))
	s.Equal(3, recoveryEngine.Len())

	// The dense query is doc2's own vector, so doc2 wins on dense score alone.
	// doc1 can only rank first if replay restored its sparse vector.
	sparseQuery := core.SparseVector{Indices: []uint32{10}, Values: []float32{3.0}}
	results, err := recoveryEngine.Search(context.Background(), []float32{0.9, 0.1, 0, 0}, sparseQuery, 3, nil)
	s.Require().NoError(err)
	s.Require().Len(results, 3)
	s.Equal("doc1", results[0].ID,
		"replay must restore doc1's sparse vector so it outranks doc2's dense match")
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
	s.Require().NoError(s.engine.Insert(context.Background(), "doc1",
		[]float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, meta))

	s.Require().NoError(s.engine.SaveToFile(context.Background(), s.snapPath))

	freshDir := s.T().TempDir()
	freshEngine, freshWal := newHNSWEngine(s.T(), freshDir, s.quant)
	defer freshWal.Close() //nolint:errcheck // test cleanup

	s.Require().NoError(freshEngine.LoadFromFile(context.Background(), s.snapPath))
	s.Equal(1, freshEngine.Len())

	results, err := freshEngine.Search(context.Background(), []float32{1, 0, 0, 0, 0, 0, 0, 0}, core.SparseVector{}, 1, nil)
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
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	s.Require().NoError(s.engine.SaveToFile(context.Background(), s.snapPath))

	// Build equivalent float32 snapshot
	float32Dir := s.T().TempDir()
	float32Engine, float32Wal := newHNSWEngine(s.T(), float32Dir, config.QuantizationNone)
	defer float32Wal.Close() //nolint:errcheck // test cleanup

	for i, v := range vectors {
		s.Require().NoError(float32Engine.Insert(context.Background(), fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	float32SnapPath := float32Dir + "/float32.bin"
	s.Require().NoError(float32Engine.SaveToFile(context.Background(), float32SnapPath))

	scalarInfo, err := os.Stat(s.snapPath)
	s.Require().NoError(err)
	float32Info, err := os.Stat(float32SnapPath)
	s.Require().NoError(err)

	s.Less(scalarInfo.Size(), float32Info.Size(),
		"scalar snapshot (%d bytes) must be smaller than float32 snapshot (%d bytes)",
		scalarInfo.Size(), float32Info.Size())
}

// =============================================================================
// Group G — Batch insert / round-based parallelism
// Covers HNSWIndex.BatchInsert's restructuring (a single batched
// graph.Add call instead of one per item -- see internal/index/hnsw_index.go)
// and the graph-level round-based parallel Prepare/Apply it now feeds
// (internal/hnsw/graph.go's addRound). These run on both suites through
// base-struct promotion.
// =============================================================================

// TestHNSW_BatchInsert_IntraBatchDuplicateID_LastWins verifies that when the
// same external ID appears more than once within a single BatchInsert call,
// only the last occurrence's data survives -- the documented, deliberate
// "silent last-wins dedup, replicating today's exact net behavior" decision
// (see HNSWIndex.BatchInsert's doc comment). This must hold regardless of
// batch size / parallelism, since HNSWIndex.bookkeepInsert publishes
// idx.metadata immediately per item specifically so this stays correct even
// though the actual graph.Add call is deferred and batched.
func (s *hnswSuiteBase) TestHNSW_BatchInsert_IntraBatchDuplicateID_LastWins() {
	dim := 8
	first := make([]float32, dim)
	first[0] = 1
	last := make([]float32, dim)
	last[1] = 1

	items := []index.BatchInsertItem{
		{ID: "dup", Vector: first, Meta: map[string]interface{}{"version": "first"}},
		{ID: "other", Vector: []float32{0, 0, 1, 0, 0, 0, 0, 0}},
		{ID: "dup", Vector: last, Meta: map[string]interface{}{"version": "last"}},
	}

	failures, err := s.engine.BatchInsert(context.Background(), items)
	s.Require().NoError(err)
	s.Require().Empty(failures)

	// last-wins: exactly 2 distinct keys survive, not 3.
	s.Equal(2, s.engine.Len())

	rec, err := s.engine.GetByID(context.Background(), "dup")
	s.Require().NoError(err)
	s.Equal("last", rec.Metadata["version"])
	s.Equal(last, rec.Vector)
}

// TestHNSW_BatchInsert_MixedNewAndUpsert covers a batch that mixes brand-new
// inserts with upserts of vectors that already existed before the batch
// started. Upserts are excluded from Graph.Add's round-based parallel path
// (see addRound's doc comment: bookkeepInsert's upsert pre-delete would
// otherwise invalidate an already-computed round-mate plan) and routed
// through the graph's serial fallback instead -- this exercises that split
// end-to-end through HNSWIndex, not just at the Graph level (already covered
// by internal/hnsw's own tests).
func (s *hnswSuiteBase) TestHNSW_BatchInsert_MixedNewAndUpsert() {
	dim := 8
	const preexisting = 20

	for i := 0; i < preexisting; i++ {
		v := make([]float32, dim)
		v[i%dim] = 1
		s.Require().NoError(s.engine.Insert(context.Background(), fmt.Sprintf("existing-%d", i), v, core.SparseVector{}, nil))
	}
	s.Require().Equal(preexisting, s.engine.Len())

	items := make([]index.BatchInsertItem, 0, preexisting+preexisting)
	// Upsert every pre-existing vector with a rotated one-hot vector.
	for i := 0; i < preexisting; i++ {
		v := make([]float32, dim)
		v[(i+1)%dim] = 1
		items = append(items, index.BatchInsertItem{ID: fmt.Sprintf("existing-%d", i), Vector: v})
	}
	// Plus an equal number of brand-new keys in the same batch.
	for i := 0; i < preexisting; i++ {
		v := make([]float32, dim)
		v[i%dim] = -1
		items = append(items, index.BatchInsertItem{ID: fmt.Sprintf("new-%d", i), Vector: v})
	}

	failures, err := s.engine.BatchInsert(context.Background(), items)
	s.Require().NoError(err)
	s.Require().Empty(failures)

	s.Equal(2*preexisting, s.engine.Len(), "upserts must not create duplicates, new keys must all land")

	for i := 0; i < preexisting; i++ {
		rec, err := s.engine.GetByID(context.Background(), fmt.Sprintf("existing-%d", i))
		s.Require().NoError(err)
		want := make([]float32, dim)
		want[(i+1)%dim] = 1
		s.Equal(want, rec.Vector, "existing-%d must reflect the upserted vector, not the original", i)
	}
	for i := 0; i < preexisting; i++ {
		_, err := s.engine.GetByID(context.Background(), fmt.Sprintf("new-%d", i))
		s.Require().NoError(err)
	}
}

// TestHNSW_BatchInsert_LargeAboveThreshold_ViaREST sends a single batch
// larger than hnsw.DefaultBatchParallelThreshold (20) through the actual
// REST layer (POST /api/v1/vectors/batch), so Graph.Add's round-based
// parallel path is genuinely exercised end-to-end -- HTTP handler -> engine
// -> HNSWIndex.BatchInsert -> Graph.Add -- not just at the graph or engine
// layer in isolation.
func (s *hnswSuiteBase) TestHNSW_BatchInsert_LargeAboveThreshold_ViaREST() {
	gin.SetMode(gin.TestMode)

	dir := s.T().TempDir()
	engine, wal := newHNSWEngine(s.T(), dir, s.quant)
	defer wal.Close() //nolint:errcheck // test cleanup

	router := api.SetupRouter(engine, "", dir+"/test.bin")

	const dim, n = 8, 120 // above DefaultBatchParallelThreshold
	vectors := generateTestVectors(n, dim)

	payloadVectors := make([]map[string]interface{}, n)
	for i, v := range vectors {
		payloadVectors[i] = map[string]interface{}{
			"id":     fmt.Sprintf("rv%d", i),
			"vector": v,
		}
	}
	body, err := json.Marshal(map[string]interface{}{"vectors": payloadVectors})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	s.Require().Equal(http.StatusOK, w.Code, w.Body.String())

	var env struct {
		Success bool `json:"success"`
		Data    struct {
			InsertedCount int `json:"inserted_count"`
		} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.True(env.Success)
	s.Equal(n, env.Data.InsertedCount)
	s.Equal(n, engine.Len())

	// The graph built via the parallel path must still be genuinely
	// searchable -- every inserted vector should find itself as its own
	// nearest neighbor.
	for i := 0; i < n; i += 17 {
		results, err := engine.Search(context.Background(), vectors[i], core.SparseVector{}, 1, nil)
		s.Require().NoError(err)
		s.Require().Len(results, 1)
		s.Equal(fmt.Sprintf("rv%d", i), results[0].ID)
	}
}
