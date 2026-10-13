package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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
func newHNSWEngine(t testing.TB, dir string, q config.Quantization) (index.FileEngine, *index.WAL) {
	t.Helper()
	walPath := dir + "/test.wal"
	wal, err := index.NewWAL(walPath)
	if err != nil {
		t.Fatalf("newHNSWEngine: failed to create WAL: %v", err)
	}
	cfg := config.EngineConfig{
		Quantization:        q,
		DistanceMetric:      config.DistanceMetricCosine,
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
	engine   index.FileEngine
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
// Persistence and recovery, through the whole engine at each quantization
// =============================================================================

// TestHNSW_Persistence_RoundTrip checks what a snapshot must bring back for
// an HNSW engine: a graph that finds the same nearest neighbour, a metadata
// index that still filters, and metadata values of their original types --
// int included, and a nested map, which gob only decodes back as
// map[string]any because persistence registers it.
func (s *hnswSuiteBase) TestHNSW_Persistence_RoundTrip() {
	// Arrange
	ctx := context.Background()
	vectors := generateTestVectors(10, 8)
	for i, v := range vectors {
		meta := map[string]any{"parity": []string{"even", "odd"}[i%2]}
		if i == 0 {
			meta = map[string]any{"parity": "even", "label": "hello", "score": 42.5, "active": true, "count": 7, "nested": map[string]any{"key": "value"}}
		}
		s.Require().NoError(s.engine.Insert(ctx, fmt.Sprintf("v%d", i), v, core.SparseVector{}, meta))
	}
	before, err := s.engine.Search(ctx, vectors[3], core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Require().NoError(s.engine.SaveToFile(ctx, s.snapPath))
	loaded, loadedWAL := newHNSWEngine(s.T(), s.T().TempDir(), s.quant)
	defer loadedWAL.Close() //nolint:errcheck // test cleanup

	// Act
	err = loaded.LoadFromFile(ctx, s.snapPath)

	// Assert
	s.Require().NoError(err)
	s.Equal(10, loaded.Len())

	after, err := loaded.Search(ctx, vectors[3], core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Equal(before[0].ID, after[0].ID, "the reloaded graph must find the same nearest neighbour")

	filtered, err := loaded.Search(ctx, vectors[0], core.SparseVector{}, 10, map[string]any{"parity": "even"})
	s.Require().NoError(err)
	s.Len(filtered, 5, "the metadata index must be rebuilt on load")

	record, err := loaded.GetByID(ctx, "v0")
	s.Require().NoError(err)
	s.Equal(map[string]any{"parity": "even", "label": "hello", "score": 42.5, "active": true, "count": 7, "nested": map[string]any{"key": "value"}}, record.Metadata)
}

// TestHNSW_Recovery_SnapshotThenWAL recovers an engine from a snapshot plus
// the writes made after it, a delete among them, which only the WAL holds.
func (s *hnswSuiteBase) TestHNSW_Recovery_SnapshotThenWAL() {
	// Arrange
	ctx := context.Background()
	vectors := generateTestVectors(25, 8)
	for i, v := range vectors[:20] {
		s.Require().NoError(s.engine.Insert(ctx, fmt.Sprintf("v%d", i), v, core.SparseVector{}, nil))
	}
	s.Require().NoError(s.engine.SaveToFile(ctx, s.snapPath))
	for i, v := range vectors[20:] {
		s.Require().NoError(s.engine.Insert(ctx, fmt.Sprintf("v%d", 20+i), v, core.SparseVector{}, nil))
	}
	deleted, err := s.engine.Delete(ctx, "v3")
	s.Require().NoError(err)
	s.Require().True(deleted)
	s.Require().NoError(s.wal.Close())
	s.wal = nil
	recovered, recoveredWAL := newHNSWEngine(s.T(), s.T().TempDir(), s.quant)
	defer recoveredWAL.Close() //nolint:errcheck // test cleanup

	// Act
	loadErr := recovered.LoadFromFile(ctx, s.snapPath)
	replayErr := recovered.ReplayWAL(s.walPath)

	// Assert
	s.Require().NoError(loadErr)
	s.Require().NoError(replayErr)
	s.Equal(24, recovered.Len(), "20 from the snapshot, 5 from the WAL, minus the deleted one")

	results, err := recovered.Search(ctx, vectors[22], core.SparseVector{}, 1, nil)
	s.Require().NoError(err)
	s.Equal("v22", results[0].ID, "a vector known only to the WAL must be in the graph")
	_, err = recovered.GetByID(ctx, "v3")
	s.Error(err, "the delete must be replayed")
}

// =============================================================================
// Batch insert: HNSWIndex.BatchInsert adds a batch to the graph in one
// graph.Add call, which builds it in rounds of parallel searches
// (internal/hnsw/graph.go's addRound).
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

	router := api.SetupRouter(engine, "")

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
