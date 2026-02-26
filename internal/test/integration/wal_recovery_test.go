package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"

	"github.com/Pradyothsp/govec/internal/index"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/core"
)

type WALRecoveryTestSuite struct {
	suite.Suite
	index      *index.VectorIndex[[]float32]
	router     *gin.Engine
	storageDir string
	walPath    string
	snapPath   string
	wal        *index.WAL
}

func (s *WALRecoveryTestSuite) SetupTest() {
	gin.SetMode(gin.TestMode)
	s.storageDir = s.T().TempDir()
	s.walPath = filepath.Join(s.storageDir, "test.wal")
	s.snapPath = filepath.Join(s.storageDir, "snapshot.bin")

	wal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	s.wal = wal
	s.T().Cleanup(func() { _ = wal.Close() })

	idMapper := core.NewIDMapper()
	s.index = index.NewVectorIndex[[]float32](wal, nil, idMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	s.router = api.SetupRouter(s.index, "", "")
}

func TestWALRecoveryTestSuite(t *testing.T) {
	suite.Run(t, new(WALRecoveryTestSuite))
}

// =============================================================================
// A. Basic Recovery Tests (3 tests)
// =============================================================================

func (s *WALRecoveryTestSuite) TestWALRecovery_CrashAfterInserts() {
	// 🔴 CRITICAL: Will reveal main.go bug (lines 40-42)

	// Phase 1: Insert 50 vectors via HTTP
	for i := 0; i < 50; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i), float32(i * 2)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Verify all in memory
	s.Assert().Len(s.index.Store, 50)

	// Phase 2: Simulate crash (no save, close WAL)
	s.wal.Close()

	// Phase 3: Recovery (LoadFromFile + ReplayWAL)
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	// Load from missing snapshot
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err, "Missing snapshot should be handled gracefully")

	// Replay WAL
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Verify all 50 vectors recovered
	s.Assert().Len(recoveredIdx.Store, 50, "All 50 vectors should be recovered from WAL")
	for i := 0; i < 50; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		internalID, err := recoveredIdx.IDMapper.ToUint32ID(vecID)
		s.Require().NoError(err, "Vector %s should exist", vecID)
		s.Assert().Contains(recoveredIdx.Store, internalID)
	}

	// NOTE: This test may FAIL with current main.go:40-42
	// Expected fix: Load from DataPath (GOB), then ReplayWAL from WalPath (JSON)
}

func (s *WALRecoveryTestSuite) TestWALRecovery_CrashDuringMixedOperations() {
	// INSERT vec1-3
	for i := 1; i <= 3; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// DELETE vec2
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/vec2", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Assert().Equal(http.StatusOK, w.Code)

	// INSERT vec4-5
	for i := 4; i <= 5; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// UPDATE vec1 (overwrite)
	payload := map[string]interface{}{
		"id":     "vec1",
		"vector": []float32{999.0},
	}
	body, _ := json.Marshal(payload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Assert().Equal(http.StatusCreated, w.Code)

	// Crash
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Final state: vec1(updated), vec3, vec4, vec5
	s.Assert().Len(recoveredIdx.Store, 4)
	vec1Recovered, _ := recoveredIdx.IDMapper.ToUint32ID("vec1")
	s.Assert().Contains(recoveredIdx.Store, vec1Recovered)
	vec1ID, _ := recoveredIdx.IDMapper.ToUint32ID("vec1")
	vec3ID, _ := recoveredIdx.IDMapper.ToUint32ID("vec3")
	vec4ID, _ := recoveredIdx.IDMapper.ToUint32ID("vec4")
	vec5ID, _ := recoveredIdx.IDMapper.ToUint32ID("vec5")
	s.Assert().Contains(recoveredIdx.Store, vec3ID)
	s.Assert().Contains(recoveredIdx.Store, vec4ID)
	s.Assert().Contains(recoveredIdx.Store, vec5ID)

	_, err = recoveredIdx.IDMapper.ToUint32ID("vec2")
	s.Assert().Error(err, "vec2 was deleted")

	// Verify vec1 has updated value
	s.Assert().Equal([]float32{999.0}, recoveredIdx.Store[vec1ID].Vector)
}

func (s *WALRecoveryTestSuite) TestWALRecovery_EmptyWAL() {
	// Recovery with no WAL and no snapshot
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)

	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err, "Missing snapshot should be handled gracefully")

	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err, "Missing WAL should be handled gracefully")

	// Verify server starts with empty index
	s.Assert().Len(recoveredIdx.Store, 0, "Index should be empty")
}

// =============================================================================
// B. Snapshot + WAL Recovery Tests (3 tests)
// =============================================================================

func (s *WALRecoveryTestSuite) TestWALRecovery_CrashBetweenAutoSaves() {
	// Insert 100 vectors
	for i := 0; i < 100; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Save snapshot
	err := s.index.SaveToFile(s.snapPath)
	s.Require().NoError(err)

	// Insert 20 more vectors
	for i := 100; i < 120; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Crash (no second save)
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Verify 120 vectors total
	s.Assert().Len(recoveredIdx.Store, 120, "All 120 vectors should be recovered")
}

func (s *WALRecoveryTestSuite) TestWALRecovery_SnapshotAfterDeletes() {
	// Insert vec1-10
	for i := 1; i <= 10; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// DELETE vec5-7
	for i := 5; i <= 7; i++ {
		req := httptest.NewRequest(http.MethodDelete, fmt.Sprintf("/api/v1/vectors/vec%d", i), nil)
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusOK, w.Code)
	}

	// Save snapshot
	err := s.index.SaveToFile(s.snapPath)
	s.Require().NoError(err)

	// INSERT vec11-12
	for i := 11; i <= 12; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// DELETE vec1
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/vec1", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Assert().Equal(http.StatusOK, w.Code)

	// Crash
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Final: vec2-4, vec8-12 (8 vectors)
	s.Assert().Len(recoveredIdx.Store, 8)

	expectedVecs := []string{"vec2", "vec3", "vec4", "vec8", "vec9", "vec10", "vec11", "vec12"}
	for _, vecID := range expectedVecs {
		internalID, err := recoveredIdx.IDMapper.ToUint32ID(vecID)
		s.Require().NoError(err, "Vector %s should exist", vecID)
		s.Assert().Contains(recoveredIdx.Store, internalID)
	}
}

func (s *WALRecoveryTestSuite) TestWALRecovery_MultipleSnapshots() {
	// Cycle 1: Insert 10, save
	for i := 0; i < 10; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("cycle1_vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}
	err := s.index.SaveToFile(s.snapPath)
	s.Require().NoError(err)

	// Cycle 2: Insert 10, save
	for i := 0; i < 10; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("cycle2_vec%d", i),
			"vector": []float32{float32(i + 100)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}
	err = s.index.SaveToFile(s.snapPath)
	s.Require().NoError(err)

	// Cycle 3: Insert 5, crash (no save)
	for i := 0; i < 5; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("cycle3_vec%d", i),
			"vector": []float32{float32(i + 200)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Crash
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Only last snapshot + WAL matters (25 vectors)
	s.Assert().Len(recoveredIdx.Store, 25, "Last snapshot (20) + WAL (5) = 25 vectors")
}

// =============================================================================
// C. Concurrent Operations Tests (2 tests)
// =============================================================================

func (s *WALRecoveryTestSuite) TestWALRecovery_ConcurrentInsertsBeforeCrash() {
	const goroutines = 200
	var wg sync.WaitGroup
	wg.Add(goroutines)

	// 200 concurrent HTTP POST requests
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			payload := map[string]interface{}{
				"id":     fmt.Sprintf("concurrent%d", id),
				"vector": []float32{float32(id)},
			}
			body, _ := json.Marshal(payload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)
		}(i)
	}

	wg.Wait()

	// Count successes
	successCount := len(s.index.Store)

	// Crash
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Verify all successful inserts recovered
	s.Assert().Equal(successCount, len(recoveredIdx.Store), "All successful inserts should be recovered")
}

func (s *WALRecoveryTestSuite) TestWALRecovery_InterleavedInsertsDeletes() {
	var wg sync.WaitGroup
	wg.Add(2)

	// Goroutine 1: Insert vec0-99
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			payload := map[string]interface{}{
				"id":     fmt.Sprintf("vec%d", i),
				"vector": []float32{float32(i)},
			}
			body, _ := json.Marshal(payload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)
		}
	}()

	// Goroutine 2: Delete vec0-49 (some may not exist yet)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			req := httptest.NewRequest(http.MethodDelete, fmt.Sprintf("/api/v1/vectors/vec%d", i), nil)
			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)
		}
	}()

	wg.Wait()

	// Final state in memory
	memoryCount := len(s.index.Store)

	// Crash
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Final state matches WAL order (no corruption)
	s.Assert().Equal(memoryCount, len(recoveredIdx.Store), "Recovered state should match pre-crash state")
}

// =============================================================================
// D. Error Recovery Tests (2 tests)
// =============================================================================

func (s *WALRecoveryTestSuite) TestWALRecovery_CorruptedWALGracefulHandling() {
	// Insert vec1-5
	for i := 1; i <= 5; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Corrupt WAL
	s.wal.Close()
	f, err := os.OpenFile(s.walPath, os.O_APPEND|os.O_WRONLY, 0644)
	s.Require().NoError(err)
	_, err = f.Write([]byte("CORRUPTED DATA\n"))
	s.Require().NoError(err)
	f.Close()

	// Attempt insert after corruption (will fail)
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	s.wal = newWal
	s.T().Cleanup(func() { _ = newWal.Close() })

	// Recovery
	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err)
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err, "Replay should not crash on corrupted WAL")

	// Verify vec1-5 recovered (corrupted line skipped)
	s.Assert().Len(recoveredIdx.Store, 5, "Valid entries should be recovered")
}

func (s *WALRecoveryTestSuite) TestWALRecovery_MissingSnapshotButValidWAL() {
	// Insert 10 vectors
	for i := 0; i < 10; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Save snapshot
	err := s.index.SaveToFile(s.snapPath)
	s.Require().NoError(err)

	// Insert 5 more
	for i := 10; i < 15; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i)},
		}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	// Delete snapshot file
	err = os.Remove(s.snapPath)
	s.Require().NoError(err)

	// Crash
	s.wal.Close()

	// Recovery
	newWal, err := index.NewWAL(s.walPath)
	s.Require().NoError(err)
	defer newWal.Close()

	recoveredIDMapper := core.NewIDMapper()
	recoveredIdx := index.NewVectorIndex[[]float32](newWal, nil, recoveredIDMapper, func(v []float32) []float32 { return v }, core.CosineSimilarity, nil)
	err = recoveredIdx.LoadFromFile(s.snapPath)
	s.Require().NoError(err, "Missing snapshot should be handled gracefully")
	err = recoveredIdx.ReplayWAL(s.walPath)
	s.Require().NoError(err)

	// Only post-snapshot vectors recovered (5 vectors: vec10-14)
	s.Assert().Len(recoveredIdx.Store, 5, "Only post-snapshot WAL entries should be recovered")

	// NOTE: This documents that snapshot deletion is catastrophic
	// Pre-snapshot data is lost (vec0-9 are gone)
}
