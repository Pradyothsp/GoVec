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
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

type PersistenceIntegrationTestSuite struct {
	suite.Suite
	index      *core.VectorIndex[[]float32]
	router     *gin.Engine
	storageDir string
}

func (s *PersistenceIntegrationTestSuite) SetupTest() {
	gin.SetMode(gin.TestMode)
	s.index = testutil.NewTestIndex(s.T())
	s.router = api.SetupRouter(s.index)
	s.storageDir = s.T().TempDir()
}

func (s *PersistenceIntegrationTestSuite) TearDownTest() {
	// Cleanup handled by TempDir
}

// =============================================================================
// Test: Server Restart Preserves Data
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestServerRestart_PreservesData() {
	storagePath := filepath.Join(s.storageDir, "restart_test.bin")

	// Phase 1: Insert 10 vectors via HTTP
	numVectors := 10
	for i := 0; i < numVectors; i++ {
		payload := map[string]interface{}{
			"id":     fmt.Sprintf("vec%d", i),
			"vector": []float32{float32(i), float32(i * 2)},
			"metadata": map[string]interface{}{
				"index": i,
				"label": fmt.Sprintf("vector_%d", i),
			},
		}

		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusCreated, w.Code, "Insert should succeed for vec%d", i)
	}

	s.Assert().Len(s.index.Store, numVectors, "All vectors should be in memory")

	// Phase 2: Save to disk (manual save simulating shutdown)
	err := s.index.SaveToFile(storagePath)
	s.Require().NoError(err, "Should save to disk")
	testutil.AssertFileExists(s.T(), storagePath)

	// Phase 3: Simulate server restart - create new index and router
	newIndex := testutil.NewTestIndex(s.T())
	err = newIndex.LoadFromFile(storagePath)
	s.Require().NoError(err, "Should load from disk")

	newRouter := api.SetupRouter(newIndex)

	// Phase 4: Verify all 10 vectors exist via HTTP queries
	s.Assert().Len(newIndex.Store, numVectors, "Should load all vectors")

	for i := 0; i < numVectors; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		s.Require().Contains(newIndex.Store, vecID, "Vector %s should exist", vecID)

		node := newIndex.Store[vecID]
		s.Assert().Equal(vecID, node.ID)
		s.Assert().Equal([]float32{float32(i), float32(i * 2)}, node.Vector)
		s.Assert().Equal(float64(i), node.Metadata["index"])
	}

	// Phase 5: Insert another vector using new router to verify it still works
	payload := map[string]interface{}{
		"id":     "new_vec",
		"vector": []float32{99.0, 100.0},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	newRouter.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusCreated, w.Code, "New insert should work after restart")
	s.Assert().Len(newIndex.Store, numVectors+1, "Should have 11 vectors total")
}

// =============================================================================
// Test: Graceful Shutdown Saves Data
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestGracefulShutdown_SavesData() {
	storagePath := filepath.Join(s.storageDir, "graceful_shutdown.bin")

	// Insert vectors
	for i := 0; i < 5; i++ {
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

	// Simulate graceful shutdown (manual save)
	err := s.index.SaveToFile(storagePath)
	s.Require().NoError(err, "Graceful shutdown save should succeed")

	// Verify file exists
	testutil.AssertFileExists(s.T(), storagePath)

	// Restart and verify data
	newIndex := testutil.NewTestIndex(s.T())
	err = newIndex.LoadFromFile(storagePath)
	s.Require().NoError(err)
	s.Assert().Len(newIndex.Store, 5, "All vectors should be saved on graceful shutdown")
}

// =============================================================================
// Test: Recovery from Crash (No Save on Exit)
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestRecoveryFromCrash_OnlyAutoSaveData() {
	storagePath := filepath.Join(s.storageDir, "crash_recovery.bin")

	// Phase 1: Insert 10 vectors and auto-save
	for i := 0; i < 10; i++ {
		s.index.Insert(fmt.Sprintf("vec%d", i), []float32{float32(i)}, nil)
	}

	// Simulate auto-save
	err := s.index.SaveToFile(storagePath)
	s.Require().NoError(err)

	// Phase 2: Insert 5 more vectors (these won't be saved - simulating crash)
	for i := 10; i < 15; i++ {
		s.index.Insert(fmt.Sprintf("vec%d", i), []float32{float32(i)}, nil)
	}

	s.Assert().Len(s.index.Store, 15, "Should have 15 vectors in memory before crash")

	// Phase 3: Simulate crash (hard shutdown - no save)
	// Just restart with LoadFromFile

	// Phase 4: Restart - load from last auto-save
	recoveredIndex := testutil.NewTestIndex(s.T())
	err = recoveredIndex.LoadFromFile(storagePath)
	s.Require().NoError(err)

	// Phase 5: Verify only 10 vectors exist (last 5 lost due to crash)
	s.Assert().Len(recoveredIndex.Store, 10, "Should only recover vectors from last auto-save")

	for i := 0; i < 10; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		s.Assert().Contains(recoveredIndex.Store, vecID)
	}

	for i := 10; i < 15; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		s.Assert().NotContains(recoveredIndex.Store, vecID, "Unsaved vectors should be lost")
	}
}

// =============================================================================
// Test: Concurrent HTTP Requests During Save
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestConcurrentHTTPRequests_DuringSave() {
	storagePath := filepath.Join(s.storageDir, "concurrent_http.bin")

	// Populate index with 100 vectors
	testutil.PopulateIndexWithVectors(s.index, 100)

	var wg sync.WaitGroup

	// Start save operation in background
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := s.index.SaveToFile(storagePath)
		s.Assert().NoError(err, "Save should succeed during concurrent requests")
	}()

	// Send 50 concurrent insert requests while saving
	numRequests := 50
	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			payload := map[string]interface{}{
				"id":     fmt.Sprintf("concurrent_vec%d", id),
				"vector": []float32{float32(id), float32(id * 2)},
			}

			body, _ := json.Marshal(payload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)

			s.Assert().Equal(http.StatusCreated, w.Code, "Concurrent insert should succeed")
		}(i)
	}

	wg.Wait()

	// Verify all operations completed
	s.Assert().GreaterOrEqual(len(s.index.Store), 100, "Should have at least original 100 vectors")
	testutil.AssertFileExists(s.T(), storagePath)
}

// =============================================================================
// Test: Corrupted File Recovery
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestCorruptedFile_StartsWithEmptyIndex() {
	storagePath := filepath.Join(s.storageDir, "corrupted.bin")

	// Create corrupted file
	testutil.CorruptFile(s.T(), storagePath)

	// Try to load
	index := testutil.NewTestIndex(s.T())
	err := index.LoadFromFile(storagePath)

	s.Assert().Error(err, "Should fail to load corrupted file")
	s.Assert().Empty(index.Store, "Index should remain empty after failed load")

	// Verify server can still operate with empty index
	router := api.SetupRouter(index)

	payload := map[string]interface{}{
		"id":     "recovery_vec",
		"vector": []float32{1.0, 2.0},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusCreated, w.Code, "Should work with empty index after corruption")
}

// =============================================================================
// Test: Partial Save Recovery (Atomic Write Verification)
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestPartialSave_AtomicWriteProtection() {
	storagePath := filepath.Join(s.storageDir, "atomic_test.bin")

	// Phase 1: Save valid data
	testutil.PopulateIndexWithVectors(s.index, 50)
	err := s.index.SaveToFile(storagePath)
	s.Require().NoError(err)

	// Verify main file exists and .tmp is cleaned up
	testutil.AssertFileExists(s.T(), storagePath)
	testutil.AssertFileNotExists(s.T(), storagePath+".tmp")

	// Phase 2: Load to verify atomic write succeeded
	loadedIndex := testutil.NewTestIndex(s.T())
	err = loadedIndex.LoadFromFile(storagePath)
	s.Require().NoError(err)
	s.Assert().Len(loadedIndex.Store, 50, "All vectors should be loaded")

	// Phase 3: Verify no .tmp files left behind
	files, err := os.ReadDir(s.storageDir)
	s.Require().NoError(err)

	for _, file := range files {
		s.Assert().NotContains(file.Name(), ".tmp", "No temp files should remain")
	}
}

// =============================================================================
// Test: Large Dataset Persistence
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestLargeDataset_Persistence() {
	storagePath := filepath.Join(s.storageDir, "large_dataset.bin")

	// Insert 1000 vectors
	numVectors := 1000
	for i := 0; i < numVectors; i++ {
		id := fmt.Sprintf("vec%d", i)
		vec := make([]float32, 128) // 128-dimensional vectors
		for j := 0; j < 128; j++ {
			vec[j] = float32(i*128 + j)
		}
		meta := map[string]any{
			"index":      i,
			"dimensions": 128,
		}
		s.index.Insert(id, vec, meta)
	}

	// Save
	err := s.index.SaveToFile(storagePath)
	s.Require().NoError(err, "Should save large dataset")

	// Load
	loadedIndex := testutil.NewTestIndex(s.T())
	err = loadedIndex.LoadFromFile(storagePath)
	s.Require().NoError(err, "Should load large dataset")

	s.Assert().Len(loadedIndex.Store, numVectors, "All 1000 vectors should be loaded")

	// Spot check a few vectors
	s.Assert().Contains(loadedIndex.Store, "vec0")
	s.Assert().Contains(loadedIndex.Store, "vec500")
	s.Assert().Contains(loadedIndex.Store, "vec999")
}

// =============================================================================
// Test: Empty Index Persistence
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestEmptyIndex_Persistence() {
	storagePath := filepath.Join(s.storageDir, "empty_index.bin")

	// Save empty index
	emptyIndex := testutil.NewTestIndex(s.T())
	err := emptyIndex.SaveToFile(storagePath)
	s.Require().NoError(err, "Should save empty index")

	// Load empty index
	loadedIndex := testutil.NewTestIndex(s.T())
	err = loadedIndex.LoadFromFile(storagePath)
	s.Require().NoError(err, "Should load empty index")

	s.Assert().Empty(loadedIndex.Store, "Loaded index should be empty")
}

// =============================================================================
// Test: Auto-Save Simulation
// =============================================================================

func (s *PersistenceIntegrationTestSuite) TestAutoSave_Simulation() {
	storagePath := filepath.Join(s.storageDir, "autosave_test.bin")

	// Insert initial vectors
	for i := 0; i < 5; i++ {
		s.index.Insert(fmt.Sprintf("vec%d", i), []float32{float32(i)}, nil)
	}

	// Simulate auto-save ticker
	saveInterval := 100 * time.Millisecond
	ticker := time.NewTicker(saveInterval)
	defer ticker.Stop()

	done := make(chan bool)
	saveCount := 0

	// Auto-save goroutine
	go func() {
		for {
			select {
			case <-ticker.C:
				err := s.index.SaveToFile(storagePath)
				s.Assert().NoError(err, "Auto-save should succeed")
				saveCount++
				if saveCount >= 3 {
					done <- true
					return
				}
			case <-done:
				return
			}
		}
	}()

	// Insert more vectors while auto-save is running
	time.Sleep(50 * time.Millisecond)
	for i := 5; i < 10; i++ {
		s.index.Insert(fmt.Sprintf("vec%d", i), []float32{float32(i)}, nil)
	}

	// Wait for auto-save to complete
	<-done

	// Verify file was created and has latest data
	loadedIndex := testutil.NewTestIndex(s.T())
	err := loadedIndex.LoadFromFile(storagePath)
	s.Require().NoError(err)
	s.Assert().Len(loadedIndex.Store, 10, "Auto-save should have saved all vectors")
}

func TestPersistenceIntegrationTestSuite(t *testing.T) {
	suite.Run(t, new(PersistenceIntegrationTestSuite))
}
