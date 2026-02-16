package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"
)

type APITestSuite struct {
	suite.Suite
	index  *core.VectorIndex
	router *gin.Engine
}

func (s *APITestSuite) SetupTest() {
	gin.SetMode(gin.TestMode)
	s.index = core.NewVectorIndex()
	s.router = api.SetupRouter(s.index)
}

func (s *APITestSuite) TearDownTest() {
	// Cleanup if needed
}

func (s *APITestSuite) TestHealthEndpoint() {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	s.Assert().JSONEq(`{"status":"ok"}`, w.Body.String())
}

func (s *APITestSuite) TestVectorInsertFlow() {
	vectors := []struct {
		id       string
		vector   []float32
		metadata map[string]interface{}
	}{
		{
			id:       "v1",
			vector:   []float32{1.0, 2.0, 3.0},
			metadata: map[string]interface{}{"label": "first", "type": "test"},
		},
		{
			id:       "v2",
			vector:   []float32{4.0, 5.0, 6.0},
			metadata: map[string]interface{}{"label": "second", "type": "test"},
		},
		{
			id:       "v3",
			vector:   []float32{7.0, 8.0, 9.0},
			metadata: nil,
		},
	}

	for _, vec := range vectors {
		payload := map[string]interface{}{
			"id":       vec.id,
			"vector":   vec.vector,
			"metadata": vec.metadata,
		}

		body, err := json.Marshal(payload)
		s.Require().NoError(err)

		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusCreated, w.Code, "Insert should succeed for %s", vec.id)
		s.Assert().JSONEq(`{"status":"inserted"}`, w.Body.String())
	}

	s.Assert().Len(s.index.Store, len(vectors), "All vectors should be in the index")

	for _, vec := range vectors {
		s.Require().Contains(s.index.Store, vec.id, "Vector %s should exist", vec.id)
		s.Assert().Equal(vec.id, s.index.Store[vec.id].ID)
		s.Assert().Equal(vec.vector, s.index.Store[vec.id].Vector)
		s.Assert().Equal(vec.metadata, s.index.Store[vec.id].Metadata)
	}
}

func (s *APITestSuite) TestVectorInsertAndOverwrite() {
	payload1 := map[string]interface{}{
		"id":       "test_vec",
		"vector":   []float32{1.0, 2.0},
		"metadata": map[string]interface{}{"version": 1},
	}

	body1, _ := json.Marshal(payload1)
	req1 := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body1))
	req1.Header.Set("Content-Type", "application/json")

	w1 := httptest.NewRecorder()
	s.router.ServeHTTP(w1, req1)

	s.Assert().Equal(http.StatusCreated, w1.Code)
	s.Require().Contains(s.index.Store, "test_vec")
	s.Assert().Equal([]float32{1.0, 2.0}, s.index.Store["test_vec"].Vector)

	payload2 := map[string]interface{}{
		"id":       "test_vec",
		"vector":   []float32{3.0, 4.0, 5.0},
		"metadata": map[string]interface{}{"version": 2, "updated": true},
	}

	body2, _ := json.Marshal(payload2)
	req2 := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body2))
	req2.Header.Set("Content-Type", "application/json")

	w2 := httptest.NewRecorder()
	s.router.ServeHTTP(w2, req2)

	s.Assert().Equal(http.StatusCreated, w2.Code)
	s.Require().Contains(s.index.Store, "test_vec")
	s.Assert().Equal([]float32{3.0, 4.0, 5.0}, s.index.Store["test_vec"].Vector)
	s.Assert().Equal(float64(2), s.index.Store["test_vec"].Metadata["version"])
	s.Assert().Len(s.index.Store, 1, "Should still have only one vector")
}

func (s *APITestSuite) TestMultipleEndpoints() {
	healthReq := httptest.NewRequest(http.MethodGet, "/health", nil)
	healthW := httptest.NewRecorder()
	s.router.ServeHTTP(healthW, healthReq)
	s.Assert().Equal(http.StatusOK, healthW.Code)

	payload := map[string]interface{}{
		"id":     "test",
		"vector": []float32{1.0, 2.0},
	}

	body, _ := json.Marshal(payload)
	vectorReq := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	vectorReq.Header.Set("Content-Type", "application/json")

	vectorW := httptest.NewRecorder()
	s.router.ServeHTTP(vectorW, vectorReq)
	s.Assert().Equal(http.StatusCreated, vectorW.Code)

	healthReq2 := httptest.NewRequest(http.MethodGet, "/health", nil)
	healthW2 := httptest.NewRecorder()
	s.router.ServeHTTP(healthW2, healthReq2)
	s.Assert().Equal(http.StatusOK, healthW2.Code)
}

func (s *APITestSuite) TestValidationErrors() {
	testCases := []struct {
		name           string
		payload        map[string]interface{}
		expectedStatus int
	}{
		{
			name:           "missing_id",
			payload:        map[string]interface{}{"vector": []float32{1.0, 2.0}},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:           "missing_vector",
			payload:        map[string]interface{}{"id": "test"},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:           "empty_id",
			payload:        map[string]interface{}{"id": "", "vector": []float32{1.0}},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			body, _ := json.Marshal(tc.payload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)

			s.Assert().Equal(tc.expectedStatus, w.Code, "Test case: %s", tc.name)
		})
	}

	s.Assert().Empty(s.index.Store, "No vectors should be inserted on validation errors")
}

func (s *APITestSuite) TestLargeVectorInsertion() {
	largeVec := make([]float32, 1536)
	for i := range largeVec {
		largeVec[i] = float32(i) * 0.001
	}

	payload := map[string]interface{}{
		"id":     "large_embedding",
		"vector": largeVec,
		"metadata": map[string]interface{}{
			"model":      "text-embedding-ada-002",
			"dimensions": 1536,
		},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusCreated, w.Code)
	s.Require().Contains(s.index.Store, "large_embedding")
	s.Assert().Len(s.index.Store["large_embedding"].Vector, 1536)
}

func (s *APITestSuite) TestConcurrentInserts() {
	var wg sync.WaitGroup
	numRequests := 50

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			payload := map[string]interface{}{
				"id":     fmt.Sprintf("vec%d", id),
				"vector": []float32{float32(id), float32(id * 2)},
				"metadata": map[string]interface{}{
					"id":    id,
					"batch": "concurrent",
				},
			}

			body, _ := json.Marshal(payload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)

			s.Assert().Equal(http.StatusCreated, w.Code, "Request %d should succeed", id)
		}(i)
	}

	wg.Wait()

	s.Assert().Len(s.index.Store, numRequests, "Race condition may cause this to fail without mutex")

	for i := 0; i < numRequests; i++ {
		vecID := fmt.Sprintf("vec%d", i)
		s.Assert().Contains(s.index.Store, vecID, "Vector %s should exist", vecID)
	}
}

func (s *APITestSuite) TestSequentialInserts() {
	numVectors := 100

	for i := 0; i < numVectors; i++ {
		payload := map[string]interface{}{
			"id":       fmt.Sprintf("seq_vec_%d", i),
			"vector":   []float32{float32(i)},
			"metadata": map[string]interface{}{"index": i},
		}

		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusCreated, w.Code, "Insert %d should succeed", i)
	}

	s.Assert().Len(s.index.Store, numVectors)
}

func (s *APITestSuite) TestEdgeCases() {
	s.Run("empty_vector", func() {
		payload := map[string]interface{}{
			"id":     "empty_vec",
			"vector": []float32{},
		}

		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusCreated, w.Code)
	})

	s.Run("special_characters_in_id", func() {
		payload := map[string]interface{}{
			"id":     "vec-with-special_chars.123",
			"vector": []float32{1.0},
		}

		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusCreated, w.Code)
	})

	s.Run("complex_metadata", func() {
		payload := map[string]interface{}{
			"id":     "complex_meta",
			"vector": []float32{1.0, 2.0},
			"metadata": map[string]interface{}{
				"nested": map[string]interface{}{
					"level1": map[string]interface{}{
						"level2": "deep",
					},
				},
				"array": []interface{}{1, "two", true},
				"mixed": map[string]interface{}{
					"string": "value",
					"number": 42,
					"bool":   true,
				},
			},
		}

		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusCreated, w.Code)
		s.Require().Contains(s.index.Store, "complex_meta")
	})
}

func TestAPITestSuite(t *testing.T) {
	suite.Run(t, new(APITestSuite))
}
