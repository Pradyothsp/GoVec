package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"

	"github.com/Pradyothsp/govec/internal/index"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

type APITestSuite struct {
	suite.Suite
	index  *index.VectorIndex[[]float32]
	router *gin.Engine
}

func (s *APITestSuite) SetupTest() {
	gin.SetMode(gin.TestMode)
	s.index = testutil.NewTestIndex(s.T())
	s.router = api.SetupRouter(s.index, "", s.T().TempDir()+"/test.bin")
}

func (s *APITestSuite) TearDownTest() {
	// Cleanup if needed
}

func (s *APITestSuite) TestHealthEndpoint() {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	s.Assert().JSONEq(`{"success":true,"data":{"status":"ok"}}`, w.Body.String())
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
		s.Assert().JSONEq(`{"success":true,"data":{"status":"inserted"}}`, w.Body.String())
	}

	s.Assert().Len(s.index.Store, len(vectors), "All vectors should be in the index")

	for _, vec := range vectors {
		internalID, err := s.index.IDMapper.ToUint32ID(vec.id)
		s.Require().NoError(err, "Should be able to map %s", vec.id)
		s.Require().Contains(s.index.Store, internalID, "Vector %s should exist", vec.id)
		s.Assert().Equal(vec.id, s.index.Store[internalID].ExternalID)
		s.Assert().Equal(vec.vector, s.index.Store[internalID].Vector)
		s.Assert().Equal(vec.metadata, s.index.Store[internalID].Metadata)
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
	internalID, err := s.index.IDMapper.ToUint32ID("test_vec")
	s.Require().NoError(err)
	s.Require().Contains(s.index.Store, internalID)
	s.Assert().Equal([]float32{1.0, 2.0}, s.index.Store[internalID].Vector)

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
	internalID2, err2 := s.index.IDMapper.ToUint32ID("test_vec")
	s.Require().NoError(err2)
	s.Require().Contains(s.index.Store, internalID2)
	s.Assert().Equal([]float32{3.0, 4.0, 5.0}, s.index.Store[internalID2].Vector)
	s.Assert().Equal(float64(2), s.index.Store[internalID2].Metadata["version"])
	s.Assert().Len(s.index.Store, 1, "Should still have only one vector")
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
	largeID, err := s.index.IDMapper.ToUint32ID("large_embedding")
	s.Require().NoError(err)
	s.Require().Contains(s.index.Store, largeID)
	s.Assert().Len(s.index.Store[largeID].Vector, 1536)
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
		_, err := s.index.IDMapper.ToUint32ID(vecID)
		s.Assert().NoError(err, "Vector %s should exist", vecID)
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
		complexID, err := s.index.IDMapper.ToUint32ID("complex_meta")
		s.Require().NoError(err)
		s.Require().Contains(s.index.Store, complexID)
	})
}

func (s *APITestSuite) TestInsertAndSearchFlow() {
	// Step 1: Insert multiple vectors via HTTP
	vectors := []struct {
		id       string
		vector   []float32
		metadata map[string]interface{}
	}{
		{
			id:       "doc1",
			vector:   []float32{1.0, 0.0, 0.0},
			metadata: map[string]interface{}{"title": "First Document", "category": "tech"},
		},
		{
			id:       "doc2",
			vector:   []float32{0.9, 0.1, 0.0},
			metadata: map[string]interface{}{"title": "Second Document", "category": "tech"},
		},
		{
			id:       "doc3",
			vector:   []float32{0.0, 1.0, 0.0},
			metadata: map[string]interface{}{"title": "Third Document", "category": "science"},
		},
		{
			id:       "doc4",
			vector:   []float32{-1.0, 0.0, 0.0},
			metadata: map[string]interface{}{"title": "Fourth Document", "category": "tech"},
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
	}

	// Step 2: Search for similar vectors via HTTP
	searchPayload := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0}, // Should be most similar to doc1
		"k":      3,
	}

	searchBody, err := json.Marshal(searchPayload)
	s.Require().NoError(err)

	searchReq := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(searchBody))
	searchReq.Header.Set("Content-Type", "application/json")

	searchW := httptest.NewRecorder()
	s.router.ServeHTTP(searchW, searchReq)

	// Step 3: Verify search results
	s.Assert().Equal(http.StatusOK, searchW.Code, "Search should succeed")

	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	err = json.Unmarshal(searchW.Body.Bytes(), &env)
	s.Require().NoError(err, "Should be able to parse search results")
	s.Require().True(env.Success)
	results := env.Data

	// Verify we got exactly k results
	s.Assert().Len(results, 3, "Should return exactly k=3 results")

	// Verify first result is doc1 (identical vector, score = 1.0)
	s.Assert().Equal("doc1", results[0]["id"], "First result should be doc1")
	s.Assert().InDelta(1.0, results[0]["score"], 0.001, "doc1 should have similarity score of 1.0")

	meta0, ok := results[0]["meta"].(map[string]interface{})
	s.Require().True(ok, "Metadata should be a map")
	s.Assert().Equal("First Document", meta0["title"], "Should return correct metadata for doc1")
	s.Assert().Equal("tech", meta0["category"], "Should return correct category for doc1")

	// Verify second result is doc2 (similar vector, score > 0.8)
	s.Assert().Equal("doc2", results[1]["id"], "Second result should be doc2")
	score1, ok := results[1]["score"].(float64)
	s.Require().True(ok, "Score should be a number")
	s.Assert().Greater(score1, 0.8, "doc2 should have high similarity")

	meta1, ok := results[1]["meta"].(map[string]interface{})
	s.Require().True(ok, "Metadata should be a map")
	s.Assert().Equal("Second Document", meta1["title"], "Should return correct metadata for doc2")

	// Verify results are sorted by score (descending)
	for i := 0; i < len(results)-1; i++ {
		scoreI, ok1 := results[i]["score"].(float64)
		scoreNext, ok2 := results[i+1]["score"].(float64)
		s.Require().True(ok1 && ok2, "All scores should be numbers")
		s.Assert().GreaterOrEqual(scoreI, scoreNext, "Results should be sorted by score descending")
	}
}

func (s *APITestSuite) TestSearchWithDifferentKValues() {
	// Insert test vectors
	for i := 1; i <= 10; i++ {
		payload := map[string]interface{}{
			"id":       fmt.Sprintf("vec%d", i),
			"vector":   []float32{float32(i), float32(i * 2)},
			"metadata": map[string]interface{}{"index": i},
		}

		body, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Assert().Equal(http.StatusCreated, w.Code)
	}

	testCases := []struct {
		name          string
		k             int
		expectedCount int
	}{
		{"k=1", 1, 1},
		{"k=5", 5, 5},
		{"k=10", 10, 10},
		{"k=20", 20, 10}, // More than available
		{"k=0", 0, 10},   // All results
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			searchPayload := map[string]interface{}{
				"vector": []float32{5.0, 10.0},
				"k":      tc.k,
			}

			body, _ := json.Marshal(searchPayload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)

			s.Assert().Equal(http.StatusOK, w.Code)

			var env struct {
				Success bool                     `json:"success"`
				Data    []map[string]interface{} `json:"data"`
			}
			err := json.Unmarshal(w.Body.Bytes(), &env)
			s.Require().NoError(err)
			s.Require().True(env.Success)

			s.Assert().Len(env.Data, tc.expectedCount, "Should return correct number of results for %s", tc.name)
		})
	}
}

func (s *APITestSuite) TestSearchEmptyIndex() {
	// Don't insert any vectors

	searchPayload := map[string]interface{}{
		"vector": []float32{1.0, 2.0, 3.0},
		"k":      5,
	}

	body, _ := json.Marshal(searchPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code, "Empty index search should succeed")

	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	err := json.Unmarshal(w.Body.Bytes(), &env)
	s.Require().NoError(err)
	s.Require().True(env.Success)

	s.Assert().Empty(env.Data, "Should return empty array for empty index")
}

func (s *APITestSuite) TestSearchValidationErrors() {
	// Insert a test vector
	insertPayload := map[string]interface{}{
		"id":     "test",
		"vector": []float32{1.0, 2.0, 3.0},
	}
	insertBody, _ := json.Marshal(insertPayload)
	insertReq := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(insertBody))
	insertReq.Header.Set("Content-Type", "application/json")
	insertW := httptest.NewRecorder()
	s.router.ServeHTTP(insertW, insertReq)
	s.Assert().Equal(http.StatusCreated, insertW.Code)

	testCases := []struct {
		name           string
		payload        map[string]interface{}
		expectedStatus int
		errorContains  string
	}{
		{
			name:           "missing_vector",
			payload:        map[string]interface{}{"k": 5},
			expectedStatus: http.StatusBadRequest,
			errorContains:  "",
		},
		{
			name:           "empty_vector",
			payload:        map[string]interface{}{"vector": []float32{}, "k": 5},
			expectedStatus: http.StatusInternalServerError,
			errorContains:  "empty query vector",
		},
		{
			name:           "dimension_mismatch",
			payload:        map[string]interface{}{"vector": []float32{1.0, 2.0}, "k": 5},
			expectedStatus: http.StatusInternalServerError,
			errorContains:  "dimension",
		},
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			body, _ := json.Marshal(tc.payload)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)

			s.Assert().Equal(tc.expectedStatus, w.Code, "Test case: %s", tc.name)

			if tc.errorContains != "" {
				var response map[string]interface{}
				err := json.Unmarshal(w.Body.Bytes(), &response)
				s.Require().NoError(err)
				s.Assert().Contains(response["error"], tc.errorContains, "Error message should contain expected text")
			}
		})
	}
}

func (s *APITestSuite) TestFilteredSearch_ByCategory() {
	vectors := []struct {
		id       string
		vector   []float32
		metadata map[string]interface{}
	}{
		{"doc1", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"category": "tech"}},
		{"doc2", []float32{0.9, 0.1, 0.0}, map[string]interface{}{"category": "tech"}},
		{"doc3", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"category": "science"}},
	}

	for _, vec := range vectors {
		body, _ := json.Marshal(map[string]interface{}{
			"id": vec.id, "vector": vec.vector, "metadata": vec.metadata,
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	searchPayload := map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"k":      10,
		"filter": map[string]interface{}{"category": "tech"},
	}
	body, _ := json.Marshal(searchPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Require().True(env.Success)
	results := env.Data

	s.Assert().Len(results, 2)
	resultIDs := make([]string, len(results))
	for i, r := range results {
		resultIDs[i] = r["id"].(string)
	}
	s.Assert().Contains(resultIDs, "doc1")
	s.Assert().Contains(resultIDs, "doc2")
	s.Assert().NotContains(resultIDs, "doc3")
}

func (s *APITestSuite) TestFilteredSearch_NoMatches() {
	for i := 1; i <= 3; i++ {
		body, _ := json.Marshal(map[string]interface{}{
			"id":       fmt.Sprintf("v%d", i),
			"vector":   []float32{float32(i), float32(i * 2)},
			"metadata": map[string]interface{}{"type": "A"},
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	searchPayload := map[string]interface{}{
		"vector": []float32{1.0, 2.0},
		"k":      10,
		"filter": map[string]interface{}{"type": "B"},
	}
	body, _ := json.Marshal(searchPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Require().True(env.Success)
	s.Assert().Empty(env.Data)
}

func (s *APITestSuite) TestFilteredSearch_MultipleFilterKeys() {
	vectors := []struct {
		id       string
		vector   []float32
		metadata map[string]interface{}
	}{
		{"doc1", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"cat": "tech", "active": true}},
		{"doc2", []float32{0.9, 0.1, 0.0}, map[string]interface{}{"cat": "tech", "active": false}},
		{"doc3", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"cat": "sci", "active": true}},
	}

	for _, vec := range vectors {
		body, _ := json.Marshal(map[string]interface{}{
			"id": vec.id, "vector": vec.vector, "metadata": vec.metadata,
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	// Filter cat=tech AND active=true → only doc1
	s.Run("cat_and_active_true", func() {
		searchPayload := map[string]interface{}{
			"vector": []float32{1.0, 0.0, 0.0},
			"k":      10,
			"filter": map[string]interface{}{"cat": "tech", "active": true},
		}
		body, _ := json.Marshal(searchPayload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusOK, w.Code)
		var env struct {
			Success bool                     `json:"success"`
			Data    []map[string]interface{} `json:"data"`
		}
		s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
		s.Require().True(env.Success)
		s.Assert().Len(env.Data, 1)
		s.Assert().Equal("doc1", env.Data[0]["id"])
	})

	// Filter cat=tech → doc1 and doc2
	s.Run("cat_only", func() {
		searchPayload := map[string]interface{}{
			"vector": []float32{1.0, 0.0, 0.0},
			"k":      10,
			"filter": map[string]interface{}{"cat": "tech"},
		}
		body, _ := json.Marshal(searchPayload)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)

		s.Assert().Equal(http.StatusOK, w.Code)
		var env struct {
			Success bool                     `json:"success"`
			Data    []map[string]interface{} `json:"data"`
		}
		s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
		s.Require().True(env.Success)
		s.Assert().Len(env.Data, 2)
		resultIDs := []string{env.Data[0]["id"].(string), env.Data[1]["id"].(string)}
		s.Assert().Contains(resultIDs, "doc1")
		s.Assert().Contains(resultIDs, "doc2")
	})
}

func (s *APITestSuite) TestFilteredSearch_FilterWithKLimit() {
	// 5 vectors all with tag=keep, varying similarity to query [1,0]
	vecs := []struct {
		id  string
		vec []float32
	}{
		{"close1", []float32{1.0, 0.0}},
		{"close2", []float32{0.9, 0.1}},
		{"mid1", []float32{0.7, 0.3}},
		{"far1", []float32{0.5, 0.5}},
		{"far2", []float32{0.3, 0.7}},
	}

	for _, v := range vecs {
		body, _ := json.Marshal(map[string]interface{}{
			"id":       v.id,
			"vector":   v.vec,
			"metadata": map[string]interface{}{"tag": "keep"},
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	searchPayload := map[string]interface{}{
		"vector": []float32{1.0, 0.0},
		"k":      2,
		"filter": map[string]interface{}{"tag": "keep"},
	}
	body, _ := json.Marshal(searchPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Require().True(env.Success)
	results := env.Data
	s.Assert().Len(results, 2)

	// Verify sorted by score descending
	score0, ok0 := results[0]["score"].(float64)
	score1, ok1 := results[1]["score"].(float64)
	s.Require().True(ok0 && ok1)
	s.Assert().GreaterOrEqual(score0, score1)
}

func (s *APITestSuite) TestFilteredSearch_NoFilterField() {
	for i := 1; i <= 3; i++ {
		body, _ := json.Marshal(map[string]interface{}{
			"id":       fmt.Sprintf("v%d", i),
			"vector":   []float32{float32(i), float32(i * 2)},
			"metadata": map[string]interface{}{"index": i},
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	// No filter field — should return all results
	searchPayload := map[string]interface{}{
		"vector": []float32{1.0, 2.0},
		"k":      10,
	}
	body, _ := json.Marshal(searchPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Require().True(env.Success)
	s.Assert().Len(env.Data, 3, "No filter should return all results")
}

func (s *APITestSuite) TestDeleteFlow() {
	// Step 1: Insert a vector
	body, _ := json.Marshal(map[string]interface{}{
		"id":       "to_delete",
		"vector":   []float32{1.0, 2.0, 3.0},
		"metadata": map[string]interface{}{"label": "temp"},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Require().Equal(http.StatusCreated, w.Code)

	// Step 2: Delete the vector
	req = httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/to_delete", nil)
	w = httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Assert().Equal(http.StatusOK, w.Code)
	s.Assert().JSONEq(`{"success":true,"data":{"status":"deleted","id":"to_delete"}}`, w.Body.String())

	// Step 3: Verify the vector is deleted (IDMapper returns error)
	_, err := s.index.IDMapper.ToUint32ID("to_delete")
	s.Assert().Error(err, "Deleted ID should not be found")

	// Step 4: Delete again — idempotency check must return 404
	req = httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/to_delete", nil)
	w = httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Assert().Equal(http.StatusNotFound, w.Code)
	s.Assert().JSONEq(`{"success":false,"data":null,"error":"vector not found"}`, w.Body.String())
}

func (s *APITestSuite) TestInsertDeleteSearchFlow() {
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"keep1", []float32{1.0, 0.0, 0.0}},
		{"delete_me", []float32{0.9, 0.1, 0.0}},
		{"keep2", []float32{0.0, 1.0, 0.0}},
	}

	for _, vec := range vectors {
		body, _ := json.Marshal(map[string]interface{}{"id": vec.id, "vector": vec.vector})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	// Delete one vector
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/delete_me", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)
	s.Require().Equal(http.StatusOK, w.Code)

	// Search and verify deleted vector is absent from results
	searchBody, _ := json.Marshal(map[string]interface{}{
		"vector": []float32{1.0, 0.0, 0.0},
		"k":      10,
	})
	req = httptest.NewRequest(http.MethodPost, "/api/v1/vectors/search", bytes.NewReader(searchBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	var env struct {
		Success bool                     `json:"success"`
		Data    []map[string]interface{} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Require().True(env.Success)
	results := env.Data
	s.Assert().Len(results, 2, "Should have 2 results after deletion")

	resultIDs := make([]string, len(results))
	for i, r := range results {
		resultIDs[i] = r["id"].(string)
	}
	s.Assert().NotContains(resultIDs, "delete_me")
	s.Assert().Contains(resultIDs, "keep1")
	s.Assert().Contains(resultIDs, "keep2")
}

func (s *APITestSuite) TestDeleteNonExistentVector() {
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/vectors/ghost", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusNotFound, w.Code)
	s.Assert().JSONEq(`{"success":false,"data":null,"error":"vector not found"}`, w.Body.String())
}

func (s *APITestSuite) TestConcurrentDeletes() {
	numVectors := 20
	for i := 0; i < numVectors; i++ {
		body, _ := json.Marshal(map[string]interface{}{
			"id":     fmt.Sprintf("del_vec%d", i),
			"vector": []float32{float32(i), float32(i * 2)},
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.router.ServeHTTP(w, req)
		s.Require().Equal(http.StatusCreated, w.Code)
	}

	var wg sync.WaitGroup
	statuses := make([]int, numVectors)
	for i := 0; i < numVectors; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodDelete, fmt.Sprintf("/api/v1/vectors/del_vec%d", idx), nil)
			w := httptest.NewRecorder()
			s.router.ServeHTTP(w, req)
			statuses[idx] = w.Code
		}(i)
	}
	wg.Wait()

	for i, status := range statuses {
		s.Assert().Equal(http.StatusOK, status, "Delete %d should return 200", i)
	}
	s.Assert().Empty(s.index.Store)
}

func (s *APITestSuite) TestStatsEndpoint() {
	// Empty index → count 0
	req := httptest.NewRequest(http.MethodGet, "/api/v1/stats", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	s.Assert().JSONEq(`{"success":true,"data":{"vector_count":0}}`, w.Body.String())

	// Insert a vector
	body, _ := json.Marshal(map[string]interface{}{
		"id":     "stat_v1",
		"vector": []float32{1.0, 2.0, 3.0},
	})
	insertReq := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	insertReq.Header.Set("Content-Type", "application/json")
	insertW := httptest.NewRecorder()
	s.router.ServeHTTP(insertW, insertReq)
	s.Require().Equal(http.StatusCreated, insertW.Code)

	// Count should now be 1
	req = httptest.NewRequest(http.MethodGet, "/api/v1/stats", nil)
	w = httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	s.Assert().JSONEq(`{"success":true,"data":{"vector_count":1}}`, w.Body.String())
}

func (s *APITestSuite) TestInfoEndpoint() {
	// Empty index → dimensions 0
	req := httptest.NewRequest(http.MethodGet, "/api/v1/info", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool                   `json:"success"`
		Data    map[string]interface{} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Assert().True(env.Success)
	s.Assert().Equal(float64(0), env.Data["dimensions"])
	s.Assert().Equal(float64(0), env.Data["vector_count"])

	// Insert a 3-dim vector
	body, _ := json.Marshal(map[string]interface{}{
		"id":     "info_v1",
		"vector": []float32{1.0, 2.0, 3.0},
	})
	insertReq := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	insertReq.Header.Set("Content-Type", "application/json")
	insertW := httptest.NewRecorder()
	s.router.ServeHTTP(insertW, insertReq)
	s.Require().Equal(http.StatusCreated, insertW.Code)

	// Dimensions should now be 3
	req = httptest.NewRequest(http.MethodGet, "/api/v1/info", nil)
	w = httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Assert().True(env.Success)
	s.Assert().Equal(float64(3), env.Data["dimensions"])
	s.Assert().Equal(float64(1), env.Data["vector_count"])
}

func (s *APITestSuite) TestBatchInsert_HappyPath() {
	body, _ := json.Marshal(map[string]interface{}{
		"vectors": []map[string]interface{}{
			{"id": "b1", "vector": []float32{1.0, 2.0, 3.0}},
			{"id": "b2", "vector": []float32{4.0, 5.0, 6.0}},
			{"id": "b3", "vector": []float32{7.0, 8.0, 9.0}},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool `json:"success"`
		Data    struct {
			InsertedCount int `json:"inserted_count"`
		} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Assert().True(env.Success)
	s.Assert().Equal(3, env.Data.InsertedCount)
	s.Assert().Len(s.index.Store, 3)
}

func (s *APITestSuite) TestBatchInsert_PartialFailure() {
	body, _ := json.Marshal(map[string]interface{}{
		"vectors": []map[string]interface{}{
			{"id": "ok1", "vector": []float32{1.0, 2.0}},
			// invalid sparse vector: indices and values length mismatch
			{"id": "bad1", "vector": []float32{1.0, 2.0}, "sparse_vector": map[string]interface{}{
				"indices": []int{0, 1},
				"values":  []float32{0.5}, // length mismatch
			}},
			{"id": "ok2", "vector": []float32{3.0, 4.0}},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)

	var env struct {
		Success bool `json:"success"`
		Data    struct {
			InsertedCount int                      `json:"inserted_count"`
			Errors        []map[string]interface{} `json:"errors"`
		} `json:"data"`
	}
	s.Require().NoError(json.Unmarshal(w.Body.Bytes(), &env))
	s.Assert().True(env.Success)
	s.Assert().Equal(2, env.Data.InsertedCount)
	s.Require().Len(env.Data.Errors, 1)
	s.Assert().Equal("bad1", env.Data.Errors[0]["id"])
}

func (s *APITestSuite) TestBatchInsert_EmptyVectors() {
	body, _ := json.Marshal(map[string]interface{}{
		"vectors": []map[string]interface{}{},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusBadRequest, w.Code)
}

func (s *APITestSuite) TestFlushEndpoint() {
	// Insert something first so the snapshot is non-trivial
	body, _ := json.Marshal(map[string]interface{}{
		"id":     "flush_v1",
		"vector": []float32{1.0, 2.0, 3.0},
	})
	insertReq := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(body))
	insertReq.Header.Set("Content-Type", "application/json")
	insertW := httptest.NewRecorder()
	s.router.ServeHTTP(insertW, insertReq)
	s.Require().Equal(http.StatusCreated, insertW.Code)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/flush", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Assert().Equal(http.StatusOK, w.Code)
	s.Assert().JSONEq(`{"success":true,"data":{"status":"flushed"}}`, w.Body.String())
}

func TestAPITestSuite(t *testing.T) {
	suite.Run(t, new(APITestSuite))
}
