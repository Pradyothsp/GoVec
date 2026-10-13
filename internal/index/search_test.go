package index

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
)

// The engines differ only in how they find candidates: brute force scans every
// record, HNSW walks its graph (filtering through the metadata index, which
// newTestHNSWIndex turns on). On indexes this small both must return exactly
// the same results, so every case runs on both.

// searchDoc is a record to insert before a search.
type searchDoc struct {
	id     string
	vector []float32
	sparse core.SparseVector
	meta   map[string]any
}

func insertDocs(t *testing.T, idx FileEngine, docs []searchDoc) {
	t.Helper()
	for _, d := range docs {
		require.NoError(t, idx.Insert(context.Background(), d.id, d.vector, d.sparse, d.meta))
	}
}

func resultIDs(results []SearchResult) []string {
	ids := make([]string, 0, len(results))
	for _, r := range results {
		ids = append(ids, r.ID)
	}
	return ids
}

func TestSearch(t *testing.T) {
	// Each document is a distinct angle from the query [1, 0, 0], so the
	// nearest-first order is fixed: a, b, c, d, e.
	docs := []searchDoc{
		{id: "a", vector: []float32{1, 0, 0}, meta: map[string]any{"category": "A", "value": 10, "active": true}},
		{id: "b", vector: []float32{0.9, 0.1, 0}, meta: map[string]any{"category": "B", "value": 20, "active": true}},
		{id: "c", vector: []float32{0.8, 0.2, 0}, meta: map[string]any{"category": "A", "value": 30, "active": false}},
		{id: "d", vector: []float32{0.1, 1, 0}, meta: map[string]any{"category": "B", "value": 20}},
		{id: "e", vector: []float32{0, 0, 1}, meta: map[string]any{"category": "A", "value": 10}},
	}
	tests := []struct {
		name    string
		docs    []searchDoc
		k       int
		filter  map[string]any
		wantIDs []string
	}{
		{name: "empty index", k: 5, wantIDs: []string{}},
		{name: "nearest first, cut to k", docs: docs, k: 3, wantIDs: []string{"a", "b", "c"}},
		{name: "k beyond the index returns every vector", docs: docs, k: 10, wantIDs: []string{"a", "b", "c", "d", "e"}},
		{name: "filter on one key", docs: docs, k: 10, filter: map[string]any{"category": "A"}, wantIDs: []string{"a", "c", "e"}},
		{name: "filter on every key given", docs: docs, k: 10, filter: map[string]any{"category": "B", "value": 20}, wantIDs: []string{"b", "d"}},
		{name: "numbers match across types", docs: docs, k: 10, filter: map[string]any{"value": 10.0}, wantIDs: []string{"a", "e"}},
		{name: "bool filter", docs: docs, k: 10, filter: map[string]any{"active": true}, wantIDs: []string{"a", "b"}},
		{name: "filter is applied before k", docs: docs, k: 2, filter: map[string]any{"category": "A"}, wantIDs: []string{"a", "c"}},
		{name: "filter matching nothing", docs: docs, k: 10, filter: map[string]any{"category": "Z"}, wantIDs: []string{}},
	}
	for _, e := range bothEngines {
		for _, tt := range tests {
			t.Run(e.name+"/"+tt.name, func(t *testing.T) {
				// Arrange
				idx, _ := e.open(t)
				insertDocs(t, idx, tt.docs)

				// Act
				results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{}, tt.k, tt.filter)

				// Assert
				require.NoError(t, err)
				assert.Equal(t, tt.wantIDs, resultIDs(results))
				for i := 1; i < len(results); i++ {
					assert.GreaterOrEqual(t, results[i-1].Score, results[i].Score, "scores must descend")
				}
				if len(results) > 0 && results[0].ID == "a" {
					assert.InDelta(t, 1, results[0].Score, 1e-5, "an identical vector scores 1")
					assert.Equal(t, docs[0].meta, results[0].Meta)
				}
			})
		}
	}
}

func TestSearch_EmptyQuery_Rejected(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			require.NoError(t, idx.Insert(context.Background(), "v1", []float32{1, 0, 0}, core.SparseVector{}, nil))

			// Act
			_, err := idx.Search(context.Background(), []float32{}, core.SparseVector{}, 5, nil)

			// Assert
			assert.ErrorIs(t, err, ErrEmptyVector)
		})
	}
}

// Filters must see a record's current metadata: an overwrite moves it from
// one filter value to another, and a delete takes it out of all of them.
func TestSearch_FiltersFollowWrites(t *testing.T) {
	ctx := context.Background()
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			require.NoError(t, idx.Insert(ctx, "moved", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{"tier": "free"}))
			require.NoError(t, idx.Insert(ctx, "deleted", []float32{0, 1, 0}, core.SparseVector{}, map[string]any{"tier": "gone"}))

			// Act
			require.NoError(t, idx.Insert(ctx, "moved", []float32{1, 0, 0}, core.SparseVector{}, map[string]any{"tier": "premium"}))
			_, err := idx.Delete(ctx, "deleted")
			require.NoError(t, err)

			// Assert
			for tier, want := range map[string][]string{"free": {}, "premium": {"moved"}, "gone": {}} {
				results, err := idx.Search(ctx, []float32{1, 0, 0}, core.SparseVector{}, 5, map[string]any{"tier": tier})
				require.NoError(t, err)
				assert.Equal(t, want, resultIDs(results), "tier %s", tier)
			}
		})
	}
}

// bothHybridEngines opens each engine with hybrid search on.
var bothHybridEngines = []struct {
	name string
	open func(t *testing.T) FileEngine
}{
	{"brute", func(t *testing.T) FileEngine { return newTestIndexWithHybrid(t) }},
	{"hnsw", func(t *testing.T) FileEngine { return newTestHNSWIndexWithHybrid(t) }},
}

// TestHybridSearch: a hybrid score is 0.7 x the dense score plus 0.3 x the
// sparse score (the dot product of the sparse vectors, floored at 0).
func TestHybridSearch(t *testing.T) {
	sparse := func(pairs ...float32) core.SparseVector {
		var sv core.SparseVector
		for i := 0; i < len(pairs); i += 2 {
			sv.Indices = append(sv.Indices, uint32(pairs[i]))
			sv.Values = append(sv.Values, pairs[i+1])
		}
		return sv
	}
	tests := []struct {
		name         string
		docs         []searchDoc
		sparseQuery  core.SparseVector
		filter       map[string]any
		wantIDs      []string
		wantTopScore float32
	}{
		{
			name: "a strong keyword match outranks a slightly closer meaning",
			docs: []searchDoc{
				{id: "close", vector: []float32{1, 0, 0}, sparse: sparse(10, 0.1)},
				{id: "keyword", vector: []float32{0.9, 0.1, 0}, sparse: sparse(10, 10)},
				{id: "far", vector: []float32{0, 1, 0}},
			},
			sparseQuery: sparse(10, 5),
			wantIDs:     []string{"keyword", "close", "far"},
		},
		{
			name: "the sparse score weighs tokens rather than counting them",
			docs: []searchDoc{
				{id: "three_tokens", vector: []float32{1, 0, 0}, sparse: sparse(10, 1, 20, 2, 30, 3)},
				{id: "one_heavy_token", vector: []float32{1, 0, 0}, sparse: sparse(10, 10)},
			},
			sparseQuery: sparse(10, 1, 20, 1, 30, 1),
			wantIDs:     []string{"one_heavy_token", "three_tokens"},
		},
		{
			name: "no shared tokens ranks by meaning",
			docs: []searchDoc{
				{id: "close", vector: []float32{1, 0, 0}, sparse: sparse(10, 1, 20, 2)},
				{id: "further", vector: []float32{0.9, 0.1, 0}, sparse: sparse(30, 1, 40, 2)},
			},
			sparseQuery: sparse(50, 5, 60, 3),
			wantIDs:     []string{"close", "further"},
		},
		{
			name: "an empty sparse query ranks by meaning",
			docs: []searchDoc{
				{id: "close", vector: []float32{1, 0, 0}},
				{id: "keyword", vector: []float32{0.9, 0.1, 0}, sparse: sparse(10, 10)},
			},
			wantIDs:      []string{"close", "keyword"},
			wantTopScore: 1,
		},
		{
			name: "a filter excludes a better hybrid match",
			docs: []searchDoc{
				{id: "books", vector: []float32{1, 0, 0}, sparse: sparse(10, 5), meta: map[string]any{"category": "books"}},
				{id: "music", vector: []float32{0.9, 0.1, 0}, sparse: sparse(10, 8), meta: map[string]any{"category": "music"}},
			},
			sparseQuery: sparse(10, 2),
			filter:      map[string]any{"category": "books"},
			wantIDs:     []string{"books"},
		},
		{
			name:         "a perfect match on both scores 1",
			docs:         []searchDoc{{id: "perfect", vector: []float32{1, 0, 0}, sparse: sparse(10, 1)}},
			sparseQuery:  sparse(10, 1),
			wantIDs:      []string{"perfect"},
			wantTopScore: 1,
		},
	}
	for _, e := range bothHybridEngines {
		for _, tt := range tests {
			t.Run(e.name+"/"+tt.name, func(t *testing.T) {
				// Arrange
				idx := e.open(t)
				insertDocs(t, idx, tt.docs)

				// Act
				results, err := idx.Search(context.Background(), []float32{1, 0, 0}, tt.sparseQuery, len(tt.docs), tt.filter)

				// Assert
				require.NoError(t, err)
				assert.Equal(t, tt.wantIDs, resultIDs(results))
				if tt.wantTopScore != 0 {
					assert.InDelta(t, tt.wantTopScore, results[0].Score, 1e-5)
				}
			})
		}
	}
}

// With hybrid search off, a sparse query is ignored rather than rejected.
func TestHybridSearch_Disabled_IgnoresSparseQuery(t *testing.T) {
	for _, e := range bothEngines {
		t.Run(e.name, func(t *testing.T) {
			// Arrange
			idx, _ := e.open(t)
			insertDocs(t, idx, []searchDoc{
				{id: "close", vector: []float32{1, 0, 0}},
				{id: "keyword", vector: []float32{0.9, 0.1, 0}, sparse: core.SparseVector{Indices: []uint32{10}, Values: []float32{10}}},
			})

			// Act
			results, err := idx.Search(context.Background(), []float32{1, 0, 0}, core.SparseVector{Indices: []uint32{10}, Values: []float32{5}}, 2, nil)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, []string{"close", "keyword"}, resultIDs(results))
			assert.InDelta(t, 1, results[0].Score, 1e-5, "the score is the dense score alone")
		})
	}
}
