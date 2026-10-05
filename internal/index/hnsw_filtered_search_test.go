package index

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
)

// TestHNSWFilteredSearch_ReturnsKMatches pins the filtered-search contract: a
// search with a filter returns min(k, matching documents) results, every one
// matching. HNSW is approximate, so this checks how many come back, not which.
//
// The candidates the graph returns are filtered afterwards, and nothing tops
// the result back up to k:
//   - metadata index off (the default): no allowlist, so only k candidates are
//     fetched and roughly k × selectivity survive the filter;
//   - metadata index on, filter above 50%: a flat 2k candidates are fetched,
//     which barely covers k when just over half the documents match.
//
// The metadata-index-on cases at or below 50% take the allowlist path, which
// pushes the filter into graph traversal; they are controls that pass today.
//
// TODO: skipped deliberately -- it documents a known, unfixed shortfall rather
// than guarding a fix. Unskip as the first step of the fix. Five of the seven
// cases fail today; both allowlist-path controls pass, and pass across six
// different graph seeds, so the failures are not seed luck.
//
// Fixing this is two changes, not one, because the two paths know different
// things (docs/hnsw-filtered-search-shortfall.md has the full write-up):
//
//   - metadata index on: len(allowlist) gives exact selectivity, so size the
//     fetch as k/selectivity plus a margin instead of a flat 2k. One search,
//     no retry.
//   - metadata index off: there is no allowlist, so selectivity is unknowable
//     before filtering. This needs fetch-filter-count-refetch, widening until
//     k matches are found or the graph is exhausted.
//
// Note what the assertion below commits to: want is exactly min(k, matching)
// on every query, so the meta_index_off/1pct case demands all 4 of 4 matching
// documents out of 400. That rules out any bounded over-fetch -- a fixed
// multiplier, however large, still fails that row. Relax want if best-effort
// is the intended contract instead.
func TestHNSWFilteredSearch_ReturnsKMatches(t *testing.T) {
	t.Skip("documents an unfixed shortfall; see the TODO above and docs/hnsw-filtered-search-shortfall.md")

	const (
		numDocs    = 400
		k          = 10
		numQueries = 20
	)

	cases := []struct {
		name      string
		metaIndex bool
		matching  int // documents in category "A"
	}{
		{"meta_index_off/90pct", false, 360},
		{"meta_index_off/51pct", false, 204},
		{"meta_index_off/10pct", false, 40},
		{"meta_index_off/1pct", false, 4},
		{"meta_index_on/51pct", true, 204},
		{"meta_index_on/49pct_control", true, 196},
		{"meta_index_on/1pct_control", true, 4},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			idx := newSeededHNSWIndex(t, tc.metaIndex)

			data := rand.New(rand.NewSource(42))
			for i := 0; i < numDocs; i++ {
				category := "B"
				if i < tc.matching {
					category = "A"
				}
				vec := []float32{data.Float32(), data.Float32(), data.Float32(), data.Float32()}
				require.NoError(t, idx.Insert(ctx, fmt.Sprintf("v%d", i), vec, core.SparseVector{},
					map[string]any{"category": category}))
			}

			want := min(k, tc.matching)
			filter := map[string]any{"category": "A"}
			var short []string
			for q := 0; q < numQueries; q++ {
				query := []float32{data.Float32(), data.Float32(), data.Float32(), data.Float32()}
				results, err := idx.Search(ctx, query, core.SparseVector{}, k, filter)
				require.NoError(t, err)

				for _, r := range results {
					assert.Equal(t, "A", r.Meta["category"], "query %d returned a non-matching document", q)
				}
				if len(results) != want {
					short = append(short, fmt.Sprintf("query %d: %d", q, len(results)))
				}
			}

			assert.Empty(t, short,
				"every query must return %d results (%d of %d documents match, k=%d); got: %v",
				want, tc.matching, numDocs, k, short)
		})
	}
}

// newSeededHNSWIndex builds an HNSW index with a fixed level-assignment seed,
// so the graph -- and therefore which queries come back short -- is the same
// on every run.
func newSeededHNSWIndex(t *testing.T, withMetaIndex bool) *HNSWIndex[[]float32] {
	t.Helper()
	var metaIndex *core.MetadataIndex
	if withMetaIndex {
		metaIndex = core.NewMetadataIndex()
	}
	idx := NewHNSWIndex[[]float32](newTestWAL(t), nil, core.NewIDMapper(),
		func(v []float32) []float32 { return v },
		hnsw.CosineDistanceFloat32, nil, nil, nil, nil, 16, 20, 200, metaIndex, nil)
	idx.graph.Rng = rand.New(rand.NewSource(1))
	return idx
}
