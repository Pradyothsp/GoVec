package hnsw

import (
	"bytes"
	"cmp"
	"math/rand"
	"sort"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func Test_maxLevel(t *testing.T) {
	var m int

	m = maxLevel(0.5, 10)
	require.Equal(t, 4, m)

	m = maxLevel(0.5, 1000)
	require.Equal(t, 11, m)
}

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[int, []float32]{
		Node: Node[int, []float32]{
			Value: []float32{0},
			Key:   0,
		},
		neighbors: []*layerNode[int, []float32]{
			{
				Node: Node[int, []float32]{
					Value: []float32{1},
					Key:   1,
				},
			},
			{
				Node: Node[int, []float32]{
					Value: []float32{2},
					Key:   2,
				},
			},
			{
				Node: Node[int, []float32]{
					Value: []float32{3},
					Key:   3,
				},
				// A slice has no separate "structural" key the way the old
				// map did -- node identity is always node.Key. The old
				// literal here keyed these two map entries 4 and 5 while
				// both nodes' own .Key field read 5; that mismatch had no
				// slice equivalent (and can't happen via real graph
				// construction, where the map key and node.Key are always
				// the same value) so the first entry's Key is corrected to
				// 4, matching its Value{4} -- see the updated best[0]
				// assertion below.
				neighbors: []*layerNode[int, []float32]{
					{
						Node: Node[int, []float32]{
							Value: []float32{4},
							Key:   4,
						},
					},
					{
						Node: Node[int, []float32]{
							Value: []float32{5},
							Key:   5,
						},
					},
				},
			},
		},
	}

	best := entry.search(2, 4, []float32{4}, EuclideanDistanceFloat32, nil, nil)

	// Node 3 (value 3) and node 5 (value 5) are both exactly distance 1 from
	// the target (4) -- a genuine tie, so either is a valid second-nearest.
	require.Len(t, best, 2)
	require.Equal(t, 4, best[0].node.Key, "exact match (distance 0) must be first")
	require.Contains(t, []int{3, 5}, best[1].node.Key, "second slot is a tie between nodes 3 and 5 (both distance 1 from the target)")
}

func newTestGraph[K cmp.Ordered]() *Graph[K, []float32] {
	return &Graph[K, []float32]{
		M:              6,
		Distance:       EuclideanDistanceFloat32,
		Ml:             0.5,
		EfSearch:       20,
		EfConstruction: 200,
		Rng:            rand.New(rand.NewSource(0)),
	}
}

func TestGraph_AddSearch(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()

	for i := 0; i < 128; i++ {
		g.Add(
			Node[int, []float32]{
				Key:   i,
				Value: []float32{float32(i)},
			},
		)
	}

	al := Analyzer[int, []float32]{Graph: g}

	// Layers should be approximately log2(128) = 7
	// Look for an approximate doubling of the number of nodes in each layer.
	require.Equal(t, []int{
		128,
		67,
		28,
		12,
		6,
		2,
		1,
		1,
	}, al.Topography())

	nearest := g.Search(
		[]float32{64.5},
		4,
	)

	// True nearest 4 to 64.5 by distance: 64 (0.5), 65 (0.5), then a tie
	// between 63 and 66 (both 1.5). 62 (2.5) is farther than both and must
	// not appear. Order among these isn't a documented guarantee of Search --
	// layerNode.search returns a bounded max-heap's raw backing array, sorted
	// only at index 0 (the heap root) -- so this checks set membership, not
	// a specific order.
	require.Len(t, nearest, 4)
	require.ElementsMatch(
		t,
		[]Node[int, []float32]{
			{64, []float32{64}},
			{65, []float32{65}},
			{63, []float32{63}},
			{66, []float32{66}},
		},
		nearest,
	)
}

func TestGraph_AddDelete(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g.Add(Node[int, []float32]{
			Key:   i,
			Value: []float32{float32(i)},
		})
	}

	require.Equal(t, 128, g.Len())
	an := Analyzer[int, []float32]{Graph: g}

	preDeleteConnectivity := an.Connectivity()

	// Delete every even node.
	for i := 0; i < 128; i += 2 {
		ok := g.Delete(i)
		require.True(t, ok)
	}

	require.Equal(t, 64, g.Len())

	postDeleteConnectivity := an.Connectivity()

	// Connectivity should stay roughly the same for the lowest layer after
	// delete-and-repair -- not exactly equal. With heuristic neighbor
	// selection, connectivity is content-dependent (redundant same-direction
	// candidates get pruned, so it's no longer always "fill to M"), so
	// repairing after a mass delete lands close to but not bit-for-bit at the
	// pre-delete figure.
	require.InDelta(
		t, preDeleteConnectivity[0],
		postDeleteConnectivity[0],
		0.5,
	)

	t.Run("DeleteNotFound", func(t *testing.T) {
		ok := g.Delete(-1)
		require.False(t, ok)
	})
}

// candidate is a small test helper to build a searchCandidate without going
// through a real search -- selectNeighborsHeuristic only depends on the
// (node, dist-to-query) pairing, not how it was produced.
func candidate(key int, vec []float32, distToQuery float32) searchCandidate[int, []float32] {
	return searchCandidate[int, []float32]{
		node: &layerNode[int, []float32]{Node: Node[int, []float32]{Key: key, Value: vec}},
		dist: distToQuery,
	}
}

func Test_selectNeighborsHeuristic_PrefersDiversityOverRawDistance(t *testing.T) {
	t.Parallel()

	// Query at origin. A and B sit on the same line from the query, very
	// close to each other (B is redundant given A: A is much closer to B
	// than B is to the query). C is farther from the query than both, but in
	// a different direction, so it isn't redundant relative to A.
	//
	// Raw "M closest by distance" would pick {A, B} (distances 1, 1.1) over
	// C (distance 2). The heuristic should instead pick {A, C}, since B adds
	// nothing A doesn't already cover.
	a := candidate(1, []float32{1, 0}, 1.0)
	b := candidate(2, []float32{1.1, 0}, 1.1)
	c := candidate(3, []float32{0, 2}, 2.0)

	selected := selectNeighborsHeuristic([]searchCandidate[int, []float32]{b, a, c}, 2, EuclideanDistanceFloat32)

	require.Len(t, selected, 2)
	assert.Equal(t, 1, selected[0].Key, "nearest candidate (A) should always be selected first")
	assert.Equal(t, 3, selected[1].Key, "diverse-but-farther C should be preferred over redundant-but-closer B")
}

func Test_selectNeighborsHeuristic_NoBackfillWhenCandidatesAreRedundant(t *testing.T) {
	t.Parallel()

	// Same redundant pair as above, but with nothing diverse available.
	// hnswlib's getNeighborsByHeuristic2 has no backfill step, so this must
	// return fewer than m rather than padding back up with the rejected
	// redundant candidate.
	a := candidate(1, []float32{1, 0}, 1.0)
	b := candidate(2, []float32{1.1, 0}, 1.1)

	selected := selectNeighborsHeuristic([]searchCandidate[int, []float32]{a, b}, 2, EuclideanDistanceFloat32)

	require.Len(t, selected, 1, "B is redundant given A, and there's no third candidate to backfill with")
	assert.Equal(t, 1, selected[0].Key)
}

func Test_selectNeighborsHeuristic_FewerCandidatesThanMReturnsAllUnpruned(t *testing.T) {
	t.Parallel()

	a := candidate(1, []float32{1, 0}, 1.0)
	b := candidate(2, []float32{1.1, 0}, 1.1) // would be pruned as redundant if the pool were large enough to prune

	selected := selectNeighborsHeuristic([]searchCandidate[int, []float32]{a, b}, 5, EuclideanDistanceFloat32)

	require.Len(t, selected, 2, "fewer candidates than m means nothing to prune, matching hnswlib's size() < M early return")
}

func Benchmark_HSNW(b *testing.B) {
	b.ReportAllocs()

	sizes := []int{100, 1000, 10000}

	// Use this to ensure that complexity is O(log n) where n = h.Len().
	for _, size := range sizes {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			g := Graph[int, []float32]{}
			g.Ml = 0.5
			g.Distance = EuclideanDistanceFloat32
			for i := 0; i < size; i++ {
				g.Add(Node[int, []float32]{
					Key:   i,
					Value: []float32{float32(i)},
				})
			}
			b.ResetTimer()

			b.Run("Search", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					g.Search(
						[]float32{float32(i % size)},
						4,
					)
				}
			})
		})
	}
}

func randFloats(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = rand.Float32()
	}
	return x
}

func Benchmark_HNSW_1536(b *testing.B) {
	b.ReportAllocs()

	g := newTestGraph[int]()
	const size = 1000
	points := make([]Node[int, []float32], size)
	for i := 0; i < size; i++ {
		points[i] = Node[int, []float32]{
			Key:   i,
			Value: randFloats(1536),
		}
		g.Add(points[i])
	}
	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g.Search(
				points[i%size].Value,
				4,
			)
		}
	})
}

func TestGraph_DefaultCosine(t *testing.T) {
	g := NewGraph[int, []float32]()
	g.Distance = CosineDistanceFloat32
	g.Add(
		Node[int, []float32]{Key: 1, Value: []float32{1, 1}},
		Node[int, []float32]{Key: 2, Value: []float32{0, 1}},
		Node[int, []float32]{Key: 3, Value: []float32{1, -1}},
	)

	neighbors := g.Search(
		[]float32{0.5, 0.5},
		1,
	)

	require.Equal(
		t,
		[]Node[int, []float32]{
			{1, []float32{1, 1}},
		},
		neighbors,
	)
}

func TestGraph_RemoveAllNodes(t *testing.T) {
	var vec = []float32{1}

	for i := 0; i < 10; i++ {
		g := NewGraph[int, []float32]()
		g.Distance = CosineDistanceFloat32
		g.Add(MakeNode(1, vec))
		g.Delete(1)
		g.Add(MakeNode(1, vec))
	}
}

// TestGraph_Int8_AddSearch tests basic int8 vector operations
func TestGraph_Int8_AddSearch(t *testing.T) {
	t.Parallel()

	g := NewGraph[int, []int8]()
	g.Distance = CosineDistanceInt8
	g.M = 6
	g.Ml = 0.5
	g.EfSearch = 20

	// Add int8 vectors (quantized values in range -127 to 127)
	g.Add(
		Node[int, []int8]{Key: 1, Value: []int8{127, 0, 0, 0}},
		Node[int, []int8]{Key: 2, Value: []int8{0, 127, 0, 0}},
		Node[int, []int8]{Key: 3, Value: []int8{0, 0, 127, 0}},
		Node[int, []int8]{Key: 4, Value: []int8{0, 0, 0, 127}},
		Node[int, []int8]{Key: 5, Value: []int8{127, 127, 0, 0}},
	)

	require.Equal(t, 5, g.Len())

	// Search for nearest neighbors
	results := g.Search([]int8{120, 10, 0, 0}, 3)

	require.Len(t, results, 3)
	// First result should be node 1 (closest to [120, 10, 0, 0])
	require.Equal(t, 1, results[0].Key)
	// Other results should be valid neighbors (2 or 5 are both reasonable)
	// HNSW is probabilistic, so exact order may vary
	foundKeys := make(map[int]bool)
	for _, r := range results {
		foundKeys[r.Key] = true
	}
	require.True(t, foundKeys[1], "Should find node 1")
	require.True(t, foundKeys[2] || foundKeys[5], "Should find node 2 or 5")
}

// TestGraph_Int8_ExportImport tests persistence with int8 vectors
func TestGraph_Int8_ExportImport(t *testing.T) {
	g1 := NewGraph[uint32, []int8]()
	g1.Distance = CosineDistanceInt8
	g1.M = 8
	g1.Ml = 0.25
	g1.EfSearch = 16

	// Add test vectors
	for i := uint32(0); i < 50; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8((i*7+uint32(j)*3)%256 - 128)
		}
		g1.Add(Node[uint32, []int8]{
			Key:   i,
			Value: vec,
		})
	}

	// Export to buffer
	buf := &bytes.Buffer{}
	err := g1.Export(buf)
	require.NoError(t, err)

	// Import into new graph
	g2 := NewGraph[uint32, []int8]()
	err = g2.Import(buf)
	require.NoError(t, err)

	// Verify graphs are equivalent
	require.Equal(t, g1.Len(), g2.Len())
	require.Equal(t, g1.M, g2.M)
	require.Equal(t, g1.Ml, g2.Ml)
	require.Equal(t, g1.EfSearch, g2.EfSearch)

	// Verify search results are identical
	query := make([]int8, 128)
	for j := range query {
		query[j] = int8((j*5)%256 - 128)
	}

	results1 := g1.Search(query, 10)
	results2 := g2.Search(query, 10)

	require.Equal(t, results1, results2)
}

// TestGraph_Int8_DistanceAccuracy compares int8 vs float32 distance calculations
// TestGraph_SearchWithDistance_AllowlistFiltersResults verifies that nodes not in
// the allowlist are excluded from results while graph traversal still proceeds normally.
func TestGraph_SearchWithDistance_AllowlistFiltersResults(t *testing.T) {
	g := newTestGraph[int]()
	for i := 1; i <= 10; i++ {
		g.Add(MakeNode(i, []float32{float32(i)}))
	}

	// Allow only even-numbered nodes
	allowlist := map[int]struct{}{2: {}, 4: {}, 6: {}, 8: {}, 10: {}}
	results := g.SearchWithDistance([]float32{5}, 3, allowlist, 0)

	require.NotEmpty(t, results)
	for _, r := range results {
		_, allowed := allowlist[r.Key]
		assert.True(t, allowed, "result key %d should be in allowlist", r.Key)
	}
}

// TestGraph_SearchWithDistance_NilAllowlist_SameAsSearch confirms that nil allowlist
// produces the same results as the unfiltered Search method — backward compatibility.
func TestGraph_SearchWithDistance_NilAllowlist_SameAsSearch(t *testing.T) {
	g := newTestGraph[int]()
	for i := 1; i <= 10; i++ {
		g.Add(MakeNode(i, []float32{float32(i)}))
	}

	unfiltered := g.Search([]float32{5}, 3)
	withNil := g.SearchWithDistance([]float32{5}, 3, nil, 0)

	require.Len(t, withNil, len(unfiltered))
	for i, node := range unfiltered {
		assert.Equal(t, node.Key, withNil[i].Key)
	}
}

// TestGraph_SearchWithDistance_ConnectivityPreservation is the critical correctness test:
// nodes NOT in the allowlist must still be traversed so the graph remains navigable.
// Without this, nodes reachable only via excluded nodes would never be found.
func TestGraph_SearchWithDistance_ConnectivityPreservation(t *testing.T) {
	// Build a graph where the best matching node (key=10, vector=10.0) is only
	// reachable through intermediate nodes that are excluded from the allowlist.
	// With M=2 and deterministic RNG, the graph forms a sparse structure where
	// lower-numbered nodes act as stepping stones.
	g := &Graph[int, []float32]{
		M:        2,
		Distance: EuclideanDistanceFloat32,
		Ml:       0.5,
		EfSearch: 50, // high efSearch to ensure thorough exploration
		Rng:      rand.New(rand.NewSource(42)),
	}
	for i := 1; i <= 10; i++ {
		g.Add(MakeNode(i, []float32{float32(i)}))
	}

	// Allow only nodes 1 and 10 — the path to 10 goes through excluded nodes.
	allowlist := map[int]struct{}{1: {}, 10: {}}
	results := g.SearchWithDistance([]float32{10}, 1, allowlist, 0)

	// Must find node 10 despite it being "behind" excluded nodes.
	require.NotEmpty(t, results, "should find node 10 even though path goes through excluded nodes")
	assert.Equal(t, 10, results[0].Key,
		"closest allowed node to query 10.0 must be node 10")
}

func TestGraph_Int8_DistanceAccuracy(t *testing.T) {
	// Test that int8 cosine distance is reasonably accurate
	aF32 := []float32{0.8, 0.6, 0.0, -0.4}
	bF32 := []float32{0.7, 0.7, 0.1, -0.3}

	// Quantize to int8 (scale by 127)
	aI8 := []int8{102, 76, 0, -51} // 0.8*127≈102, 0.6*127≈76, -0.4*127≈-51
	bI8 := []int8{89, 89, 13, -38} // 0.7*127≈89, 0.1*127≈13, -0.3*127≈-38

	distFloat := CosineDistanceFloat32(aF32, bF32)
	distInt8 := CosineDistanceInt8(aI8, bI8)

	// Distance should be very close (within 0.01)
	require.InDelta(t, distFloat, distInt8, 0.01,
		"Int8 distance should be within 0.01 of float32 distance")

	t.Logf("Float32 distance: %.6f", distFloat)
	t.Logf("Int8 distance:    %.6f", distInt8)
	t.Logf("Difference:       %.6f", distFloat-distInt8)
}

// TestGraph_EfConstruction_AffectsGraphQuality verifies EfConstruction (not EfSearch)
// governs graph-build quality. Both graphs here share M and EfSearch — only
// EfConstruction differs — so any recall difference must come from the construction
// path using EfConstruction (graph.go's Add), not from query-time search breadth.
func TestGraph_EfConstruction_AffectsGraphQuality(t *testing.T) {
	const dim, n, k = 8, 500, 10

	rng := rand.New(rand.NewSource(42))
	vectors := make([][]float32, n)
	for i := range vectors {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		vectors[i] = v
	}

	buildGraph := func(efConstruction int) *Graph[int, []float32] {
		g := NewGraph[int, []float32]()
		g.M = 6 // deliberately low, so construction quality (not connection count) dominates recall
		g.EfConstruction = efConstruction
		g.EfSearch = 200 // held generous and constant across both graphs
		g.Distance = CosineDistanceFloat32
		g.Rng = rand.New(rand.NewSource(1)) // deterministic level assignment for a fair comparison

		nodes := make([]Node[int, []float32], n)
		for i, v := range vectors {
			nodes[i] = MakeNode(i, v)
		}
		g.Add(nodes...)

		return g
	}

	recallAtK := func(g *Graph[int, []float32]) float64 {
		hits, total := 0, 0
		for i, q := range vectors {
			truth := bruteForceKNN(vectors, i, q, k)

			found := 0
			for _, r := range g.Search(q, k+1) { // +1 since the query vector itself may be returned
				if r.Key != i && truth[r.Key] {
					found++
				}
			}
			hits += found
			total += k
		}

		return float64(hits) / float64(total)
	}

	lowRecall := recallAtK(buildGraph(1))
	highRecall := recallAtK(buildGraph(200))

	assert.Greater(t, highRecall, lowRecall,
		"a wider EfConstruction should build a measurably better graph, independent of EfSearch")
	assert.Greater(t, highRecall, 0.6, "EfConstruction=200 should reach reasonably good recall at M=6")
}

// bruteForceKNN returns the set of the k nearest neighbor keys to q by exact
// cosine distance, excluding the query's own index.
func bruteForceKNN(vectors [][]float32, excludeIdx int, q []float32, k int) map[int]bool {
	type cand struct {
		id   int
		dist float32
	}
	cands := make([]cand, 0, len(vectors)-1)
	for j, v := range vectors {
		if j == excludeIdx {
			continue
		}
		cands = append(cands, cand{j, CosineDistanceFloat32(q, v)})
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].dist < cands[b].dist })

	truth := make(map[int]bool, k)
	for _, c := range cands[:k] {
		truth[c.id] = true
	}

	return truth
}
