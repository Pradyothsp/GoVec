package hnsw

import (
	"bytes"
	"cmp"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/metric"
)

func Test_binaryVarint(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	i := 1337

	n, err := binaryWrite(buf, i)
	require.NoError(t, err)
	require.Equal(t, 2, n)

	// Ensure that binaryRead doesn't read past the
	// varint.
	buf.Write([]byte{0, 0, 0, 0})

	var j int
	_, err = binaryRead(buf, &j)
	require.NoError(t, err)
	require.Equal(t, 1337, j)

	require.Equal(
		t,
		[]byte{0, 0, 0, 0},
		buf.Bytes(),
	)
}

func Test_binaryWrite_string(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	s := "hello"

	n, err := binaryWrite(buf, s)
	require.NoError(t, err)
	// 5 bytes for the string, 1 byte for the length.
	require.Equal(t, 5+1, n)

	var s2 string
	_, err = binaryRead(buf, &s2)
	require.NoError(t, err)
	require.Equal(t, "hello", s2)

	require.Empty(t, buf.Bytes())
}

func verifyGraphNodes[K cmp.Ordered, V VectorType](t *testing.T, g *Graph[K, V]) {
	for _, layer := range g.layers {
		for _, node := range layer.nodes {
			for _, neighbor := range node.neighbors {
				_, ok := layer.nodes[neighbor.Key]
				if !ok {
					t.Errorf(
						"node %v has neighbor %v, but neighbor does not exist",
						node.Key, neighbor.Key,
					)
				}
			}
		}
	}
}

// requireGraphApproxEquals checks that two graphs are equal.
func requireGraphApproxEquals[K cmp.Ordered, V VectorType](t *testing.T, g1, g2 *Graph[K, V]) {
	require.Equal(t, g1.Len(), g2.Len())
	a1 := Analyzer[K, V]{g1}
	a2 := Analyzer[K, V]{g2}

	require.Equal(
		t,
		a1.Topography(),
		a2.Topography(),
	)

	require.Equal(
		t,
		a1.Connectivity(),
		a2.Connectivity(),
	)

	require.NotNil(t, g1.Metric.Distance)
	require.NotNil(t, g2.Metric.Distance)
	// Test distance function by creating sample vectors of the correct type V
	var testVec1, testVec2 V
	switch any(testVec1).(type) {
	case []float32:
		testVec1 = any([]float32{0.5}).(V)
		testVec2 = any([]float32{1}).(V)
	case []int8:
		testVec1 = any([]int8{63}).(V)
		testVec2 = any([]int8{127}).(V)
	}
	require.Equal(
		t,
		g1.Metric.Distance(testVec1, testVec2),
		g2.Metric.Distance(testVec1, testVec2),
	)

	require.Equal(t,
		g1.M,
		g2.M,
	)

	require.Equal(t,
		g1.Ml,
		g2.Ml,
	)

	require.Equal(t,
		g1.EfSearch,
		g2.EfSearch,
	)

	require.NotNil(t, g1.Rng)
	require.NotNil(t, g2.Rng)
}

// roundTripTopology exports g's topology and imports it into a fresh graph
// that gets its vectors from g, as GoVec's snapshot load does from its records.
// The topology carries no search settings, so the fresh graph is given g's,
// as the caller of a real load does.
func roundTripTopology[K cmp.Ordered, V VectorType](t *testing.T, g *Graph[K, V]) *Graph[K, V] {
	t.Helper()

	buf := &bytes.Buffer{}
	require.NoError(t, g.ExportTopology(buf))

	loaded := NewGraph[K, V]()
	loaded.Metric = g.Metric
	loaded.EfSearch = g.EfSearch
	require.NoError(t, loaded.ImportTopology(buf, g.Lookup))

	return loaded
}

func TestGraph_TopologyRoundTrip(t *testing.T) {
	// Arrange
	g1 := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g1.Add(Node[int, []float32]{i, randFloats(1)})
	}

	// Act
	g2 := roundTripTopology(t, g1)

	// Assert
	requireGraphApproxEquals(t, g1, g2)
	require.Equal(t, g1.Search([]float32{0.5}, 10), g2.Search([]float32{0.5}, 10))
	verifyGraphNodes(t, g2)
}

func TestGraph_Int8_TopologyRoundTrip(t *testing.T) {
	// Arrange
	g1 := NewGraph[uint32, []int8]()
	g1.Metric = metric.Metric[[]int8]{Distance: metric.CosineDistanceInt8}
	g1.M = 8
	g1.Ml = 0.25
	g1.EfSearch = 16
	for i := uint32(0); i < 50; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8((i*7+uint32(j)*3)%256 - 128)
		}
		g1.Add(Node[uint32, []int8]{Key: i, Value: vec})
	}

	query := make([]int8, 128)
	for j := range query {
		query[j] = int8((j*5)%256 - 128)
	}

	// Act
	g2 := roundTripTopology(t, g1)

	// Assert
	requireGraphApproxEquals(t, g1, g2)
	require.Equal(t, g1.Search(query, 10), g2.Search(query, 10))
}

// The topology restores the graph's structure, not how it is searched: the
// loading graph's distance function and EfSearch survive the import.
func TestGraph_ImportTopology_KeepsSearchSettings(t *testing.T) {
	// Arrange
	saved := newTestGraph[int]()
	for i := 0; i < 10; i++ {
		saved.Add(Node[int, []float32]{i, randFloats(1)})
	}

	buf := &bytes.Buffer{}
	require.NoError(t, saved.ExportTopology(buf))

	loaded := NewGraph[int, []float32]()
	loaded.Metric = metric.Metric[[]float32]{Distance: metric.CosineDistanceFloat32}
	loaded.EfSearch = saved.EfSearch + 57

	// Act
	err := loaded.ImportTopology(buf, saved.Lookup)

	// Assert
	require.NoError(t, err)
	require.Equal(t, saved.EfSearch+57, loaded.EfSearch)
	require.Equal(t, metric.CosineDistanceFloat32([]float32{1}, []float32{0.5}), loaded.Metric.Distance([]float32{1}, []float32{0.5}))
	require.Equal(t, saved.M, loaded.M)
	require.Equal(t, saved.Ml, loaded.Ml)
}

func TestGraph_ImportTopology_MissingVector(t *testing.T) {
	// Arrange
	g1 := newTestGraph[int]()
	for i := 0; i < 10; i++ {
		g1.Add(Node[int, []float32]{i, randFloats(1)})
	}

	buf := &bytes.Buffer{}
	require.NoError(t, g1.ExportTopology(buf))

	noVectors := func(int) ([]float32, bool) { return nil, false }

	// Act
	err := newTestGraph[int]().ImportTopology(buf, noVectors)

	// Assert
	require.ErrorContains(t, err, "no vector found")
}

const benchGraphSize = 100

func newBenchGraph() *Graph[int, []float32] {
	g := newTestGraph[int]()
	for i := 0; i < benchGraphSize; i++ {
		g.Add(Node[int, []float32]{i, randFloats(256)})
	}

	return g
}

func BenchmarkGraph_ImportTopology(b *testing.B) {
	b.ReportAllocs()
	g := newBenchGraph()

	buf := &bytes.Buffer{}
	require.NoError(b, g.ExportTopology(buf))

	b.ResetTimer()
	b.SetBytes(int64(buf.Len()))

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rdr := bytes.NewReader(buf.Bytes())
		loaded := newTestGraph[int]()
		b.StartTimer()
		require.NoError(b, loaded.ImportTopology(rdr, g.Lookup))
	}
}

func BenchmarkGraph_ExportTopology(b *testing.B) {
	b.ReportAllocs()
	g := newBenchGraph()

	var buf bytes.Buffer
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		require.NoError(b, g.ExportTopology(&buf))
		if i == 0 {
			b.SetBytes(int64(buf.Len()))
		}
		buf.Reset()
	}
}
