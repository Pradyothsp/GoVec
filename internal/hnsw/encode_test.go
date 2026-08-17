package hnsw

import (
	"bytes"
	"cmp"
	"testing"

	"github.com/stretchr/testify/require"
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

	require.NotNil(t, g1.Distance)
	require.NotNil(t, g2.Distance)
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
		g1.Distance(testVec1, testVec2),
		g2.Distance(testVec1, testVec2),
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

func TestGraph_ExportImport(t *testing.T) {
	g1 := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g1.Add(
			Node[int, []float32]{
				i, randFloats(1),
			},
		)
	}

	buf := &bytes.Buffer{}
	err := g1.Export(buf)
	require.NoError(t, err)

	// Don't use newTestGraph to ensure parameters
	// are imported.
	g2 := &Graph[int, []float32]{}
	err = g2.Import(buf)
	require.NoError(t, err)

	requireGraphApproxEquals(t, g1, g2)

	n1 := g1.Search(
		[]float32{0.5},
		10,
	)

	n2 := g2.Search(
		[]float32{0.5},
		10,
	)

	require.Equal(t, n1, n2)

	verifyGraphNodes(t, g1)
	verifyGraphNodes(t, g2)
}

func TestSavedGraph(t *testing.T) {
	dir := t.TempDir()

	g1, err := LoadSavedGraph[int, []float32](dir + "/graph")
	require.NoError(t, err)
	require.Equal(t, 0, g1.Len())
	g1.Distance = EuclideanDistanceFloat32
	for i := 0; i < 128; i++ {
		g1.Add(
			Node[int, []float32]{
				i, randFloats(1),
			},
		)
	}

	err = g1.Save()
	require.NoError(t, err)

	g2, err := LoadSavedGraph[int, []float32](dir + "/graph")
	require.NoError(t, err)

	requireGraphApproxEquals(t, g1.Graph, g2.Graph)
}

const benchGraphSize = 100

func BenchmarkGraph_Import(b *testing.B) {
	b.ReportAllocs()
	g := newTestGraph[int]()
	for i := 0; i < benchGraphSize; i++ {
		g.Add(
			Node[int, []float32]{
				i, randFloats(256),
			},
		)
	}

	buf := &bytes.Buffer{}
	err := g.Export(buf)
	require.NoError(b, err)

	b.ResetTimer()
	b.SetBytes(int64(buf.Len()))

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rdr := bytes.NewReader(buf.Bytes())
		g := newTestGraph[int]()
		b.StartTimer()
		err = g.Import(rdr)
		require.NoError(b, err)
	}
}

func BenchmarkGraph_Export(b *testing.B) {
	b.ReportAllocs()
	g := newTestGraph[int]()
	for i := 0; i < benchGraphSize; i++ {
		g.Add(
			Node[int, []float32]{
				i, randFloats(256),
			},
		)
	}

	var buf bytes.Buffer
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := g.Export(&buf)
		require.NoError(b, err)
		if i == 0 {
			ln := buf.Len()
			b.SetBytes(int64(ln))
		}
		buf.Reset()
	}
}
