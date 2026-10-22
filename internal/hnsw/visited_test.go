package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Pradyothsp/govec/internal/metric"
)

func TestVisitedSet_Visit(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		// arrange leaves the set as it stands before the visit under test.
		arrange func(s *visitedSet)
		key     int
		want    bool
	}{
		{
			name:    "a key not yet reached is new",
			arrange: func(s *visitedSet) { s.reset() },
			key:     3,
			want:    true,
		},
		{
			name: "a key already reached in this traversal is not",
			arrange: func(s *visitedSet) {
				s.reset()
				s.visit(3)
			},
			key:  3,
			want: false,
		},
		{
			name: "a reset forgets the last traversal",
			arrange: func(s *visitedSet) {
				s.reset()
				s.visit(3)
				s.reset()
			},
			key:  3,
			want: true,
		},
		{
			name: "growing past the end keeps this traversal's marks",
			arrange: func(s *visitedSet) {
				s.reset()
				s.visit(3)
				s.visit(10_000)
			},
			key:  3,
			want: false,
		},
		{
			name: "a wrapped epoch doesn't match a tag from 65,535 traversals ago",
			arrange: func(s *visitedSet) {
				s.tags = make([]uint16, 8)
				s.tags[3] = 1 // left by the traversal whose epoch was 1
				s.epoch = math.MaxUint16
				s.reset()
			},
			key:  3,
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Arrange
			s := &visitedSet{}
			tt.arrange(s)

			// Act
			got := s.visit(tt.key)

			// Assert
			assert.Equal(t, tt.want, got)
			assert.NotZero(t, s.epoch, "epoch 0 would match every slot never written")
		})
	}
}

func TestVisitedSet_NegativeKeyPanics(t *testing.T) {
	t.Parallel()

	// Arrange
	s := &visitedSet{}
	s.reset()

	// Act
	visit := func() { s.visit(-1) }

	// Assert
	assert.PanicsWithValue(t, "hnsw: negative key -1", visit)
}

// TestGraph_ConcurrentSearch_MatchesSerial runs the same queries from several
// goroutines at once. Searches share the graph's pool of visited sets; if two
// ever held the same set, one would skip nodes the other reached and return
// different neighbors. Run under -race it also checks the memory accesses.
func TestGraph_ConcurrentSearch_MatchesSerial(t *testing.T) {
	t.Parallel()

	// Arrange
	const dim, n, k, searchers = 16, 500, 10, 8
	g := NewGraph[uint32, []float32](metric.Metric[[]float32]{Distance: metric.CosineDistanceFloat32}, Config{})
	g.Rng = rand.New(rand.NewSource(5))
	for i, v := range randomVectors(5, n, dim) {
		g.Add(MakeNode(uint32(i), v))
	}
	queries := randomVectors(6, 50, dim)
	want := make([][]Node[uint32, []float32], len(queries))
	for i, q := range queries {
		want[i] = g.Search(q, k)
		require.Len(t, want[i], k)
	}

	// Act
	got := make([][][]Node[uint32, []float32], searchers)
	var wg sync.WaitGroup
	for s := range searchers {
		got[s] = make([][]Node[uint32, []float32], len(queries))
		wg.Go(func() {
			for i, q := range queries {
				got[s][i] = g.Search(q, k)
			}
		})
	}
	wg.Wait()

	// Assert
	for s := range searchers {
		assert.Equal(t, want, got[s], "searcher %d", s)
	}
}
