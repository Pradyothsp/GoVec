package core

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// randomSparse returns a valid sparse vector: up to n strictly ascending
// tokens below 50, with weights of either sign.
func randomSparse(rng *rand.Rand, n int) SparseVector {
	var sv SparseVector
	token := uint32(0)
	for range n {
		token += 1 + rng.Uint32N(5)
		if token >= 50 {
			break
		}
		sv.Indices = append(sv.Indices, token)
		sv.Values = append(sv.Values, rng.Float32()*2-1)
	}

	return sv
}

// Scores must give every document exactly the dot product a full scan gives
// it -- hybrid search's results must not change by reading the index -- and
// leave out only documents whose dot product is 0 because they share no token.
func TestInvertedIndex_Scores_MatchesFullScan(t *testing.T) {
	// Arrange
	rng := rand.New(rand.NewPCG(4, 50)) //nolint:gosec // deterministic test data
	docs := map[uint32]SparseVector{}
	ix := NewInvertedIndex()
	for id := range uint32(200) {
		docs[id] = randomSparse(rng, 8)
		ix.Add(id, docs[id])
	}
	for id := uint32(0); id < 200; id += 3 {
		ix.Remove(id, docs[id])
		delete(docs, id)
	}

	for range 50 {
		query := randomSparse(rng, 6)

		// Act
		scores := ix.Scores(query)

		// Assert
		for id, doc := range docs {
			want := query.Dot(doc)
			got, found := scores[id]
			if !found {
				assert.Zero(t, want, "doc %d left out but its dot product isn't 0", id)
				continue
			}
			assert.Equal(t, want, got, "doc %d", id)
		}
		for id := range scores {
			assert.Contains(t, docs, id, "removed doc %d must not be scored", id)
		}
	}
}

func TestInvertedIndex_AddRemoveCount(t *testing.T) {
	// Arrange
	ix := NewInvertedIndex()
	doc1 := SparseVector{Indices: []uint32{10, 20}, Values: []float32{1, 2}}
	doc2 := SparseVector{Indices: []uint32{10, 30}, Values: []float32{3, 4}}

	// Act
	ix.Add(1, doc1)
	ix.Add(2, doc2)
	tokensAfterAdd, postingsAfterAdd := ix.Tokens(), ix.Postings()
	ix.Remove(1, doc1)

	// Assert
	assert.Equal(t, 3, tokensAfterAdd, "tokens 10, 20 and 30")
	assert.Equal(t, 4, postingsAfterAdd, "token 10 is in both documents")
	assert.Equal(t, 2, ix.Tokens(), "token 20 had only doc 1, so it is dropped")
	assert.Equal(t, 2, ix.Postings())
	assert.Equal(t, map[uint32]float32{2: 3}, ix.Scores(SparseVector{Indices: []uint32{10}, Values: []float32{1}}))
}

func TestInvertedIndex_RemoveMissing_IsNoOp(t *testing.T) {
	// Arrange
	ix := NewInvertedIndex()
	doc := SparseVector{Indices: []uint32{5}, Values: []float32{1}}
	ix.Add(1, doc)

	// Act
	ix.Remove(2, doc)
	ix.Remove(1, SparseVector{Indices: []uint32{99}, Values: []float32{1}})

	// Assert
	assert.Equal(t, 1, ix.Postings())
}

func TestInvertedIndex_Clear(t *testing.T) {
	// Arrange
	ix := NewInvertedIndex()
	ix.Add(1, SparseVector{Indices: []uint32{5, 6}, Values: []float32{1, 1}})

	// Act
	ix.Clear()

	// Assert
	assert.Zero(t, ix.Tokens())
	require.Empty(t, ix.Scores(SparseVector{Indices: []uint32{5}, Values: []float32{1}}))
}
