package core

// posting is one document's weight for one token.
type posting struct {
	docID  uint32
	weight float32
}

// InvertedIndex maps each sparse token to the documents that carry it and
// their weight for it, like the index at the back of a book. Hybrid search
// reads it to score only the documents that share a token with the query:
// every other document's sparse score is 0, so it needn't be looked at.
//
// Sparse scoring used to compute a dot product against every stored
// document's sparse vector on every query, while this index was kept up to
// date and never read. That made hybrid search linear in the store even
// under HNSW.
//
// NOT thread-safe -- callers must hold the owning index's lock.
type InvertedIndex struct {
	postings map[uint32][]posting
}

// NewInvertedIndex returns an empty InvertedIndex ready for use.
func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{postings: make(map[uint32][]posting)}
}

// Add records docID under each of sv's tokens. If docID is being updated,
// Remove must be called first with its old sparse vector.
func (ix *InvertedIndex) Add(docID uint32, sv SparseVector) {
	for i, token := range sv.Indices {
		ix.postings[token] = append(ix.postings[token], posting{docID: docID, weight: sv.Values[i]})
	}
}

// Remove deletes docID from each of sv's tokens, dropping any token it leaves
// with no documents. Removing a docID that isn't there is a no-op.
func (ix *InvertedIndex) Remove(docID uint32, sv SparseVector) {
	for _, token := range sv.Indices {
		kept := ix.postings[token][:0:0]
		for _, p := range ix.postings[token] {
			if p.docID != docID {
				kept = append(kept, p)
			}
		}

		if len(kept) == 0 {
			delete(ix.postings, token)
		} else {
			ix.postings[token] = kept
		}
	}
}

// Clear removes every document.
func (ix *InvertedIndex) Clear() {
	clear(ix.postings)
}

// Scores returns query's dot product with every document that shares at least
// one token with it, keyed by docID. A document that shares none is absent:
// its dot product is 0. Each score equals query.Dot of the document's sparse
// vector exactly -- the products are added in the same token order.
func (ix *InvertedIndex) Scores(query SparseVector) map[uint32]float32 {
	scores := make(map[uint32]float32)
	for i, token := range query.Indices {
		for _, p := range ix.postings[token] {
			scores[p.docID] += query.Values[i] * p.weight
		}
	}

	return scores
}

// Tokens returns how many distinct tokens have at least one document.
func (ix *InvertedIndex) Tokens() int {
	return len(ix.postings)
}

// Postings returns how many (token, document) pairs the index holds.
func (ix *InvertedIndex) Postings() int {
	n := 0
	for _, ps := range ix.postings {
		n += len(ps)
	}

	return n
}
