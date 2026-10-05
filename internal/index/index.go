package index

import (
	"context"

	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/storage"
)

// VectorIndex is a thread-safe in-memory store for vector embeddings, searched
// by an exact linear scan. T is the vector storage type (e.g., []float32 for no
// quantization, []int8 for scalar quantization). Everything about storing a
// record is in the embedded records; VectorIndex adds only the scan.
type VectorIndex[T any] struct {
	records[T]
	distanceFunc func(T, T) (float32, error) // similarity between a query and a stored vector, higher = closer
}

// NewVectorIndex creates an empty VectorIndex ready for use.
// Pass a non-nil vectorStore to enable mmap-backed vector storage.
func NewVectorIndex[T any](wal *WAL, invertedIndex map[uint32][]core.Posting, idMapper *core.IDMapper, encodeFunc func([]float32) T, distanceFunc func(T, T) (float32, error), vectorStore *storage.MmapStore) *VectorIndex[T] {
	return &VectorIndex[T]{
		records: records[T]{
			Store:         make(map[uint32]*core.VectorNode[T]),
			InvertedIndex: invertedIndex,
			IDMapper:      idMapper,
			indexType:     "brute",
			wal:           wal,
			encodeFunc:    encodeFunc,
			vectorStore:   vectorStore,
		},
		distanceFunc: distanceFunc,
	}
}

// Insert adds or updates a vector in the index with thread-safety.
func (idx *VectorIndex[T]) Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	return idx.insert(ctx, id, vec, sparse, meta, idx.insertInternal)
}

// BatchInsert adds or updates multiple vectors with a single WAL fsync for the
// whole batch. If writing the batch to the WAL fails, no item is applied and the
// batch aborts entirely — see WAL.WriteEntries for the atomicity this relies on.
func (idx *VectorIndex[T]) BatchInsert(ctx context.Context, items []BatchInsertItem) ([]BatchInsertError, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	accepted, failures, err := idx.logBatch(ctx, items)
	if err != nil {
		return nil, err
	}

	for _, item := range accepted {
		if err := idx.insertInternal(item.ID, item.Vector, item.Sparse, item.Meta); err != nil {
			zerolog.Ctx(ctx).Error().Err(err).Str("id", item.ID).Msg("failed to apply batch insert item")
			failures = append(failures, BatchInsertError{ID: item.ID, Err: err})
		}
	}
	return failures, nil
}

// insertInternal applies an insert without writing it to the WAL.
// PRECONDITION: idx.mu.Lock() must be held by caller.
func (idx *VectorIndex[T]) insertInternal(id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	_, _, err := idx.put(id, vec, sparse, meta)
	return err
}

// Search returns the top-limit nearest neighbours to query, scoring every
// stored vector that passes the filters. With a sparse query and hybrid search
// on, scores blend dense and sparse similarity (see records.scorer).
func (idx *VectorIndex[T]) Search(_ context.Context, query []float32, sparseQuery core.SparseVector, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encodedQuery, err := idx.encodeQuery(query, sparseQuery)
	if err != nil {
		return nil, err
	}

	var candidates []*core.VectorNode[T]
	if allowlist, indexed := idx.filterAllowlist(filters); indexed {
		if len(allowlist) == 0 {
			return nil, nil
		}
		candidates = make([]*core.VectorNode[T], 0, len(allowlist))
		for id := range allowlist {
			if node, ok := idx.Store[id]; ok {
				candidates = append(candidates, node)
			}
		}
	} else {
		candidates = make([]*core.VectorNode[T], 0, len(idx.Store))
		for _, node := range idx.Store {
			if core.MatchFilter(node.Metadata, filters) {
				candidates = append(candidates, node)
			}
		}
	}

	score := idx.scorer(sparseQuery)
	results := make([]SearchResult, 0, len(candidates))
	for _, node := range candidates {
		dense, err := idx.distanceFunc(encodedQuery, node.Vector)
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{ID: node.ExternalID, Score: score(node.InternalID, dense), Meta: node.Metadata})
	}

	return rankTop(results, limit), nil
}

// Delete removes a vector from the index by ID.
func (idx *VectorIndex[T]) Delete(ctx context.Context, id string) (bool, error) {
	return idx.delete(ctx, id, idx.deleteInternal)
}

// deleteInternal applies a delete without writing it to the WAL, returning
// core.ErrNotFound if the ID doesn't exist.
// PRECONDITION: idx.mu.Lock() must be held by caller.
func (idx *VectorIndex[T]) deleteInternal(id string) error {
	return idx.remove(id, nil)
}

// ReplayWAL reads the WAL file and applies changes to the index.
func (idx *VectorIndex[T]) ReplayWAL(walPath string) error {
	return idx.replay(walPath, idx.insertInternal, idx.deleteInternal)
}

// Reset removes every vector durably (see Engine.Reset).
func (idx *VectorIndex[T]) Reset(ctx context.Context, snapshotPath string) error {
	err := idx.reset(snapshotPath, func(path string) error {
		return idx.writeSnapshot(path, idx.emptyContents())
	}, nil)
	if err == nil {
		zerolog.Ctx(ctx).Info().Str("path", snapshotPath).Msg("index reset")
	}
	return err
}
