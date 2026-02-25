package index

import (
	"fmt"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
)

// NewEngine creates a new vector engine based on the configuration
func NewEngine(cfg config.EngineConfig, wal *WAL) (Engine, error) {
	mathBlock, err := core.ResolveMetric(string(cfg.DistanceMetric))
	if err != nil {
		return nil, err
	}

	idMapper := core.NewIDMapper()

	var invertedIndex map[uint32][]core.Posting
	if cfg.EnableHybridSearch {
		invertedIndex = make(map[uint32][]core.Posting)
	}

	var metaIndex *core.MetadataIndex
	if cfg.EnableMetadataIndex {
		metaIndex = core.NewMetadataIndex()
	}

	if cfg.IndexType == config.IndexTypeHNSW {
		return newHNSWEngine(cfg, wal, invertedIndex, metaIndex, idMapper, mathBlock)
	}

	return newBruteEngine(cfg, wal, invertedIndex, metaIndex, idMapper, mathBlock)
}

// newBruteEngine creates a VectorIndex using linear scan search.
func newBruteEngine(cfg config.EngineConfig, wal *WAL, invertedIndex map[uint32][]core.Posting, metaIndex *core.MetadataIndex, idMapper *core.IDMapper, mathBlock core.MathBlock) (Engine, error) {
	switch cfg.Quantization {
	case config.QuantizationScalar:
		if mathBlock.Int8Func == nil {
			return nil, fmt.Errorf("distance metric '%s' does not support scalar quantization", cfg.DistanceMetric)
		}
		idx := NewVectorIndex[[]int8](wal, invertedIndex, idMapper, core.QuantizeVector, mathBlock.Int8Func)
		idx.metaIndex = metaIndex
		return idx, nil

	case config.QuantizationNone:
		identityFunc := func(v []float32) []float32 { return v }
		idx := NewVectorIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, mathBlock.FloatFunc)
		idx.metaIndex = metaIndex
		return idx, nil

	default:
		return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
	}
}

// newHNSWEngine creates an HNSWIndex using approximate nearest neighbor search.
func newHNSWEngine(cfg config.EngineConfig, wal *WAL, invertedIndex map[uint32][]core.Posting, metaIndex *core.MetadataIndex, idMapper *core.IDMapper, mathBlock core.MathBlock) (Engine, error) {
	m := cfg.HnswM
	if m <= 0 {
		m = 16
	}

	efSearch := cfg.HnswEfSearch
	if efSearch <= 0 {
		efSearch = 20
	}

	if cfg.Quantization == config.QuantizationScalar {
		if mathBlock.Int8Func == nil {
			return nil, fmt.Errorf("distance metric '%s' does not support scalar quantization", cfg.DistanceMetric)
		}
		return NewHNSWIndex[[]int8](wal, invertedIndex, idMapper,
			core.QuantizeVector, mathBlock.Int8Func, hnsw.CosineDistanceInt8, m, efSearch, metaIndex), nil
	}

	if cfg.Quantization == config.QuantizationNone {
		identityFunc := func(v []float32) []float32 { return v }
		return NewHNSWIndex[[]float32](wal, invertedIndex, idMapper,
			identityFunc, mathBlock.FloatFunc, hnsw.CosineDistanceFloat32, m, efSearch, metaIndex), nil
	}

	return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
}
