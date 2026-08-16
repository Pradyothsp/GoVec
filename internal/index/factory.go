package index

import (
	"fmt"

	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/storage"
)

// NewEngine creates a new vector engine based on the provided configuration.
// storageCfg.EnableMmap activates mmap-backed vector storage; storageCfg.MmapStorePath
// is the directory for .vec chunk files, and engineCfg.Dimensions must be > 0 when
// EnableMmap is true.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func NewEngine(engineCfg config.EngineConfig, storageCfg config.StorageConfig, wal *WAL) (Engine, error) {
	log.Info().
		Str("index_type", string(engineCfg.IndexType)).
		Str("quantization", string(engineCfg.Quantization)).
		Str("distance_metric", string(engineCfg.DistanceMetric)).
		Bool("mmap_enabled", storageCfg.EnableMmap).
		Bool("hybrid_search", engineCfg.EnableHybridSearch).
		Msg("initializing vector engine")

	if storageCfg.EnableMmap && engineCfg.Dimensions <= 0 {
		return nil, fmt.Errorf("engine.dimensions must be set (> 0) when storage.enable_mmap is true")
	}

	mathBlock, err := core.ResolveMetric(string(engineCfg.DistanceMetric))
	if err != nil {
		return nil, err
	}

	idMapper := core.NewIDMapper()

	var invertedIndex map[uint32][]core.Posting
	if engineCfg.EnableHybridSearch {
		invertedIndex = make(map[uint32][]core.Posting)
	}

	var metaIndex *core.MetadataIndex
	if engineCfg.EnableMetadataIndex {
		metaIndex = core.NewMetadataIndex()
	}

	// Create mmap store once and pass to the engine constructor.
	var mmapStore *storage.MmapStore
	if storageCfg.EnableMmap {
		isInt8 := engineCfg.Quantization == config.QuantizationScalar
		mmapStore, err = storage.NewMmapStore(storageCfg.MmapStorePath, engineCfg.Dimensions, isInt8)
		if err != nil {
			return nil, fmt.Errorf("failed to create mmap store: %w", err)
		}
	}

	if engineCfg.IndexType == config.IndexTypeHNSW {
		return newHNSWEngine(engineCfg, wal, invertedIndex, metaIndex, idMapper, mathBlock, mmapStore)
	}

	return newBruteEngine(engineCfg, wal, invertedIndex, metaIndex, idMapper, mathBlock, mmapStore)
}

// newBruteEngine creates a VectorIndex using linear scan search.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func newBruteEngine(cfg config.EngineConfig, wal *WAL, invertedIndex map[uint32][]core.Posting, metaIndex *core.MetadataIndex, idMapper *core.IDMapper, mathBlock core.MathBlock, mmapStore *storage.MmapStore) (Engine, error) {
	switch cfg.Quantization {
	case config.QuantizationScalar:
		if mathBlock.Int8Func == nil {
			return nil, fmt.Errorf("distance metric '%s' does not support scalar quantization", cfg.DistanceMetric)
		}
		idx := NewVectorIndex[[]int8](wal, invertedIndex, idMapper, core.QuantizeVector, mathBlock.Int8Func, mmapStore)
		idx.metaIndex = metaIndex
		idx.quantization = string(cfg.Quantization)
		idx.indexType = "brute"
		idx.distanceMetric = string(cfg.DistanceMetric)
		idx.dimensions = cfg.Dimensions
		return idx, nil

	case config.QuantizationNone:
		identityFunc := func(v []float32) []float32 { return v }
		idx := NewVectorIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, mathBlock.FloatFunc, mmapStore)
		idx.metaIndex = metaIndex
		idx.quantization = string(cfg.Quantization)
		idx.indexType = "brute"
		idx.distanceMetric = string(cfg.DistanceMetric)
		idx.dimensions = cfg.Dimensions
		return idx, nil

	default:
		return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
	}
}

// hnswDistanceFuncFloat32 picks the graph-level distance function matching the
// configured metric. HNSW needs its own DistanceFunc (lower = closer) separate
// from core.MathBlock (higher = closer, used by the brute-force engine).
func hnswDistanceFuncFloat32(metric config.DistanceMetric) (hnsw.DistanceFunc[[]float32], error) {
	switch metric {
	case config.DistanceMetricCosine:
		return hnsw.CosineDistanceFloat32, nil
	case config.DistanceMetricEuclidean:
		return hnsw.EuclideanDistanceFloat32, nil
	default:
		return nil, fmt.Errorf("unsupported distance metric for HNSW: '%s'", metric)
	}
}

// hnswDistanceFuncInt8 is the int8 counterpart of hnswDistanceFuncFloat32.
func hnswDistanceFuncInt8(metric config.DistanceMetric) (hnsw.DistanceFunc[[]int8], error) {
	switch metric {
	case config.DistanceMetricCosine:
		return hnsw.CosineDistanceInt8, nil
	case config.DistanceMetricEuclidean:
		return hnsw.EuclideanDistanceInt8, nil
	default:
		return nil, fmt.Errorf("unsupported distance metric for HNSW: '%s'", metric)
	}
}

// newHNSWEngine creates an HNSWIndex using approximate nearest neighbor search.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func newHNSWEngine(cfg config.EngineConfig, wal *WAL, invertedIndex map[uint32][]core.Posting, metaIndex *core.MetadataIndex, idMapper *core.IDMapper, mathBlock core.MathBlock, mmapStore *storage.MmapStore) (Engine, error) {
	m := cfg.HnswM
	if m <= 0 {
		m = 16
	}

	efSearch := cfg.HnswEfSearch
	if efSearch <= 0 {
		efSearch = 20
	}

	efConstruction := cfg.HnswEfConstruction
	if efConstruction <= 0 {
		efConstruction = 200
	}

	if cfg.Quantization == config.QuantizationScalar {
		if mathBlock.Int8Func == nil {
			return nil, fmt.Errorf("distance metric '%s' does not support scalar quantization", cfg.DistanceMetric)
		}
		graphDistanceFunc, err := hnswDistanceFuncInt8(cfg.DistanceMetric)
		if err != nil {
			return nil, err
		}
		idx := NewHNSWIndex[[]int8](wal, invertedIndex, idMapper,
			core.QuantizeVector, mathBlock.Int8Func, graphDistanceFunc, m, efSearch, efConstruction, metaIndex, mmapStore)
		idx.quantization = string(cfg.Quantization)
		idx.indexType = "hnsw"
		idx.distanceMetric = string(cfg.DistanceMetric)
		idx.dimensions = cfg.Dimensions
		return idx, nil
	}

	if cfg.Quantization == config.QuantizationNone {
		graphDistanceFunc, err := hnswDistanceFuncFloat32(cfg.DistanceMetric)
		if err != nil {
			return nil, err
		}
		identityFunc := func(v []float32) []float32 { return v }
		idx := NewHNSWIndex[[]float32](wal, invertedIndex, idMapper,
			identityFunc, mathBlock.FloatFunc, graphDistanceFunc, m, efSearch, efConstruction, metaIndex, mmapStore)
		idx.quantization = string(cfg.Quantization)
		idx.indexType = "hnsw"
		idx.distanceMetric = string(cfg.DistanceMetric)
		idx.dimensions = cfg.Dimensions
		return idx, nil
	}

	return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
}
