package index

import (
	"fmt"
	"strings"

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

	// Checked before the mmap store opens, so a bad metric leaves nothing open.
	// Each engine then resolves it for its vector type.
	if !core.IsMetric(engineCfg.DistanceMetric) {
		return nil, fmt.Errorf("unknown distance metric '%s', must be one of: %s", engineCfg.DistanceMetric, strings.Join(core.MetricNames(), ", "))
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
		var err error
		mmapStore, err = storage.NewMmapStore(storageCfg.MmapStorePath, engineCfg.Dimensions, isInt8)
		if err != nil {
			return nil, fmt.Errorf("failed to create mmap store: %w", err)
		}
	}

	if engineCfg.IndexType == config.IndexTypeHNSW {
		return newHNSWEngine(engineCfg, wal, invertedIndex, metaIndex, idMapper, mmapStore)
	}

	return newBruteEngine(engineCfg, wal, invertedIndex, metaIndex, idMapper, mmapStore)
}

// newBruteEngine creates a VectorIndex using linear scan search.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func newBruteEngine(cfg config.EngineConfig, wal *WAL, invertedIndex map[uint32][]core.Posting, metaIndex *core.MetadataIndex, idMapper *core.IDMapper, mmapStore *storage.MmapStore) (Engine, error) {
	switch cfg.Quantization {
	case config.QuantizationScalar:
		metric, err := core.ResolveMetric[[]int8](cfg.DistanceMetric)
		if err != nil {
			return nil, err
		}

		idx := NewVectorIndex[[]int8](wal, invertedIndex, idMapper, core.QuantizeVector, metric.Similarity, mmapStore)
		idx.metaIndex = metaIndex
		idx.configure(&cfg)
		return idx, nil

	case config.QuantizationNone:
		metric, err := core.ResolveMetric[[]float32](cfg.DistanceMetric)
		if err != nil {
			return nil, err
		}

		identityFunc := func(v []float32) []float32 { return v }
		idx := NewVectorIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, metric.Similarity, mmapStore)
		idx.metaIndex = metaIndex
		idx.configure(&cfg)
		return idx, nil

	default:
		return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
	}
}

// newHNSWEngine creates an HNSWIndex using approximate nearest neighbor search.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func newHNSWEngine(cfg config.EngineConfig, wal *WAL, invertedIndex map[uint32][]core.Posting, metaIndex *core.MetadataIndex, idMapper *core.IDMapper, mmapStore *storage.MmapStore) (Engine, error) {
	// Falls back to hnsw's own defaults (not a separate hand-copied literal)
	// so this and NewGraph()'s standalone-library default can't drift apart.
	m := cfg.HnswM
	if m <= 0 {
		m = hnsw.DefaultM
	}

	efSearch := cfg.HnswEfSearch
	if efSearch <= 0 {
		efSearch = hnsw.DefaultEfSearch
	}

	efConstruction := cfg.HnswEfConstruction
	if efConstruction <= 0 {
		efConstruction = hnsw.DefaultEfConstruction
	}

	// 0 is a valid, meaningful value for both (see their own doc comments:
	// BatchParallelism 0 = runtime.GOMAXPROCS(0), BatchParallelThreshold 0 =
	// hnsw.DefaultBatchParallelThreshold) -- passed straight through to
	// Graph rather than resolved here, unlike m/efSearch/efConstruction
	// above.
	batchParallelism := cfg.HnswBatchParallelism
	batchParallelThreshold := cfg.HnswBatchParallelThreshold

	if cfg.Quantization == config.QuantizationScalar {
		metric, err := core.ResolveMetric[[]int8](cfg.DistanceMetric)
		if err != nil {
			return nil, err
		}

		idx := NewHNSWIndex[[]int8](wal, invertedIndex, idMapper, core.QuantizeVector, metric,
			m, efSearch, efConstruction, metaIndex, mmapStore)
		idx.setBatchParallelism(batchParallelism, batchParallelThreshold)
		idx.configure(&cfg)
		return idx, nil
	}

	if cfg.Quantization == config.QuantizationNone {
		metric, err := core.ResolveMetric[[]float32](cfg.DistanceMetric)
		if err != nil {
			return nil, err
		}

		identityFunc := func(v []float32) []float32 { return v }
		idx := NewHNSWIndex[[]float32](wal, invertedIndex, idMapper, identityFunc, metric,
			m, efSearch, efConstruction, metaIndex, mmapStore)
		idx.setBatchParallelism(batchParallelism, batchParallelThreshold)
		idx.configure(&cfg)
		return idx, nil
	}

	return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
}
