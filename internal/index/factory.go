package index

import (
	"fmt"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/metric"
	"github.com/Pradyothsp/govec/internal/storage"
)

// NewEngine creates a new vector engine based on the provided configuration.
// storageCfg.EnableMmap activates mmap-backed vector storage; storageCfg.MmapStorePath
// is the directory for .vec chunk files, and engineCfg.Dimensions must be > 0 when
// EnableMmap is true.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func NewEngine(engineCfg config.EngineConfig, storageCfg config.StorageConfig, wal *WAL) (FileEngine, error) {
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

	// The metric and quantization are checked before the mmap store opens, so
	// bad config leaves nothing open.
	if !metric.IsKnown(engineCfg.DistanceMetric) {
		return nil, fmt.Errorf("unknown distance metric '%s', must be one of: %s", engineCfg.DistanceMetric, strings.Join(metric.Names(), ", "))
	}

	if engineCfg.Quantization != config.QuantizationNone && engineCfg.Quantization != config.QuantizationScalar {
		return nil, fmt.Errorf("unknown quantization: '%s'", engineCfg.Quantization)
	}

	// Create the mmap store once, for whichever engine is built.
	var mmapStore *storage.MmapStore
	if storageCfg.EnableMmap {
		isInt8 := engineCfg.Quantization == config.QuantizationScalar
		var err error
		mmapStore, err = storage.NewMmapStore(storageCfg.MmapStorePath, engineCfg.Dimensions, isInt8)
		if err != nil {
			return nil, fmt.Errorf("failed to create mmap store: %w", err)
		}
	}

	if engineCfg.Quantization == config.QuantizationScalar {
		return newEngine[[]int8](engineCfg, wal, storageCfg.DataPath, mmapStore)
	}
	return newEngine[[]float32](engineCfg, wal, storageCfg.DataPath, mmapStore)
}

// newEngine builds the engine config asks for, storing vectors as T.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func newEngine[T metric.VectorType](cfg config.EngineConfig, wal *WAL, snapshotPath string, mmapStore *storage.MmapStore) (FileEngine, error) {
	opts, err := optionsFrom[T](cfg, wal, snapshotPath, mmapStore)
	if err != nil {
		return nil, err
	}

	if cfg.IndexType == config.IndexTypeHNSW {
		return NewHNSWIndex(opts, graphConfigFrom(cfg)), nil
	}
	return NewVectorIndex(opts), nil
}
