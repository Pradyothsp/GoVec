package index

import (
	"fmt"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
)

// NewEngine creates a new vector engine based on the configuration
func NewEngine(cfg config.EngineConfig, wal *WAL) (Engine, error) {
	// Get math functions
	mathBlock, err := core.ResolveMetric(cfg.DistanceMetric)
	if err != nil {
		return nil, err
	}

	// Create engine based on quantization type
	switch cfg.Quantization {
	case "scalar":
		// Int8 quantization
		if mathBlock.Int8Func == nil {
			return nil, fmt.Errorf("distance metric '%s' does not support scalar quantization", cfg.DistanceMetric)
		}
		return NewVectorIndex[[]int8](
			wal,
			core.QuantizeVector,
			mathBlock.Int8Func,
		), nil

	case "none":
		// No quantization (float32)
		identityFunc := func(v []float32) []float32 { return v }
		return NewVectorIndex[[]float32](
			wal,
			identityFunc,
			mathBlock.FloatFunc,
		), nil

	default:
		return nil, fmt.Errorf("unknown quantization: '%s'", cfg.Quantization)
	}
}
