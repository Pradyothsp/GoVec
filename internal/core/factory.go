package core

import (
	"fmt"

	"github.com/Pradyothsp/govec/internal/config"
)

// NewEngine creates a new vector engine based on the configuration
func NewEngine(cfg config.EngineConfig, wal *WAL) (Engine, error) {
	// Validate config
	if cfg.Quantization != "none" {
		return nil, fmt.Errorf("only 'none' quantization supported in Stage 1, got '%s'", cfg.Quantization)
	}

	// Get math functions
	mathBlock, err := resolveMetric(cfg.DistanceMetric)
	if err != nil {
		return nil, err
	}

	// Build engine (only float32 in Stage 1)
	identityFunc := func(v []float32) []float32 { return v }

	return NewVectorIndex[[]float32](
		wal,
		identityFunc,
		mathBlock.FloatFunc,
	), nil
}
