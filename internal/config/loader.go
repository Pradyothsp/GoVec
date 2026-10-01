package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Loader is responsible for loading configuration from multiple sources
type Loader struct {
	configPath string
}

// NewLoader creates a new configuration loader
func NewLoader(configPath string) *Loader {
	return &Loader{
		configPath: configPath,
	}
}

// Load loads configuration with the following priority:
// 1. Environment variables (highest)
// 2. YAML config file
// 3. Built-in defaults (lowest)
func (l *Loader) Load() (*Config, error) {
	// Start with defaults
	cfg := DefaultConfig()

	// Load from file if it exists
	if err := l.loadFromFile(cfg); err != nil {
		return nil, fmt.Errorf("failed to load config file: %w", err)
	}

	// Override with environment variables
	l.loadFromEnv(cfg)

	// Validate final configuration
	if err := Validate(cfg); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return cfg, nil
}

// loadFromFile loads configuration from YAML file
// Missing file is not an error - gracefully fall back to defaults
func (l *Loader) loadFromFile(cfg *Config) error {
	data, err := os.ReadFile(l.configPath)
	if err != nil {
		if os.IsNotExist(err) {
			// File not found is not an error
			return nil
		}
		return err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("failed to parse YAML: %w", err)
	}

	return nil
}

// loadFromEnv applies GOVEC_* environment overrides on top of the YAML config.
//
// Every field in Config has one, so a deployment that can only set environment
// variables -- a container, most obviously -- can configure the server without
// mounting a config.yaml. The variable name is GOVEC_ plus the YAML key, so
// storage.wal_path is GOVEC_WAL_PATH; the section name appears only where the
// key alone would be ambiguous (server.port vs grpc.port).
//
// See env.go for the shared read/parse/warn behaviour.
func (l *Loader) loadFromEnv(cfg *Config) {
	// Server
	envString("GOVEC_SERVER_HOST", &cfg.Server.Host)
	envInt("GOVEC_SERVER_PORT", &cfg.Server.Port)
	envDuration("GOVEC_SHUTDOWN_TIMEOUT", &cfg.Server.ShutdownTimeout)
	envDuration("GOVEC_READ_HEADER_TIMEOUT", &cfg.Server.ReadHeaderTimeout)
	envString("GOVEC_API_KEY", &cfg.Server.APIKey)
	envString("GOVEC_LOG_LEVEL", &cfg.Server.LogLevel)
	envBool("GOVEC_PPROF_ENABLED", &cfg.Server.PprofEnabled)
	envString("GOVEC_PPROF_ADDR", &cfg.Server.PprofAddr)

	// Storage
	envString("GOVEC_DATA_PATH", &cfg.Storage.DataPath)
	envString("GOVEC_WAL_PATH", &cfg.Storage.WalPath)
	envBool("GOVEC_AUTO_SAVE_ENABLED", &cfg.Storage.AutoSaveEnabled)
	envDuration("GOVEC_AUTO_SAVE_INTERVAL", &cfg.Storage.AutoSaveInterval)
	envBool("GOVEC_ENABLE_MMAP", &cfg.Storage.EnableMmap)
	envString("GOVEC_MMAP_STORE_PATH", &cfg.Storage.MmapStorePath)

	// Engine
	envString("GOVEC_INDEX_TYPE", &cfg.Engine.IndexType)
	envString("GOVEC_QUANTIZATION", &cfg.Engine.Quantization)
	envString("GOVEC_DISTANCE_METRIC", &cfg.Engine.DistanceMetric)
	envInt("GOVEC_DIMENSIONS", &cfg.Engine.Dimensions)
	envBool("GOVEC_ENABLE_HYBRID_SEARCH", &cfg.Engine.EnableHybridSearch)
	envBool("GOVEC_ENABLE_METADATA_INDEX", &cfg.Engine.EnableMetadataIndex)
	envInt("GOVEC_HNSW_M", &cfg.Engine.HnswM)
	envInt("GOVEC_HNSW_EF_SEARCH", &cfg.Engine.HnswEfSearch)
	envInt("GOVEC_HNSW_EF_CONSTRUCTION", &cfg.Engine.HnswEfConstruction)
	envInt("GOVEC_HNSW_BATCH_PARALLELISM", &cfg.Engine.HnswBatchParallelism)
	envInt("GOVEC_HNSW_BATCH_PARALLEL_THRESHOLD", &cfg.Engine.HnswBatchParallelThreshold)

	// gRPC
	envBool("GOVEC_GRPC_ENABLED", &cfg.GRPC.Enabled)
	envInt("GOVEC_GRPC_PORT", &cfg.GRPC.Port)
	envInt("GOVEC_GRPC_MAX_RECV_MSG_SIZE_MB", &cfg.GRPC.MaxRecvMsgSizeMB)
}
