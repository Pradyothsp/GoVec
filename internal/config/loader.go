package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
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

// loadFromEnv loads configuration from environment variables
// Environment variable naming: GOVEC_<SECTION>_<KEY>
func (l *Loader) loadFromEnv(cfg *Config) {
	// Server configuration
	if val := os.Getenv("GOVEC_SERVER_HOST"); val != "" {
		cfg.Server.Host = val
	}
	if val := os.Getenv("GOVEC_SERVER_PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			cfg.Server.Port = port
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_SERVER_PORT, using previous value")
		}
	}
	if val := os.Getenv("GOVEC_SHUTDOWN_TIMEOUT"); val != "" {
		if duration, err := time.ParseDuration(val); err == nil {
			cfg.Server.ShutdownTimeout = duration
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_SHUTDOWN_TIMEOUT, using previous value")
		}
	}
	if val := os.Getenv("GOVEC_API_KEY"); val != "" {
		cfg.Server.APIKey = val
	}
	if val := os.Getenv("GOVEC_LOG_LEVEL"); val != "" {
		cfg.Server.LogLevel = val
	}
	if val := os.Getenv("GOVEC_PPROF_ENABLED"); val != "" {
		switch strings.ToLower(val) {
		case "true":
			cfg.Server.PprofEnabled = true
		case "false":
			cfg.Server.PprofEnabled = false
		default:
			log.Warn().Str("value", val).Msg("failed to parse GOVEC_PPROF_ENABLED, expected 'true' or 'false', using previous value")
		}
	}
	if val := os.Getenv("GOVEC_PPROF_ADDR"); val != "" {
		cfg.Server.PprofAddr = val
	}

	// Engine configuration
	if val := os.Getenv("GOVEC_INDEX_TYPE"); val != "" {
		cfg.Engine.IndexType = IndexType(val)
	}
	if val := os.Getenv("GOVEC_QUANTIZATION"); val != "" {
		cfg.Engine.Quantization = Quantization(val)
	}
	if val := os.Getenv("GOVEC_HNSW_M"); val != "" {
		if n, err := strconv.Atoi(val); err == nil {
			cfg.Engine.HnswM = n
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_HNSW_M, using previous value")
		}
	}
	if val := os.Getenv("GOVEC_HNSW_EF_SEARCH"); val != "" {
		if n, err := strconv.Atoi(val); err == nil {
			cfg.Engine.HnswEfSearch = n
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_HNSW_EF_SEARCH, using previous value")
		}
	}

	// Storage configuration
	if val := os.Getenv("GOVEC_STORAGE_PATH"); val != "" {
		cfg.Storage.DataPath = val
	}
	if val := os.Getenv("GOVEC_AUTO_SAVE_ENABLED"); val != "" {
		switch strings.ToLower(val) {
		case "true":
			cfg.Storage.AutoSaveEnabled = true
		case "false":
			cfg.Storage.AutoSaveEnabled = false
		default:
			log.Warn().Str("value", val).Msg("failed to parse GOVEC_AUTO_SAVE_ENABLED, expected 'true' or 'false', using previous value")
		}
	}
	if val := os.Getenv("GOVEC_AUTO_SAVE_INTERVAL"); val != "" {
		if duration, err := time.ParseDuration(val); err == nil {
			cfg.Storage.AutoSaveInterval = duration
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_AUTO_SAVE_INTERVAL, using previous value")
		}
	}

	// gRPC configuration
	if val := os.Getenv("GOVEC_GRPC_ENABLED"); val != "" {
		switch strings.ToLower(val) {
		case "true":
			cfg.GRPC.Enabled = true
		case "false":
			cfg.GRPC.Enabled = false
		default:
			log.Warn().Str("value", val).Msg("failed to parse GOVEC_GRPC_ENABLED, expected 'true' or 'false', using previous value")
		}
	}
	if val := os.Getenv("GOVEC_GRPC_PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			cfg.GRPC.Port = port
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_GRPC_PORT, using previous value")
		}
	}
	if val := os.Getenv("GOVEC_GRPC_MAX_RECV_MSG_SIZE_MB"); val != "" {
		if n, err := strconv.Atoi(val); err == nil {
			cfg.GRPC.MaxRecvMsgSizeMB = n
		} else {
			log.Warn().Str("value", val).Err(err).Msg("failed to parse GOVEC_GRPC_MAX_RECV_MSG_SIZE_MB, using previous value")
		}
	}
}
