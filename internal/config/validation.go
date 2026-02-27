package config

import (
	"fmt"
	"strings"
	"time"
)

// Validate validates the entire configuration
func Validate(cfg *Config) error {
	if err := ValidateServerConfig(cfg.Server); err != nil {
		return fmt.Errorf("server config: %w", err)
	}

	if err := ValidateStorageConfig(cfg.Storage); err != nil {
		return fmt.Errorf("storage config: %w", err)
	}

	if err := ValidateEngineConfig(cfg.Engine); err != nil {
		return fmt.Errorf("engine config: %w", err)
	}

	if err := ValidateGRPCConfig(cfg.GRPC, cfg.Server.Port); err != nil {
		return fmt.Errorf("grpc config: %w", err)
	}

	return nil
}

var validLogLevels = map[string]bool{
	"trace": true, "debug": true, "info": true, "warn": true, "error": true,
}

// ValidateServerConfig validates server configuration
//
//nolint:gocritic // ServerConfig is passed by value to match existing codebase convention
func ValidateServerConfig(cfg ServerConfig) error {
	// Port must be in valid range
	if cfg.Port < 1 || cfg.Port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535, got %d", cfg.Port)
	}

	// Shutdown timeout must be non-negative
	if cfg.ShutdownTimeout < 0 {
		return fmt.Errorf("shutdown_timeout must be non-negative, got %v", cfg.ShutdownTimeout)
	}

	// ReadHeaderTimeout must be positive (zero disables it, which is a security risk)
	if cfg.ReadHeaderTimeout <= 0 {
		return fmt.Errorf("read_header_timeout must be positive, got %v", cfg.ReadHeaderTimeout)
	}

	// Log level must be one of the known zerolog levels
	if cfg.LogLevel != "" && !validLogLevels[strings.ToLower(cfg.LogLevel)] {
		return fmt.Errorf("invalid log_level: '%s', must be one of: trace, debug, info, warn, error", cfg.LogLevel)
	}

	return nil
}

// ValidateStorageConfig validates storage configuration
func ValidateStorageConfig(cfg StorageConfig) error {
	// Storage path cannot be empty
	if strings.TrimSpace(cfg.DataPath) == "" {
		return fmt.Errorf("data_path cannot be empty")
	}

	// Auto-save interval must be >= 1s when enabled
	if cfg.AutoSaveEnabled && cfg.AutoSaveInterval < time.Second {
		return fmt.Errorf("auto_save_interval must be >= 1s when auto_save_enabled is true, got %v", cfg.AutoSaveInterval)
	}

	return nil
}

// ValidateGRPCConfig validates gRPC server configuration.
// Validation is only applied when Enabled is true.
//
//nolint:gocritic // GRPCConfig is passed by value to match existing codebase convention
func ValidateGRPCConfig(cfg GRPCConfig, httpPort int) error {
	if !cfg.Enabled {
		return nil
	}
	if cfg.Port < 1 || cfg.Port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535, got %d", cfg.Port)
	}
	if cfg.Port == httpPort {
		return fmt.Errorf("grpc port %d must differ from http port %d", cfg.Port, httpPort)
	}
	if cfg.MaxRecvMsgSizeMB < 1 {
		return fmt.Errorf("max_recv_msg_size_mb must be >= 1, got %d", cfg.MaxRecvMsgSizeMB)
	}
	return nil
}

// ValidateEngineConfig validates engine configuration
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func ValidateEngineConfig(cfg EngineConfig) error {
	indexType := cfg.IndexType
	if indexType == "" {
		indexType = IndexTypeBrute
	}

	if indexType != IndexTypeBrute && indexType != IndexTypeHNSW {
		return fmt.Errorf("invalid index_type: '%s', must be 'brute' or 'hnsw'", cfg.IndexType)
	}

	if cfg.Quantization != QuantizationNone && cfg.Quantization != QuantizationScalar {
		return fmt.Errorf("invalid quantization: '%s', must be 'none' or 'scalar'", cfg.Quantization)
	}

	if cfg.DistanceMetric != DistanceMetricCosine {
		return fmt.Errorf("invalid distance metric: '%s', must be 'cosine'", cfg.DistanceMetric)
	}

	return nil
}
