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

	return nil
}

// ValidateServerConfig validates server configuration
func ValidateServerConfig(cfg ServerConfig) error {
	// Port must be in valid range
	if cfg.Port < 1 || cfg.Port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535, got %d", cfg.Port)
	}

	// Shutdown timeout must be non-negative
	if cfg.ShutdownTimeout < 0 {
		return fmt.Errorf("shutdown_timeout must be non-negative, got %v", cfg.ShutdownTimeout)
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
