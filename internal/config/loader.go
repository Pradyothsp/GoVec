package config

import (
	"fmt"
	"log" // Added for logging warnings
	"os"
	"strconv"
	"strings"
	"time"

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
			log.Printf("WARNING: Failed to parse GOVEC_SERVER_PORT='%s' as integer: %v. Using previous value.\n", val, err)
		}
	}
	if val := os.Getenv("GOVEC_SHUTDOWN_TIMEOUT"); val != "" {
		if duration, err := time.ParseDuration(val); err == nil {
			cfg.Server.ShutdownTimeout = duration
		} else {
			log.Printf("WARNING: Failed to parse GOVEC_SHUTDOWN_TIMEOUT='%s' as duration: %v. Using previous value.\n", val, err)
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
			log.Printf("WARNING: Failed to parse GOVEC_AUTO_SAVE_ENABLED='%s' as boolean. Expected 'true' or 'false'. Using previous value.\n", val)
		}
	}
	if val := os.Getenv("GOVEC_AUTO_SAVE_INTERVAL"); val != "" {
		if duration, err := time.ParseDuration(val); err == nil {
			cfg.Storage.AutoSaveInterval = duration
		} else {
			log.Printf("WARNING: Failed to parse GOVEC_AUTO_SAVE_INTERVAL='%s' as duration: %v. Using previous value.\n", val, err)
		}
	}
}
