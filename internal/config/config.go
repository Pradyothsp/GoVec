package config

import (
	"fmt"
	"time"
)

// Config is the root configuration aggregate
type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Storage StorageConfig `yaml:"storage"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	Host            string        `yaml:"host"`
	Port            int           `yaml:"port"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`
}

// Address returns the server address in "host:port" format
func (s ServerConfig) Address() string {
	return fmt.Sprintf("%s:%d", s.Host, s.Port)
}

// StorageConfig holds persistence configuration
type StorageConfig struct {
	DataPath         string        `yaml:"data_path"`
	AutoSaveEnabled  bool          `yaml:"auto_save_enabled"`
	AutoSaveInterval time.Duration `yaml:"auto_save_interval"`
}

// DefaultConfig returns configuration with sensible defaults
// matching the current hardcoded behavior
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:            "",
			Port:            8000,
			ShutdownTimeout: 10 * time.Second,
		},
		Storage: StorageConfig{
			DataPath:         "./govec_data.bin",
			AutoSaveEnabled:  true,
			AutoSaveInterval: 60 * time.Second,
		},
	}
}
