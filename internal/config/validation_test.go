package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestValidate(t *testing.T) {
	t.Run("valid default config", func(t *testing.T) {
		cfg := DefaultConfig()
		err := Validate(cfg)
		assert.NoError(t, err)
	})

	t.Run("invalid server config", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.Server.Port = 99999
		err := Validate(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "server config")
	})

	t.Run("invalid storage config", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.Storage.DataPath = ""
		err := Validate(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "storage config")
	})

	t.Run("invalid engine config", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.Engine.Quantization = "invalid"
		err := Validate(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "engine config")
	})
}

func TestValidateEngineConfig(t *testing.T) {
	tests := []struct {
		name      string
		cfg       EngineConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "valid none quantization",
			cfg: EngineConfig{
				Quantization:   "none",
				DistanceMetric: "cosine",
			},
			wantError: false,
		},
		{
			name: "valid scalar quantization",
			cfg: EngineConfig{
				Quantization:   "scalar",
				DistanceMetric: "cosine",
			},
			wantError: false,
		},
		{
			name: "invalid quantization type",
			cfg: EngineConfig{
				Quantization:   "invalid",
				DistanceMetric: "cosine",
			},
			wantError: true,
			errorMsg:  "invalid quantization",
		},
		{
			name: "invalid distance metric",
			cfg: EngineConfig{
				Quantization:   "none",
				DistanceMetric: "euclidean",
			},
			wantError: true,
			errorMsg:  "invalid distance metric",
		},
		{
			name: "empty quantization",
			cfg: EngineConfig{
				Quantization:   "",
				DistanceMetric: "cosine",
			},
			wantError: true,
			errorMsg:  "invalid quantization",
		},
		{
			name: "empty distance metric",
			cfg: EngineConfig{
				Quantization:   "none",
				DistanceMetric: "",
			},
			wantError: true,
			errorMsg:  "invalid distance metric",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateEngineConfig(tt.cfg)
			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateServerConfig(t *testing.T) {
	tests := []struct {
		name      string
		cfg       ServerConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "valid config",
			cfg: ServerConfig{
				Host:              "",
				Port:              8000,
				ShutdownTimeout:   10 * time.Second,
				ReadHeaderTimeout: 10 * time.Second,
			},
			wantError: false,
		},
		{
			name: "port too low",
			cfg: ServerConfig{
				Port: 0,
			},
			wantError: true,
			errorMsg:  "port must be between 1 and 65535",
		},
		{
			name: "port too high",
			cfg: ServerConfig{
				Port: 65536,
			},
			wantError: true,
			errorMsg:  "port must be between 1 and 65535",
		},
		{
			name: "negative shutdown timeout",
			cfg: ServerConfig{
				Port:              8000,
				ShutdownTimeout:   -1 * time.Second,
				ReadHeaderTimeout: 10 * time.Second,
			},
			wantError: true,
			errorMsg:  "shutdown_timeout must be non-negative",
		},
		{
			name: "zero shutdown timeout is allowed",
			cfg: ServerConfig{
				Port:              8000,
				ShutdownTimeout:   0,
				ReadHeaderTimeout: 10 * time.Second,
			},
			wantError: false,
		},
		{
			name: "very short shutdown timeout is allowed",
			cfg: ServerConfig{
				Port:              8000,
				ShutdownTimeout:   100 * time.Millisecond,
				ReadHeaderTimeout: 10 * time.Second,
			},
			wantError: false,
		},
		{
			name: "minimum port",
			cfg: ServerConfig{
				Port:              1,
				ShutdownTimeout:   1 * time.Second,
				ReadHeaderTimeout: 10 * time.Second,
			},
			wantError: false,
		},
		{
			name: "maximum port",
			cfg: ServerConfig{
				Port:              65535,
				ShutdownTimeout:   1 * time.Second,
				ReadHeaderTimeout: 10 * time.Second,
			},
			wantError: false,
		},
		{
			name: "zero read_header_timeout is invalid",
			cfg: ServerConfig{
				Port:              8000,
				ReadHeaderTimeout: 0,
			},
			wantError: true,
			errorMsg:  "read_header_timeout must be positive",
		},
		{
			name: "negative read_header_timeout is invalid",
			cfg: ServerConfig{
				Port:              8000,
				ReadHeaderTimeout: -1 * time.Second,
			},
			wantError: true,
			errorMsg:  "read_header_timeout must be positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateServerConfig(tt.cfg)
			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateStorageConfig(t *testing.T) {
	tests := []struct {
		name      string
		cfg       StorageConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "valid config",
			cfg: StorageConfig{
				DataPath:         "./govec_data.bin",
				AutoSaveEnabled:  true,
				AutoSaveInterval: 60 * time.Second,
			},
			wantError: false,
		},
		{
			name: "empty data path",
			cfg: StorageConfig{
				DataPath: "",
			},
			wantError: true,
			errorMsg:  "data_path cannot be empty",
		},
		{
			name: "whitespace data path",
			cfg: StorageConfig{
				DataPath: "   ",
			},
			wantError: true,
			errorMsg:  "data_path cannot be empty",
		},
		{
			name: "auto-save interval too short",
			cfg: StorageConfig{
				DataPath:         "./data.bin",
				AutoSaveEnabled:  true,
				AutoSaveInterval: 500 * time.Millisecond,
			},
			wantError: true,
			errorMsg:  "auto_save_interval must be >= 1s",
		},
		{
			name: "auto-save disabled with short interval is allowed",
			cfg: StorageConfig{
				DataPath:         "./data.bin",
				AutoSaveEnabled:  false,
				AutoSaveInterval: 0,
			},
			wantError: false,
		},
		{
			name: "minimum valid auto-save interval",
			cfg: StorageConfig{
				DataPath:         "./data.bin",
				AutoSaveEnabled:  true,
				AutoSaveInterval: 1 * time.Second,
			},
			wantError: false,
		},
		{
			name: "long data path",
			cfg: StorageConfig{
				DataPath:         "/very/long/path/to/govec/data/storage/file.bin",
				AutoSaveEnabled:  true,
				AutoSaveInterval: 60 * time.Second,
			},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateStorageConfig(tt.cfg)
			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
