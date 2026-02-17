package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLoader(t *testing.T) {
	loader := NewLoader("config.yaml")
	assert.NotNil(t, loader)
	assert.Equal(t, "config.yaml", loader.configPath)
}

func TestLoader_Load_WithMissingFile(t *testing.T) {
	// Missing file should fall back to defaults
	loader := NewLoader("nonexistent.yaml")
	cfg, err := loader.Load()

	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Should have default values
	assert.Equal(t, 8000, cfg.Server.Port)
	assert.Equal(t, "./govec_data.bin", cfg.Storage.DataPath)
}

func TestLoader_Load_WithValidYAML(t *testing.T) {
	// Create temporary YAML file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	yamlContent := `
server:
  host: "localhost"
  port: 9000
  shutdown_timeout: 15s

storage:
  data_path: "/tmp/test.bin"
  auto_save_enabled: false
  auto_save_interval: 120s
`

	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	loader := NewLoader(configPath)
	cfg, err := loader.Load()

	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Verify values from YAML
	assert.Equal(t, "localhost", cfg.Server.Host)
	assert.Equal(t, 9000, cfg.Server.Port)
	assert.Equal(t, 15*time.Second, cfg.Server.ShutdownTimeout)

	assert.Equal(t, "/tmp/test.bin", cfg.Storage.DataPath)
	assert.Equal(t, false, cfg.Storage.AutoSaveEnabled)
	assert.Equal(t, 120*time.Second, cfg.Storage.AutoSaveInterval)
}

func TestLoader_Load_WithInvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Invalid YAML syntax
	yamlContent := `
server:
  host: "localhost"
  port: invalid_port
    bad_indent: true
`

	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	loader := NewLoader(configPath)
	cfg, err := loader.Load()

	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "failed to load config file")
}

func TestLoader_Load_WithInvalidConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Valid YAML but invalid config (port out of range)
	yamlContent := `
server:
  host: ""
  port: 99999
  shutdown_timeout: 10s
`

	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	loader := NewLoader(configPath)
	cfg, err := loader.Load()

	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "invalid configuration")
}

func TestLoader_Load_WithEnvironmentVariables(t *testing.T) {
	// Set environment variables
	envVars := map[string]string{
		"GOVEC_SERVER_HOST":        "0.0.0.0",
		"GOVEC_SERVER_PORT":        "7000",
		"GOVEC_SHUTDOWN_TIMEOUT":   "20s",
		"GOVEC_STORAGE_PATH":       "/custom/path.bin",
		"GOVEC_AUTO_SAVE_ENABLED":  "false",
		"GOVEC_AUTO_SAVE_INTERVAL": "90s",
	}

	// Set env vars
	for key, val := range envVars {
		t.Setenv(key, val)
	}

	loader := NewLoader("nonexistent.yaml")
	cfg, err := loader.Load()

	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Verify environment variables were applied
	assert.Equal(t, "0.0.0.0", cfg.Server.Host)
	assert.Equal(t, 7000, cfg.Server.Port)
	assert.Equal(t, 20*time.Second, cfg.Server.ShutdownTimeout)

	assert.Equal(t, "/custom/path.bin", cfg.Storage.DataPath)
	assert.Equal(t, false, cfg.Storage.AutoSaveEnabled)
	assert.Equal(t, 90*time.Second, cfg.Storage.AutoSaveInterval)
}

func TestLoader_Load_EnvironmentOverridesFile(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Create YAML with some values
	yamlContent := `
server:
  host: "localhost"
  port: 9000
  shutdown_timeout: 15s
`

	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	// Set environment variable to override port
	t.Setenv("GOVEC_SERVER_PORT", "7777")

	loader := NewLoader(configPath)
	cfg, err := loader.Load()

	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Port should be from environment, not file
	assert.Equal(t, 7777, cfg.Server.Port)

	// Other values should be from file
	assert.Equal(t, "localhost", cfg.Server.Host)
	assert.Equal(t, 15*time.Second, cfg.Server.ShutdownTimeout)
}

func TestLoader_Load_InvalidEnvironmentVariables(t *testing.T) {
	// Invalid port value
	t.Setenv("GOVEC_SERVER_PORT", "not_a_number")

	loader := NewLoader("nonexistent.yaml")
	cfg, err := loader.Load()

	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Should fall back to default when env var is invalid
	assert.Equal(t, 8000, cfg.Server.Port)
}

func TestLoader_Load_PartialYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Only override server config, leave others as defaults
	yamlContent := `
server:
  port: 9999
`

	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	loader := NewLoader(configPath)
	cfg, err := loader.Load()

	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Server port should be from file
	assert.Equal(t, 9999, cfg.Server.Port)

	// Other values should be defaults
	assert.Equal(t, "", cfg.Server.Host)
	assert.Equal(t, "./govec_data.bin", cfg.Storage.DataPath)
}

func TestLoader_Load_BooleanEnvironmentVariable(t *testing.T) {
	tests := []struct {
		name     string
		value    string
		expected bool
	}{
		{"true lowercase", "true", true},
		{"True uppercase", "True", true},
		{"TRUE all caps", "TRUE", true},
		{"false lowercase", "false", false},
		{"False uppercase", "False", false},
		{"FALSE all caps", "FALSE", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOVEC_AUTO_SAVE_ENABLED", tt.value)

			loader := NewLoader("nonexistent.yaml")
			cfg, err := loader.Load()

			require.NoError(t, err)
			assert.Equal(t, tt.expected, cfg.Storage.AutoSaveEnabled)
		})
	}
}

func TestLoader_Load_InvalidBooleanEnvironmentVariable(t *testing.T) {
	// Invalid boolean value should fall back to default (true)
	t.Setenv("GOVEC_AUTO_SAVE_ENABLED", "yes")

	loader := NewLoader("nonexistent.yaml")
	cfg, err := loader.Load()

	require.NoError(t, err)
	// Since "yes" is not a valid boolean, it should keep the default value (true)
	assert.Equal(t, true, cfg.Storage.AutoSaveEnabled)
}

func TestLoader_Load_DurationParsing(t *testing.T) {
	tests := []struct {
		name     string
		envVar   string
		value    string
		expected time.Duration
	}{
		{"seconds", "GOVEC_SHUTDOWN_TIMEOUT", "30s", 30 * time.Second},
		{"minutes", "GOVEC_AUTO_SAVE_INTERVAL", "2m", 2 * time.Minute},
		{"hours", "GOVEC_SHUTDOWN_TIMEOUT", "1h", 1 * time.Hour},
		{"mixed", "GOVEC_AUTO_SAVE_INTERVAL", "1h30m", 90 * time.Minute},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(tt.envVar, tt.value)

			loader := NewLoader("nonexistent.yaml")
			cfg, err := loader.Load()

			require.NoError(t, err)
			if tt.envVar == "GOVEC_SHUTDOWN_TIMEOUT" {
				assert.Equal(t, tt.expected, cfg.Server.ShutdownTimeout)
			} else {
				assert.Equal(t, tt.expected, cfg.Storage.AutoSaveInterval)
			}
		})
	}
}
