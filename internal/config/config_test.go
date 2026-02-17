package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	// Server defaults
	assert.Equal(t, "", cfg.Server.Host)
	assert.Equal(t, 8000, cfg.Server.Port)
	assert.Equal(t, 10*time.Second, cfg.Server.ShutdownTimeout)

	// Storage defaults
	assert.Equal(t, "./govec_data.bin", cfg.Storage.DataPath)
	assert.Equal(t, true, cfg.Storage.AutoSaveEnabled)
	assert.Equal(t, 60*time.Second, cfg.Storage.AutoSaveInterval)
}

func TestServerConfig_Address(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		port     int
		expected string
	}{
		{
			name:     "empty host binds to all interfaces",
			host:     "",
			port:     8000,
			expected: ":8000",
		},
		{
			name:     "localhost binding",
			host:     "localhost",
			port:     9000,
			expected: "localhost:9000",
		},
		{
			name:     "specific IP binding",
			host:     "0.0.0.0",
			port:     8080,
			expected: "0.0.0.0:8080",
		},
		{
			name:     "IPv6 address",
			host:     "::1",
			port:     3000,
			expected: "::1:3000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ServerConfig{
				Host: tt.host,
				Port: tt.port,
			}
			assert.Equal(t, tt.expected, cfg.Address())
		})
	}
}
