package testutil

import (
	"os"
	"path/filepath"
	"time"

	"github.com/Pradyothsp/govec/internal/config"
)

// TestConfig returns configuration suitable for testing
func TestConfig() *config.Config {
	return &config.Config{
		Server: config.ServerConfig{
			Host:            "localhost",
			Port:            0, // Random free port for parallel tests
			ShutdownTimeout: 1 * time.Second,
		},
		Storage: config.StorageConfig{
			DataPath:         filepath.Join(os.TempDir(), "test_govec.bin"),
			AutoSaveEnabled:  false, // Faster tests
			AutoSaveInterval: 0,
		},
	}
}
