package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestValidate changes one setting of the defaults per case. A rejected
// setting is reported under its section, with a message that says what is
// allowed; one at the edge of what is allowed still validates.
func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		change  func(c *Config)
		wantErr string // "" means valid
	}{
		{"the defaults", func(c *Config) {}, ""},

		{"server: port 1", func(c *Config) { c.Server.Port = 1 }, ""},
		{"server: port 65535", func(c *Config) { c.Server.Port = 65535 }, ""},
		{"server: port 0", func(c *Config) { c.Server.Port = 0 }, "server config: port must be between 1 and 65535"},
		{"server: port 65536", func(c *Config) { c.Server.Port = 65536 }, "server config: port must be between 1 and 65535"},
		{"server: zero shutdown timeout", func(c *Config) { c.Server.ShutdownTimeout = 0 }, ""},
		{"server: negative shutdown timeout", func(c *Config) { c.Server.ShutdownTimeout = -time.Second }, "server config: shutdown_timeout must be non-negative"},
		// Zero would disable it, leaving slow-header connections open forever.
		{"server: zero read_header_timeout", func(c *Config) { c.Server.ReadHeaderTimeout = 0 }, "server config: read_header_timeout must be positive"},
		{"server: log level in capitals", func(c *Config) { c.Server.LogLevel = "DEBUG" }, ""},
		{"server: unknown log level", func(c *Config) { c.Server.LogLevel = "verbose" }, "server config: invalid log_level"},

		{"storage: empty data path", func(c *Config) { c.Storage.DataPath = "" }, "storage config: data_path cannot be empty"},
		{"storage: blank data path", func(c *Config) { c.Storage.DataPath = "   " }, "storage config: data_path cannot be empty"},
		{"storage: auto-save every second", func(c *Config) { c.Storage.AutoSaveInterval = time.Second }, ""},
		{"storage: auto-save more often than a second", func(c *Config) { c.Storage.AutoSaveInterval = 500 * time.Millisecond }, "storage config: auto_save_interval must be >= 1s"},
		{"storage: any interval with auto-save off", func(c *Config) {
			c.Storage.AutoSaveEnabled, c.Storage.AutoSaveInterval = false, 0
		}, ""},

		{"engine: no index type means brute", func(c *Config) { c.Engine.IndexType = "" }, ""},
		{"engine: unknown index type", func(c *Config) { c.Engine.IndexType = "ivf" }, "engine config: invalid index_type"},
		{"engine: scalar quantization", func(c *Config) { c.Engine.Quantization = QuantizationScalar }, ""},
		{"engine: unknown quantization", func(c *Config) { c.Engine.Quantization = "pq" }, "engine config: invalid quantization"},
		{"engine: empty quantization", func(c *Config) { c.Engine.Quantization = "" }, "engine config: invalid quantization"},
		{"engine: euclidean", func(c *Config) { c.Engine.DistanceMetric = DistanceMetricEuclidean }, ""},
		{"engine: unknown metric", func(c *Config) { c.Engine.DistanceMetric = "manhattan" }, "engine config: invalid distance metric"},
		{"engine: empty metric", func(c *Config) { c.Engine.DistanceMetric = "" }, "engine config: invalid distance metric"},
		// The construction search supplies the pool M neighbours are picked
		// from; a smaller pool silently under-connects every node.
		{"engine: hnsw ef_construction below M", func(c *Config) {
			c.Engine.IndexType, c.Engine.HnswM, c.Engine.HnswEfConstruction = IndexTypeHNSW, 32, 16
		}, "engine config: hnsw_ef_construction (16) must be >= hnsw_m (32)"},
		{"engine: hnsw ef_construction equal to M", func(c *Config) {
			c.Engine.IndexType, c.Engine.HnswM, c.Engine.HnswEfConstruction = IndexTypeHNSW, 32, 32
		}, ""},
		{"engine: negative batch parallelism", func(c *Config) { c.Engine.HnswBatchParallelism = -1 }, "engine config: hnsw_batch_parallelism must be >= 0"},
		{"engine: negative batch threshold", func(c *Config) { c.Engine.HnswBatchParallelThreshold = -1 }, "engine config: hnsw_batch_parallel_threshold must be >= 0"},

		{"grpc: off ignores its settings", func(c *Config) { c.GRPC.Enabled, c.GRPC.Port = false, 0 }, ""},
		{"grpc: on with its defaults", func(c *Config) { c.GRPC.Enabled = true }, ""},
		{"grpc: port out of range", func(c *Config) { c.GRPC.Enabled, c.GRPC.Port = true, 70000 }, "grpc config: port must be between 1 and 65535"},
		{"grpc: same port as http", func(c *Config) { c.GRPC.Enabled, c.GRPC.Port = true, c.Server.Port }, "grpc config: grpc port 9697 must differ from http port 9697"},
		{"grpc: message size under 1 MB", func(c *Config) { c.GRPC.Enabled, c.GRPC.MaxRecvMsgSizeMB = true, 0 }, "grpc config: max_recv_msg_size_mb must be >= 1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			cfg := DefaultConfig()
			tt.change(cfg)

			// Act
			err := Validate(cfg)

			// Assert
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			assert.ErrorContains(t, err, tt.wantErr)
		})
	}
}
