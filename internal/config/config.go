package config

import (
	"fmt"
	"time"
)

// IndexType selects the vector index algorithm.
type IndexType string

const (
	// IndexTypeBrute uses a linear scan O(n) — exact results, simpler, default.
	IndexTypeBrute IndexType = "brute"
	// IndexTypeHNSW uses a Hierarchical Navigable Small World graph O(log n) — approximate, faster at scale.
	IndexTypeHNSW IndexType = "hnsw"
)

// Quantization selects the vector storage precision.
type Quantization string

const (
	// QuantizationNone stores vectors as float32 (4 bytes/dim).
	QuantizationNone Quantization = "none"
	// QuantizationScalar stores vectors as int8 (1 byte/dim), ~4x memory reduction.
	QuantizationScalar Quantization = "scalar"
)

// DistanceMetric selects the similarity measure.
type DistanceMetric string

const (
	// DistanceMetricCosine measures the angle between vectors (ignores magnitude).
	DistanceMetricCosine DistanceMetric = "cosine"
)

// EngineConfig holds vector engine configuration
type EngineConfig struct {
	IndexType          IndexType      `yaml:"index_type"`
	Quantization       Quantization   `yaml:"quantization"`
	DistanceMetric     DistanceMetric `yaml:"distance_metric"`
	EnableHybridSearch bool           `yaml:"enable_hybrid_search"`
	HnswM              int            `yaml:"hnsw_m"`
	HnswEfSearch       int            `yaml:"hnsw_ef_search"`
}

// Config is the root configuration aggregate
type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Storage StorageConfig `yaml:"storage"`
	Engine  EngineConfig  `yaml:"engine"`
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
	WalPath          string        `yaml:"wal_path"`
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
			WalPath:          "./govec.wal",
			AutoSaveEnabled:  true,
			AutoSaveInterval: 60 * time.Second,
		},
		Engine: EngineConfig{
			IndexType:          IndexTypeBrute,
			Quantization:       QuantizationNone,
			DistanceMetric:     DistanceMetricCosine,
			EnableHybridSearch: false,
			HnswM:              16,
			HnswEfSearch:       20,
		},
	}
}
