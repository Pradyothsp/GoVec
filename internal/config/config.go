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
	// DistanceMetricEuclidean measures straight-line distance between vectors (magnitude-sensitive).
	DistanceMetricEuclidean DistanceMetric = "euclidean"
)

// EngineConfig holds vector engine configuration
type EngineConfig struct {
	IndexType           IndexType      `yaml:"index_type"`
	Quantization        Quantization   `yaml:"quantization"`
	DistanceMetric      DistanceMetric `yaml:"distance_metric"`
	Dimensions          int            `yaml:"dimensions"` // Required when storage.enable_mmap is true
	EnableHybridSearch  bool           `yaml:"enable_hybrid_search"`
	EnableMetadataIndex bool           `yaml:"enable_metadata_index"`
	HnswM               int            `yaml:"hnsw_m"`
	HnswEfSearch        int            `yaml:"hnsw_ef_search"`
	HnswEfConstruction  int            `yaml:"hnsw_ef_construction"`
}

// GRPCConfig holds gRPC server configuration.
type GRPCConfig struct {
	Enabled          bool `yaml:"enabled"`              // default: false
	Port             int  `yaml:"port"`                 // default: 50051
	MaxRecvMsgSizeMB int  `yaml:"max_recv_msg_size_mb"` // default: 64
}

// Config is the root configuration aggregate
type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Storage StorageConfig `yaml:"storage"`
	Engine  EngineConfig  `yaml:"engine"`
	GRPC    GRPCConfig    `yaml:"grpc"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	Host              string        `yaml:"host"`
	Port              int           `yaml:"port"`
	ShutdownTimeout   time.Duration `yaml:"shutdown_timeout"`
	ReadHeaderTimeout time.Duration `yaml:"read_header_timeout"`
	APIKey            string        `yaml:"api_key"`
	LogLevel          string        `yaml:"log_level"` // trace, debug, info, warn, error
	PprofEnabled      bool          `yaml:"pprof_enabled"`
	PprofAddr         string        `yaml:"pprof_addr"` // must be localhost-bound in production
}

// Address returns the server address in "host:port" format
func (s *ServerConfig) Address() string {
	return fmt.Sprintf("%s:%d", s.Host, s.Port)
}

// StorageConfig holds persistence configuration
type StorageConfig struct {
	DataPath         string        `yaml:"data_path"`
	WalPath          string        `yaml:"wal_path"`
	AutoSaveEnabled  bool          `yaml:"auto_save_enabled"`
	AutoSaveInterval time.Duration `yaml:"auto_save_interval"`
	EnableMmap       bool          `yaml:"enable_mmap"`     // Feature flag: store vectors in mmap-backed files
	MmapStorePath    string        `yaml:"mmap_store_path"` // Directory for .vec chunk files
}

// DefaultConfig returns configuration with sensible defaults
// matching the current hardcoded behavior
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:              "",
			Port:              8000,
			ShutdownTimeout:   10 * time.Second,
			ReadHeaderTimeout: 10 * time.Second,
			LogLevel:          "info",
			PprofEnabled:      false,
			PprofAddr:         "localhost:6060",
		},
		Storage: StorageConfig{
			DataPath:         "./govec_data.bin",
			WalPath:          "./govec.wal",
			AutoSaveEnabled:  true,
			AutoSaveInterval: 60 * time.Second,
			EnableMmap:       false,
			MmapStorePath:    "./govec_mmap/",
		},
		Engine: EngineConfig{
			IndexType:           IndexTypeBrute,
			Quantization:        QuantizationNone,
			DistanceMetric:      DistanceMetricCosine,
			Dimensions:          0, // 0 = no upfront enforcement; must be >0 when EnableMmap: true
			EnableHybridSearch:  false,
			EnableMetadataIndex: false,
			HnswM:               16,
			HnswEfSearch:        20,
			HnswEfConstruction:  200,
		},
		GRPC: GRPCConfig{
			Enabled:          false,
			Port:             50051,
			MaxRecvMsgSizeMB: 64,
		},
	}
}
