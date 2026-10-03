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
	// HnswBatchParallelism caps how many items run through Graph.Add's
	// round-based parallel Prepare phase concurrently (see
	// internal/hnsw/graph.go's addRound). 0 (default) resolves to
	// runtime.GOMAXPROCS(0).
	HnswBatchParallelism int `yaml:"hnsw_batch_parallelism"`
	// HnswBatchParallelThreshold is the minimum node count a single Add()
	// call needs before round-based parallelism kicks in at all; below it,
	// every node runs through the serial path instead. 0 (default)
	// resolves to hnsw.DefaultBatchParallelThreshold.
	HnswBatchParallelThreshold int `yaml:"hnsw_batch_parallel_threshold"`
}

// GRPCConfig holds gRPC server configuration.
type GRPCConfig struct {
	Enabled          bool `yaml:"enabled"`              // default: false
	Port             int  `yaml:"port"`                 // default: 9698
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
			Port:              9697,
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
			// Must match internal/hnsw's DefaultM/DefaultEfSearch/DefaultEfConstruction
			// (see that package's doc comments for the reasoning behind each value).
			// Kept as separate literals rather than importing internal/hnsw here --
			// this package is meant to stay independent of any specific index
			// implementation -- so keep them in sync by hand if either changes.
			HnswM:              16,
			HnswEfSearch:       50,
			HnswEfConstruction: 200,
			// 0 = auto (runtime.GOMAXPROCS(0)) / hnsw.DefaultBatchParallelThreshold --
			// see internal/hnsw/graph.go's own doc comments on these fields for
			// the reasoning; kept as separate literals here for the same
			// index-implementation-independence reason as HnswM/HnswEfSearch/
			// HnswEfConstruction above.
			HnswBatchParallelism:       0,
			HnswBatchParallelThreshold: 20,
		},
		GRPC: GRPCConfig{
			Enabled:          false,
			Port:             9698,
			MaxRecvMsgSizeMB: 64,
		},
	}
}
