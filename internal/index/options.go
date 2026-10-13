package index

import (
	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/hnsw"
	"github.com/Pradyothsp/govec/internal/metric"
	"github.com/Pradyothsp/govec/internal/storage"
)

// Options is what either engine is built from, besides which engine it is.
// T, the stored vector type, decides the rest: []float32 is stored as given,
// []int8 is scalar-quantized.
//
// An engine used to be built in pieces -- a constructor, then configure, then
// for one engine a metadata index assigned and for the other batch settings --
// so one built without every step quietly ran without part of its config.
type Options[T metric.VectorType] struct {
	WAL           *WAL               // nil: writes aren't logged
	Metric        metric.Metric[T]   // how vectors are compared
	Dimensions    int                // the index's width; 0 learns it from the first insert
	HybridSearch  bool               // keep an inverted index for sparse scoring
	MetadataIndex bool               // keep a metadata index for filtered search
	VectorStore   *storage.MmapStore // nil keeps vectors on the heap
}

// optionsFrom is Options for an engine built from config, logging to wal and
// keeping vectors in store.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func optionsFrom[T metric.VectorType](cfg config.EngineConfig, wal *WAL, store *storage.MmapStore) (Options[T], error) {
	m, err := metric.Resolve[T](cfg.DistanceMetric)
	if err != nil {
		return Options[T]{}, err
	}

	return Options[T]{
		WAL:           wal,
		Metric:        m,
		Dimensions:    cfg.Dimensions,
		HybridSearch:  cfg.EnableHybridSearch,
		MetadataIndex: cfg.EnableMetadataIndex,
		VectorStore:   store,
	}, nil
}

// graphConfigFrom is the HNSW graph config engine config asks for. Zero
// values are passed through for hnsw.NewGraph to default.
//
//nolint:gocritic // EngineConfig is passed by value to match existing codebase convention
func graphConfigFrom(cfg config.EngineConfig) hnsw.Config {
	return hnsw.Config{
		M:                      cfg.HnswM,
		EfSearch:               cfg.HnswEfSearch,
		EfConstruction:         cfg.HnswEfConstruction,
		BatchParallelism:       cfg.HnswBatchParallelism,
		BatchParallelThreshold: cfg.HnswBatchParallelThreshold,
	}
}

// init sets up empty records for an engine of indexType built from opts.
func (r *records[T]) init(opts Options[T], indexType config.IndexType) {
	r.Store = make(map[uint32]*core.VectorNode[T])
	r.IDMapper = core.NewIDMapper()
	if opts.HybridSearch {
		r.inverted = core.NewInvertedIndex()
	}
	if opts.MetadataIndex {
		r.metaIndex = core.NewMetadataIndex()
	}

	r.indexType = indexType
	r.quantization = quantizationOf[T]()
	r.metric = opts.Metric
	r.dimensions = opts.Dimensions
	r.configuredDimensions = opts.Dimensions

	r.wal = opts.WAL
	r.encodeFunc = encoderFor[T]()
	r.vectorStore = opts.VectorStore
}

// encoderFor is how an input vector becomes a stored T: unchanged as
// []float32, scalar-quantized as []int8.
func encoderFor[T metric.VectorType]() func([]float32) T {
	var encode func([]float32) T

	// T is one of VectorType's types, so exactly one case matches.
	switch p := any(&encode).(type) {
	case *func([]float32) []float32:
		*p = func(v []float32) []float32 { return v }
	case *func([]float32) []int8:
		*p = core.QuantizeVector
	}

	return encode
}
