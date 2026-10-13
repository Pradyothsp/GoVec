// Package metric is the vector math behind search: the similarity and
// distance kernels, and the metrics built from them. It imports nothing else
// from GoVec, so the HNSW graph can run on a metric without reaching GoVec's
// domain types in core.
package metric

import (
	"fmt"
	"slices"
	"strings"
)

// Name names a distance metric, as config and snapshots spell it.
type Name string

const (
	// Cosine measures the angle between vectors (ignores magnitude).
	Cosine Name = "cosine"
	// Euclidean measures straight-line distance between vectors (magnitude-sensitive).
	Euclidean Name = "euclidean"
)

// Metric is everything an index needs to know about one distance metric, for
// vectors stored as T. Brute force ranks on Similarity. The HNSW graph walks
// on Distance, and its results are reported as Score(distance), so both
// engines give a match the same score and hybrid blending treats them alike.
//
// A metric used to be spelled out in three places -- a similarity registry
// in core, an HNSW distance switch in the engine factory, and an "euclidean"
// string check in HNSW search -- and nothing made them agree. Now each metric
// is one entry in metrics, and metric_test.go checks that Score(Distance)
// matches Similarity for every entry.
type Metric[T VectorType] struct {
	Name Name

	// Similarity scores a stored vector against a query, higher = closer.
	Similarity func(a, b T) (float32, error)

	// Distance orders the HNSW graph, lower = closer.
	Distance DistanceFunc[T]

	// Score converts a Distance result to Similarity's scale.
	Score func(distance float32) float32

	// Precompute and CachedDistance are an optional pair, nil together: when
	// set, the graph computes Precompute(v) once per stored vector and
	// compares with CachedDistance, passing both sides' cached values. Cosine
	// uses it to compute each vector's norm once instead of on every
	// comparison.
	Precompute     func(v T) float64
	CachedDistance func(aVec T, aCache float64, bVec T, bCache float64) float32

	// SquaredDistance and FromSquaredDistance are a second optional pair, nil
	// together: a cheaper distance that ranks the same, for comparisons inside
	// the graph, and the conversion back to Distance for the results it
	// returns. Euclidean uses it to skip a sqrt per comparison.
	SquaredDistance     DistanceFunc[T]
	FromSquaredDistance func(squared float32) float32
}

// metricDef is one metric for every vector type. Score works on a distance,
// which is a float32 whatever the vector type, so the two share it.
type metricDef struct {
	score          func(distance float32) float32
	scaleInvariant bool // see Name.ScaleInvariant
	float32        Metric[[]float32]
	int8           Metric[[]int8]
}

// metrics is every distance metric GoVec supports. Adding an entry here is
// all a new metric needs.
var metrics = map[Name]metricDef{
	Cosine: {
		score:          func(distance float32) float32 { return 1 - distance },
		scaleInvariant: true,
		float32: Metric[[]float32]{
			Similarity:     CosineSimilarity,
			Distance:       CosineDistanceFloat32,
			Precompute:     PrecomputeSquaredNormFloat32,
			CachedDistance: CachedCosineDistanceFloat32,
		},
		int8: Metric[[]int8]{
			Similarity:     CosineSimilarityInt8,
			Distance:       CosineDistanceInt8,
			Precompute:     PrecomputeSquaredNormInt8,
			CachedDistance: CachedCosineDistanceInt8,
		},
	},
	Euclidean: {
		// Euclidean distance is unbounded, so it maps into (0, 1] the way
		// EuclideanSimilarity does.
		score: func(distance float32) float32 { return 1 / (1 + distance) },
		float32: Metric[[]float32]{
			Similarity:          EuclideanSimilarity,
			Distance:            EuclideanDistanceFloat32,
			SquaredDistance:     SquaredEuclideanDistanceFloat32,
			FromSquaredDistance: FromSquaredEuclideanDistance,
		},
		int8: Metric[[]int8]{
			Similarity:          EuclideanSimilarityInt8,
			Distance:            EuclideanDistanceInt8,
			SquaredDistance:     SquaredEuclideanDistanceInt8,
			FromSquaredDistance: FromSquaredEuclideanDistance,
		},
	},
}

// ScaleInvariant reports whether metric n ignores a vector's length, so a
// vector can be unit-normalized without changing any score. Cosine is; scalar
// quantization relies on it to bring any vector into its [-1, 1] range.
// Euclidean isn't: normalizing would change its distances. False for an
// unknown name.
func (n Name) ScaleInvariant() bool {
	return metrics[n].scaleInvariant
}

// Resolve returns the metric called name, for vectors stored as T.
func Resolve[T VectorType](name Name) (Metric[T], error) {
	var m Metric[T]

	def, ok := metrics[name]
	if !ok {
		return m, fmt.Errorf("unknown distance metric '%s', must be one of: %s", name, strings.Join(Names(), ", "))
	}

	// T is one of VectorType's types, so exactly one case matches.
	switch p := any(&m).(type) {
	case *Metric[[]float32]:
		*p = def.float32
	case *Metric[[]int8]:
		*p = def.int8
	}

	m.Name = name
	m.Score = def.score
	return m, nil
}

// IsKnown reports whether name is a supported metric.
func IsKnown(name Name) bool {
	_, ok := metrics[name]
	return ok
}

// Names lists every supported metric, sorted, for error messages.
func Names() []string {
	names := make([]string, 0, len(metrics))
	for name := range metrics {
		names = append(names, string(name))
	}

	slices.Sort(names)
	return names
}
