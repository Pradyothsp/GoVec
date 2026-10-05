package index

import (
	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
)

// prepareForEncode applies the metric-dependent fix-up QuantizeVector's
// [-1, 1] precondition needs, when scalar quantization is active. A vector
// outside that range would otherwise be silently clamped to a different
// vector -- SIFT-scale features (unnormalized, values well outside [-1, 1])
// clamp to near-uniform int8 vectors this way, destroying recall. Under
// cosine distance the fix is always safe (unit-normalize); under Euclidean,
// normalizing would corrupt the distance itself, so out-of-range vectors are
// rejected instead. No-op when quantization is "none".
func prepareForEncode(vec []float32, quantization, distanceMetric string) ([]float32, error) {
	if quantization != string(config.QuantizationScalar) {
		return vec, nil
	}

	if err := checkEncodable(vec, quantization, distanceMetric); err != nil {
		return nil, err
	}
	if distanceMetric == string(config.DistanceMetricCosine) {
		return core.NormalizeVector(vec), nil
	}
	return vec, nil
}

// checkEncodable reports whether prepareForEncode would reject vec, without
// normalizing it: only scalar quantization under Euclidean can reject.
func checkEncodable(vec []float32, quantization, distanceMetric string) error {
	if quantization == string(config.QuantizationScalar) && distanceMetric == string(config.DistanceMetricEuclidean) {
		return core.ValidateQuantizationRange(vec)
	}
	return nil
}

// storedCopy returns a vector the index can keep. A float32 vector passes
// through encoding untouched, so without a copy the index would hold the
// caller's slice: memory the caller still owns, with whatever spare capacity
// JSON or protobuf decoding left in it -- 721 MB held for 614 MB of vectors at
// 100k x 1536. int8 encoding already allocates an exact slice.
func storedCopy[T any](encoded T) T {
	v, ok := any(encoded).([]float32)
	if !ok {
		return encoded
	}

	// make + copy, not slices.Clone: Clone rounds capacity up to the
	// allocator's size class.
	owned := make([]float32, len(v))
	copy(owned, v)
	if out, ok := any(owned).(T); ok {
		return out
	}
	return encoded
}
