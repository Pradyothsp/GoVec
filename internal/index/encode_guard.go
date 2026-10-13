package index

import (
	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/metric"
)

// prepareForEncode applies the metric-dependent fix-up QuantizeVector's
// [-1, 1] precondition needs, when scalar quantization is active. A vector
// outside that range would otherwise be silently clamped to a different
// vector -- SIFT-scale features (unnormalized, values well outside [-1, 1])
// clamp to near-uniform int8 vectors this way, destroying recall. Under a
// scale-invariant metric such as cosine the fix is always safe
// (unit-normalize); under one that isn't, such as Euclidean, normalizing would
// corrupt the distance itself, so out-of-range vectors are rejected instead.
// No-op without scalar quantization.
func prepareForEncode(vec []float32, quantization config.Quantization, distanceMetric metric.Name) ([]float32, error) {
	if quantization != config.QuantizationScalar {
		return vec, nil
	}

	if err := checkEncodable(vec, quantization, distanceMetric); err != nil {
		return nil, err
	}
	if distanceMetric.ScaleInvariant() {
		return core.NormalizeVector(vec), nil
	}
	return vec, nil
}

// checkEncodable reports whether prepareForEncode would reject vec, without
// normalizing it: only scalar quantization under a metric that isn't
// scale-invariant can reject.
func checkEncodable(vec []float32, quantization config.Quantization, distanceMetric metric.Name) error {
	if quantization == config.QuantizationScalar && !distanceMetric.ScaleInvariant() {
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
