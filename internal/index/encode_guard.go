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

	switch distanceMetric {
	case string(config.DistanceMetricCosine):
		return core.NormalizeVector(vec), nil
	case string(config.DistanceMetricEuclidean):
		if err := core.ValidateQuantizationRange(vec); err != nil {
			return nil, err
		}
	}

	return vec, nil
}
