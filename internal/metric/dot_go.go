//go:build !goexperiment.simd || !(arm64 || amd64)

package metric

func dotFloat32(a, b []float32) float32 {
	return dotFloat32Go(a, b)
}
