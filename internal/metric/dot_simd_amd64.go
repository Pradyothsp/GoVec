//go:build goexperiment.simd

package metric

import "simd/archsimd"

// Not every amd64 CPU has AVX2 and FMA, and running the instructions on one
// without them crashes the process, so this is checked once and the Go
// kernel used otherwise.
var useAVX2FMA = archsimd.X86.AVX2() && archsimd.X86.FMA()

func dotFloat32(a, b []float32) float32 {
	if !useAVX2FMA {
		return dotFloat32Go(a, b)
	}

	return dotFloat32AVX2(a, b)
}

// dotFloat32AVX2 is the arm64 NEON kernel (dot_simd_arm64.go) at twice the
// width: four 8-lane accumulators over the same 32-element blocks. Unlike
// NEON's, it has not been timed: no amd64 machine was at hand, so CI's tests
// show it correct, not faster.
func dotFloat32AVX2(a, b []float32) float32 {
	b = b[:len(a)]
	var zero archsimd.Float32x8
	s0, s1, s2, s3 := zero, zero, zero, zero
	n := len(a) &^ 31
	for i := 0; i < n; i += 32 {
		x, y := a[i:i+32:i+32], b[i:i+32:i+32]
		s0 = archsimd.LoadFloat32x8(x[0:8]).MulAdd(archsimd.LoadFloat32x8(y[0:8]), s0)
		s1 = archsimd.LoadFloat32x8(x[8:16]).MulAdd(archsimd.LoadFloat32x8(y[8:16]), s1)
		s2 = archsimd.LoadFloat32x8(x[16:24]).MulAdd(archsimd.LoadFloat32x8(y[16:24]), s2)
		s3 = archsimd.LoadFloat32x8(x[24:32]).MulAdd(archsimd.LoadFloat32x8(y[24:32]), s3)
	}

	var lanes [8]float32
	s0.Add(s1).Add(s2.Add(s3)).StoreArray(&lanes)
	sum := ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))

	return sum + dotFloat32Go(a[n:], b[n:])
}
