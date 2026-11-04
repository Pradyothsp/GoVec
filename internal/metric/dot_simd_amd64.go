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
// width: eight 8-lane accumulators, 64 floats per loop. An FMA waits on its
// own accumulator's previous result, so more accumulators keep more FMAs in
// flight; on NEON, four left the FMA units idle.
func dotFloat32AVX2(a, b []float32) float32 {
	b = b[:len(a)]
	var zero archsimd.Float32x8
	s0, s1, s2, s3, s4, s5, s6, s7 := zero, zero, zero, zero, zero, zero, zero, zero
	n := len(a) &^ 63
	for i := 0; i < n; i += 64 {
		x, y := a[i:i+64:i+64], b[i:i+64:i+64]
		s0 = archsimd.LoadFloat32x8(x[0:8]).MulAdd(archsimd.LoadFloat32x8(y[0:8]), s0)
		s1 = archsimd.LoadFloat32x8(x[8:16]).MulAdd(archsimd.LoadFloat32x8(y[8:16]), s1)
		s2 = archsimd.LoadFloat32x8(x[16:24]).MulAdd(archsimd.LoadFloat32x8(y[16:24]), s2)
		s3 = archsimd.LoadFloat32x8(x[24:32]).MulAdd(archsimd.LoadFloat32x8(y[24:32]), s3)
		s4 = archsimd.LoadFloat32x8(x[32:40]).MulAdd(archsimd.LoadFloat32x8(y[32:40]), s4)
		s5 = archsimd.LoadFloat32x8(x[40:48]).MulAdd(archsimd.LoadFloat32x8(y[40:48]), s5)
		s6 = archsimd.LoadFloat32x8(x[48:56]).MulAdd(archsimd.LoadFloat32x8(y[48:56]), s6)
		s7 = archsimd.LoadFloat32x8(x[56:64]).MulAdd(archsimd.LoadFloat32x8(y[56:64]), s7)
	}

	var lanes [8]float32
	s0.Add(s1).Add(s2.Add(s3)).Add(s4.Add(s5).Add(s6.Add(s7))).StoreArray(&lanes)
	sum := ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))

	return sum + dotFloat32Go(a[n:], b[n:])
}
