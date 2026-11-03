//go:build goexperiment.simd

package metric

import "simd/archsimd"

// dotFloat32 on arm64 uses NEON, which every arm64 CPU has: 113 ns a call at
// 1536 dimensions against dotFloat32Go's 263 ns, and HNSW search on DBpedia
// 10k took 0.87 ms against 1.13 ms (Apple M2), recall@10 unchanged.
//
// Eight accumulators, not four: a fused multiply-add waits on its own
// accumulator's previous result, so four left the FMA units idle (216 ns).
func dotFloat32(a, b []float32) float32 {
	b = b[:len(a)]
	var zero archsimd.Float32x4
	s0, s1, s2, s3, s4, s5, s6, s7 := zero, zero, zero, zero, zero, zero, zero, zero
	n := len(a) &^ 31
	for i := 0; i < n; i += 32 {
		x, y := a[i:i+32:i+32], b[i:i+32:i+32]
		s0 = archsimd.LoadFloat32x4(x[0:4]).MulAdd(archsimd.LoadFloat32x4(y[0:4]), s0)
		s1 = archsimd.LoadFloat32x4(x[4:8]).MulAdd(archsimd.LoadFloat32x4(y[4:8]), s1)
		s2 = archsimd.LoadFloat32x4(x[8:12]).MulAdd(archsimd.LoadFloat32x4(y[8:12]), s2)
		s3 = archsimd.LoadFloat32x4(x[12:16]).MulAdd(archsimd.LoadFloat32x4(y[12:16]), s3)
		s4 = archsimd.LoadFloat32x4(x[16:20]).MulAdd(archsimd.LoadFloat32x4(y[16:20]), s4)
		s5 = archsimd.LoadFloat32x4(x[20:24]).MulAdd(archsimd.LoadFloat32x4(y[20:24]), s5)
		s6 = archsimd.LoadFloat32x4(x[24:28]).MulAdd(archsimd.LoadFloat32x4(y[24:28]), s6)
		s7 = archsimd.LoadFloat32x4(x[28:32]).MulAdd(archsimd.LoadFloat32x4(y[28:32]), s7)
	}

	var lanes [4]float32
	s0.Add(s1).Add(s2.Add(s3)).Add(s4.Add(s5).Add(s6.Add(s7))).StoreArray(&lanes)
	sum := (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])

	return sum + dotFloat32Go(a[n:], b[n:])
}
