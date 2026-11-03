package metric

// DotFloat32 is the dot product of a and b. b must be at least as long as a.
// It is the HNSW graph's cosine distance and its cached norms, so it is most
// of a search's time at 1536 dimensions.
//
// Built with GOEXPERIMENT=simd, it runs on SIMD registers (dot_simd_*.go);
// otherwise, or on a CPU without the instructions, on dotFloat32Go. Every
// version sums in float32, in a different order, so results agree to float32
// rounding, not bit for bit.
func DotFloat32(a, b []float32) float32 {
	return dotFloat32(a, b)
}

// dotFloat32Go keeps eight running totals, in float32. float64 totals convert
// every element first, which halves how many fit in a register: 375 ns a call
// at 1536 dimensions against 263 ns, and HNSW search on DBpedia 10k took
// 1.35 ms against 1.12 ms. The cost is precision: cosine distance moves by up
// to ~3e-7 from the exact value, against ~4e-15, with recall@10 unchanged.
func dotFloat32Go(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	b = b[:len(a)]
	n := len(a) &^ 7
	for i := 0; i < n; i += 8 {
		// Fixed-length subslices let the compiler drop the bounds checks.
		x, y := a[i:i+8:i+8], b[i:i+8:i+8]
		s0 += x[0] * y[0]
		s1 += x[1] * y[1]
		s2 += x[2] * y[2]
		s3 += x[3] * y[3]
		s4 += x[4] * y[4]
		s5 += x[5] * y[5]
		s6 += x[6] * y[6]
		s7 += x[7] * y[7]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}
