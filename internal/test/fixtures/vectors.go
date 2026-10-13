package fixtures

// Common test vectors
var (
	Vec3dSimple    = []float32{1.0, 2.0, 3.0}
	Vec3dAlternate = []float32{4.0, 5.0, 6.0}
	Vec3dThird     = []float32{7.0, 8.0, 9.0}
	Vec1536d       []float32 // OpenAI embedding size
)

// Common metadata patterns
var (
	MetaSimple = map[string]any{
		"label": "test",
		"count": 42,
	}
	MetaNested = map[string]any{
		"category": "test",
		"nested": map[string]any{
			"level":  2,
			"active": true,
		},
		"tags": []string{"go", "vector", "db"},
	}
)

func init() {
	Vec1536d = make([]float32, 1536)
	for i := 0; i < 1536; i++ {
		Vec1536d[i] = float32(i) * 0.001
	}
}

// GenerateVectors generates N vectors programmatically with specified dimensions
func GenerateVectors(n, dims int) [][]float32 {
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dims)
		for j := 0; j < dims; j++ {
			vec[j] = float32(i*dims+j) * 0.01
		}
		vectors[i] = vec
	}
	return vectors
}
