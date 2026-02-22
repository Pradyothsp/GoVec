package fixtures

import (
	"fmt"

	"github.com/Pradyothsp/govec/internal/core"
)

// Common test vectors
var (
	Vec2DSimple    = []float32{1.0, 2.0}
	Vec3dSimple    = []float32{1.0, 2.0, 3.0}
	Vec3dAlternate = []float32{4.0, 5.0, 6.0}
	Vec3dThird     = []float32{7.0, 8.0, 9.0}
	VecEmpty       = []float32{}
	Vec1536d       []float32 // OpenAI embedding size
	Vec4096d       []float32 // Large vector
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
	MetaEmpty   = map[string]any{}
	MetaComplex = map[string]any{
		"string": "value",
		"int":    123,
		"float":  3.14,
		"bool":   true,
		"array":  []string{"a", "b", "c"},
		"nested": map[string]any{"key": "value"},
	}
)

// Common sparse vectors for hybrid search
var (
	SparseEmpty = core.SparseVector{}
	SparseBM25  = core.SparseVector{
		Indices: []uint32{12, 452},
		Values:  []float32{0.9, 1.8},
	}
	SparseTfIdf = core.SparseVector{
		Indices: []uint32{5, 100, 200},
		Values:  []float32{0.5, 1.2, 0.8},
	}
	SparseLarge = core.SparseVector{
		Indices: []uint32{1, 10, 100, 1000, 5000},
		Values:  []float32{0.1, 0.2, 0.3, 0.4, 0.5},
	}
)

func init() {
	// Initialize large vectors
	Vec1536d = make([]float32, 1536)
	for i := 0; i < 1536; i++ {
		Vec1536d[i] = float32(i) * 0.001
	}

	Vec4096d = make([]float32, 4096)
	for i := 0; i < 4096; i++ {
		Vec4096d[i] = float32(i) * 0.0001
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

// GenerateNormalizedVectors generates N normalized (unit) vectors with specified dimensions
func GenerateNormalizedVectors(n, dims int) [][]float32 {
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dims)
		var sumSquares float32
		for j := 0; j < dims; j++ {
			vec[j] = float32(i*dims+j) * 0.01
			sumSquares += vec[j] * vec[j]
		}
		// Normalize
		if sumSquares > 0 {
			magnitude := float32(1.0) / float32(sumSquares)
			for j := 0; j < dims; j++ {
				vec[j] *= magnitude
			}
		}
		vectors[i] = vec
	}
	return vectors
}

// GenerateSparseVectors generates N sparse vectors for testing
func GenerateSparseVectors(n int) []core.SparseVector {
	sparse := make([]core.SparseVector, n)
	for i := 0; i < n; i++ {
		numIndices := 3 + i%5 // 3-7 indices per vector
		indices := make([]uint32, numIndices)
		values := make([]float32, numIndices)
		for j := 0; j < numIndices; j++ {
			indices[j] = uint32((i+1)*100 + j*10)
			values[j] = float32(i+j) * 0.1
		}
		sparse[i] = core.SparseVector{
			Indices: indices,
			Values:  values,
		}
	}
	return sparse
}

// GenerateMetadata generates N metadata maps with varying complexity
func GenerateMetadata(n int) []map[string]any {
	metadata := make([]map[string]any, n)
	for i := 0; i < n; i++ {
		metadata[i] = map[string]any{
			"id":    fmt.Sprintf("vec%d", i),
			"label": "test",
			"group": i % 3,
		}
	}
	return metadata
}
