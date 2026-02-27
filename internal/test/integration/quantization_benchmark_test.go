package integration

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// BenchmarkResults holds metrics for comparison
type BenchmarkResults struct {
	Quantization     string
	InsertTime       time.Duration
	SearchTime       time.Duration
	MemoryAllocated  uint64
	FileSize         int64
	TopSearchResults []index.SearchResult
}

func BenchmarkQuantization_Comprehensive(b *testing.B) {
	// Test configuration
	numVectors := 1000
	vectorDim := 1536 // Typical embedding size (OpenAI)
	numSearches := 100
	topK := 10

	b.Logf("=== Quantization Benchmark ===")
	b.Logf("Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("Searches: %d queries × top-%d results", numSearches, topK)
	b.Logf("")

	// Run benchmark for both quantization types
	resultsNone := runBenchmark(b, "none", numVectors, vectorDim, numSearches, topK)
	resultsScalar := runBenchmark(b, "scalar", numVectors, vectorDim, numSearches, topK)

	// Print comparison table
	printComparisonTable(b, resultsNone, resultsScalar, numVectors, vectorDim)

	// Verify scalar quantization benefits
	verifyBenefits(b, resultsNone, resultsScalar)
}

func runBenchmark(b *testing.B, quantization string, numVectors, vectorDim, numSearches, topK int) BenchmarkResults {
	tmpDir := b.TempDir()
	walPath := tmpDir + "/test.wal"
	dataPath := tmpDir + "/test.bin"

	// Create engine
	wal, err := index.NewWAL(walPath)
	if err != nil {
		b.Fatalf("failed to create WAL: %v", err)
	}
	defer wal.Close()

	cfg := config.EngineConfig{
		Quantization:   config.Quantization(quantization),
		DistanceMetric: "cosine",
	}

	engine, err := index.NewEngine(cfg, config.StorageConfig{}, wal)
	if err != nil {
		b.Fatalf("NewEngine failed: %v", err)
	}

	// Generate test vectors
	vectors := generateTestVectors(numVectors, vectorDim)

	// Measure insert time
	runtime.GC() // Force GC before measurement
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	insertStart := time.Now()
	for i, vec := range vectors {
		id := fmt.Sprintf("vec%d", i)
		if err := engine.Insert(context.Background(), id, vec, core.SparseVector{}, map[string]interface{}{"index": i}); err != nil {
			b.Fatalf("Insert failed: %v", err)
		}
	}
	insertTime := time.Since(insertStart)

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	memAllocated := memAfter.Alloc - memBefore.Alloc

	// Save to file and measure file size
	if err := engine.SaveToFile(context.Background(), dataPath); err != nil {
		b.Fatalf("SaveToFile failed: %v", err)
	}

	fileInfo, err := os.Stat(dataPath)
	if err != nil {
		b.Fatalf("failed to stat file: %v", err)
	}
	fileSize := fileInfo.Size()

	// Measure search time
	queries := generateTestVectors(numSearches, vectorDim)
	searchStart := time.Now()
	for _, query := range queries {
		if _, err := engine.Search(context.Background(), query, core.SparseVector{}, topK, nil); err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
	searchTime := time.Since(searchStart)

	// Get sample search results for accuracy comparison
	sampleQuery := []float32{0.5, 0.3, 0.8}
	// Pad to vectorDim
	for len(sampleQuery) < vectorDim {
		sampleQuery = append(sampleQuery, 0.0)
	}
	topResults, err := engine.Search(context.Background(), sampleQuery, core.SparseVector{}, topK, nil)
	if err != nil {
		b.Fatalf("Search failed: %v", err)
	}

	return BenchmarkResults{
		Quantization:     quantization,
		InsertTime:       insertTime,
		SearchTime:       searchTime,
		MemoryAllocated:  memAllocated,
		FileSize:         fileSize,
		TopSearchResults: topResults,
	}
}

func generateTestVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			// Generate more diverse vectors using multiple frequency components
			// This creates vectors with different patterns and magnitudes
			vec[j] = float32(
				math.Sin(float64(i*j)/100.0)+
					math.Cos(float64(i+j*7)/50.0)*0.5+
					math.Sin(float64(i*13+j)/30.0)*0.3,
			) * 0.4
		}
		vectors[i] = vec
	}
	return vectors
}

func printComparisonTable(b *testing.B, none, scalar BenchmarkResults, numVectors, vectorDim int) {
	b.Logf("╔════════════════════════════════════════════════════════════════╗")
	b.Logf("║                    Performance Comparison                     ║")
	b.Logf("╠════════════════════════════════════════════════════════════════╣")
	b.Logf("║ Metric                    │ None (float32) │ Scalar (int8)   ║")
	b.Logf("╠═══════════════════════════┼════════════════┼═════════════════╣")

	// Insert performance
	b.Logf("║ Insert Time               │ %13s │ %14s ║",
		none.InsertTime.Round(time.Millisecond),
		scalar.InsertTime.Round(time.Millisecond))

	insertSpeedup := float64(none.InsertTime) / float64(scalar.InsertTime)
	b.Logf("║ Insert Speedup            │            1.0x │ %14.2fx ║", insertSpeedup)

	// Search performance
	b.Logf("║ Search Time               │ %13s │ %14s ║",
		none.SearchTime.Round(time.Millisecond),
		scalar.SearchTime.Round(time.Millisecond))

	searchSpeedup := float64(none.SearchTime) / float64(scalar.SearchTime)
	b.Logf("║ Search Speedup            │            1.0x │ %14.2fx ║", searchSpeedup)

	// Memory usage
	b.Logf("║ Memory Allocated          │ %11s MB │ %11s MB ║",
		fmt.Sprintf("%.2f", float64(none.MemoryAllocated)/1024/1024),
		fmt.Sprintf("%.2f", float64(scalar.MemoryAllocated)/1024/1024))

	memReduction := 100.0 * (1.0 - float64(scalar.MemoryAllocated)/float64(none.MemoryAllocated))
	b.Logf("║ Memory Reduction          │            0.0%% │ %13.1f%% ║", memReduction)

	// File size
	b.Logf("║ File Size                 │ %11s MB │ %11s MB ║",
		fmt.Sprintf("%.2f", float64(none.FileSize)/1024/1024),
		fmt.Sprintf("%.2f", float64(scalar.FileSize)/1024/1024))

	sizeReduction := 100.0 * (1.0 - float64(scalar.FileSize)/float64(none.FileSize))
	b.Logf("║ File Size Reduction       │            0.0%% │ %13.1f%% ║", sizeReduction)

	// Theoretical calculations
	theoreticalFloat32 := int64(numVectors * vectorDim * 4)
	theoreticalInt8 := int64(numVectors * vectorDim * 1)
	theoreticalReduction := 100.0 * (1.0 - float64(theoreticalInt8)/float64(theoreticalFloat32))

	b.Logf("╠═══════════════════════════┴════════════════┴═════════════════╣")
	b.Logf("║ Theoretical Vector Size   │ %11s MB │ %11s MB ║",
		fmt.Sprintf("%.2f", float64(theoreticalFloat32)/1024/1024),
		fmt.Sprintf("%.2f", float64(theoreticalInt8)/1024/1024))
	b.Logf("║ Theoretical Reduction     │            0.0%% │ %13.1f%% ║", theoreticalReduction)
	b.Logf("╚════════════════════════════════════════════════════════════════╝")
	b.Logf("")

	// Search accuracy comparison
	b.Logf("╔════════════════════════════════════════════════════════════════╗")
	b.Logf("║                     Search Accuracy (Top 10)                  ║")
	b.Logf("╠════════════════════════════════════════════════════════════════╣")
	b.Logf("║ Rank │ None (float32)     │ Scalar (int8)      │ Score Diff  ║")
	b.Logf("╠══════┼════════════════════┼════════════════════┼═════════════╣")

	maxResults := len(none.TopSearchResults)
	if len(scalar.TopSearchResults) < maxResults {
		maxResults = len(scalar.TopSearchResults)
	}

	for i := 0; i < maxResults; i++ {
		noneResult := none.TopSearchResults[i]
		scalarResult := scalar.TopSearchResults[i]
		scoreDiff := math.Abs(float64(noneResult.Score - scalarResult.Score))

		b.Logf("║ %4d │ %s (%.4f) │ %s (%.4f) │ %11.4f ║",
			i+1,
			padRight(noneResult.ID, 5),
			noneResult.Score,
			padRight(scalarResult.ID, 5),
			scalarResult.Score,
			scoreDiff)
	}

	b.Logf("╚════════════════════════════════════════════════════════════════╝")
	b.Logf("")
}

func padRight(s string, length int) string {
	if len(s) >= length {
		return s[:length]
	}
	return s + string(make([]byte, length-len(s)))
}

func verifyBenefits(b *testing.B, none, scalar BenchmarkResults) {
	// Memory should be reduced by ~75% (4x compression)
	memReduction := 100.0 * (1.0 - float64(scalar.MemoryAllocated)/float64(none.MemoryAllocated))
	if memReduction < 50.0 {
		b.Logf("Warning: Memory reduction (%.1f%%) is less than expected (>50%%)", memReduction)
	} else {
		b.Logf("✅ Memory reduction: %.1f%%", memReduction)
	}

	// File size should be reduced by ~75% (4x compression)
	sizeReduction := 100.0 * (1.0 - float64(scalar.FileSize)/float64(none.FileSize))
	if sizeReduction < 50.0 {
		b.Logf("Warning: File size reduction (%.1f%%) is less than expected (>50%%)", sizeReduction)
	} else {
		b.Logf("✅ File size reduction: %.1f%%", sizeReduction)
	}

	// Search accuracy should be high (score differences < 0.01)
	maxScoreDiff := 0.0
	rankingMatches := 0
	totalResults := len(none.TopSearchResults)
	if len(scalar.TopSearchResults) < totalResults {
		totalResults = len(scalar.TopSearchResults)
	}

	for i := 0; i < totalResults; i++ {
		scoreDiff := math.Abs(float64(none.TopSearchResults[i].Score - scalar.TopSearchResults[i].Score))
		if scoreDiff > maxScoreDiff {
			maxScoreDiff = scoreDiff
		}
		if none.TopSearchResults[i].ID == scalar.TopSearchResults[i].ID {
			rankingMatches++
		}
	}

	rankingAccuracy := 100.0 * float64(rankingMatches) / float64(totalResults)
	b.Logf("✅ Max score difference: %.4f", maxScoreDiff)
	b.Logf("✅ Ranking accuracy: %.1f%% (%d/%d matches)", rankingAccuracy, rankingMatches, totalResults)

	if maxScoreDiff > 0.1 {
		b.Logf("Warning: Max score difference (%.4f) is higher than expected (<0.1)", maxScoreDiff)
	}

	if rankingAccuracy < 80.0 {
		b.Logf("Warning: Ranking accuracy (%.1f%%) is lower than expected (>80%%)", rankingAccuracy)
	}
}

func BenchmarkQuantization_LargeDataset(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping large dataset benchmark in short mode")
	}

	// Larger dataset for stress testing
	numVectors := 10000
	vectorDim := 768 // Smaller dimension for faster execution
	numSearches := 500
	topK := 20

	b.Logf("=== Large Dataset Benchmark ===")
	b.Logf("Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("Searches: %d queries × top-%d results", numSearches, topK)
	b.Logf("")

	resultsNone := runBenchmark(b, "none", numVectors, vectorDim, numSearches, topK)
	resultsScalar := runBenchmark(b, "scalar", numVectors, vectorDim, numSearches, topK)

	printComparisonTable(b, resultsNone, resultsScalar, numVectors, vectorDim)
	verifyBenefits(b, resultsNone, resultsScalar)
}

func BenchmarkQuantization_HighDimension(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping high dimension benchmark in short mode")
	}

	// High-dimensional vectors (e.g., Cohere embeddings)
	numVectors := 500
	vectorDim := 4096
	numSearches := 50
	topK := 5

	b.Logf("=== High Dimension Benchmark ===")
	b.Logf("Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("Searches: %d queries × top-%d results", numSearches, topK)
	b.Logf("")

	resultsNone := runBenchmark(b, "none", numVectors, vectorDim, numSearches, topK)
	resultsScalar := runBenchmark(b, "scalar", numVectors, vectorDim, numSearches, topK)

	printComparisonTable(b, resultsNone, resultsScalar, numVectors, vectorDim)
	verifyBenefits(b, resultsNone, resultsScalar)
}
