package integration

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// =============================================================================
// Core helpers (new — do not modify runBenchmark in quantization_benchmark_test.go)
// =============================================================================

// runBenchmarkWithIndex is like runBenchmark but also sets cfg.IndexType, enabling
// brute vs HNSW comparisons. The Quantization field in the returned BenchmarkResults
// is repurposed as a composite label ("indexType/quantization").
func runBenchmarkWithIndex(b *testing.B, quantization, indexType string,
	numVectors, vectorDim, numSearches, topK int) BenchmarkResults {
	b.Helper()
	tmpDir := b.TempDir()
	walPath := tmpDir + "/test.wal"
	dataPath := tmpDir + "/test.bin"

	wal, err := index.NewWAL(walPath)
	if err != nil {
		b.Fatalf("runBenchmarkWithIndex: failed to create WAL: %v", err)
	}
	defer wal.Close() //nolint:errcheck // benchmark cleanup

	cfg := config.EngineConfig{
		Quantization:   config.Quantization(quantization),
		DistanceMetric: "cosine",
		IndexType:      config.IndexType(indexType),
	}
	engine, err := index.NewEngine(cfg, wal)
	if err != nil {
		b.Fatalf("runBenchmarkWithIndex: NewEngine failed: %v", err)
	}

	vectors := generateTestVectors(numVectors, vectorDim)

	runtime.GC()
	var memBefore, memAfter runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	insertStart := time.Now()
	for i, vec := range vectors {
		if err := engine.Insert(fmt.Sprintf("vec%d", i), vec, core.SparseVector{}, map[string]interface{}{"index": i}); err != nil {
			b.Fatalf("Insert failed: %v", err)
		}
	}
	insertTime := time.Since(insertStart)

	runtime.ReadMemStats(&memAfter)
	memAllocated := memAfter.Alloc - memBefore.Alloc

	if err := engine.SaveToFile(dataPath); err != nil {
		b.Fatalf("SaveToFile failed: %v", err)
	}
	fileInfo, err := os.Stat(dataPath)
	if err != nil {
		b.Fatalf("failed to stat snapshot: %v", err)
	}
	fileSize := fileInfo.Size()

	queries := generateTestVectors(numSearches, vectorDim)
	searchStart := time.Now()
	for _, q := range queries {
		if _, err := engine.Search(q, core.SparseVector{}, topK, nil); err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
	searchTime := time.Since(searchStart)

	// Collect a fixed sample query for cross-run comparison
	sampleQuery := []float32{0.5, 0.3, 0.8}
	for len(sampleQuery) < vectorDim {
		sampleQuery = append(sampleQuery, 0.0)
	}
	topResults, err := engine.Search(sampleQuery, core.SparseVector{}, topK, nil)
	if err != nil {
		b.Fatalf("sample Search failed: %v", err)
	}

	return BenchmarkResults{
		Quantization:     fmt.Sprintf("%s/%s", indexType, quantization),
		InsertTime:       insertTime,
		SearchTime:       searchTime,
		MemoryAllocated:  memAllocated,
		FileSize:         fileSize,
		TopSearchResults: topResults,
	}
}

// newHNSWEngineForBench creates an engine for use inside benchmark functions.
// Caller is responsible for closing the returned WAL.
func newHNSWEngineForBench(b *testing.B, quantization, indexType string) (index.Engine, *index.WAL) {
	b.Helper()
	tmpDir := b.TempDir()
	walPath := tmpDir + "/bench.wal"

	wal, err := index.NewWAL(walPath)
	if err != nil {
		b.Fatalf("newHNSWEngineForBench: failed to create WAL: %v", err)
	}
	cfg := config.EngineConfig{
		Quantization:   config.Quantization(quantization),
		DistanceMetric: "cosine",
		IndexType:      config.IndexType(indexType),
	}
	engine, err := index.NewEngine(cfg, wal)
	if err != nil {
		b.Fatalf("newHNSWEngineForBench: NewEngine failed: %v", err)
	}
	return engine, wal
}

// measureRecall wraps the package-level calculateRecall with a k-truncation step.
func measureRecall(groundTruth, approximate []index.SearchResult, k int) float64 {
	gt := groundTruth
	if len(gt) > k {
		gt = gt[:k]
	}
	approx := approximate
	if len(approx) > k {
		approx = approx[:k]
	}
	return calculateRecall(gt, approx)
}

// printFourWayTable logs a 4-column comparison across brute/hnsw × none/scalar.
func printFourWayTable(b *testing.B, bruteNone, bruteScalar, hnswNone, hnswScalar BenchmarkResults, numVectors, vectorDim int) {
	b.Helper()
	b.Logf("╔═══════════════════════════════════════════════════════════════════════╗")
	b.Logf("║          4-Way Comparison: Brute vs HNSW × None vs Scalar            ║")
	b.Logf("╠═══════════════════════════════════════════════════════════════════════╣")
	b.Logf("║  Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("╠══════════════════╤═════════════╤═════════════╤═════════════╤══════════╣")
	b.Logf("║ Metric           │ brute/none  │brute/scalar │  hnsw/none  │hnsw/scala║")
	b.Logf("╠══════════════════╪═════════════╪═════════════╪═════════════╪══════════╣")
	b.Logf("║ Insert Time      │ %11s │ %11s │ %11s │ %8s ║",
		bruteNone.InsertTime.Round(time.Millisecond),
		bruteScalar.InsertTime.Round(time.Millisecond),
		hnswNone.InsertTime.Round(time.Millisecond),
		hnswScalar.InsertTime.Round(time.Millisecond))
	b.Logf("║ Search Time      │ %11s │ %11s │ %11s │ %8s ║",
		bruteNone.SearchTime.Round(time.Millisecond),
		bruteScalar.SearchTime.Round(time.Millisecond),
		hnswNone.SearchTime.Round(time.Millisecond),
		hnswScalar.SearchTime.Round(time.Millisecond))
	b.Logf("║ File Size (KB)   │ %11d │ %11d │ %11d │ %8d ║",
		bruteNone.FileSize/1024,
		bruteScalar.FileSize/1024,
		hnswNone.FileSize/1024,
		hnswScalar.FileSize/1024)
	b.Logf("╠══════════════════╪═════════════╪═════════════╪═════════════╪══════════╣")

	// Search speedup rows (brute/none as 1.0x baseline)
	baseSearch := float64(bruteNone.SearchTime)
	b.Logf("║ Search Speedup   │       1.00x │ %10.2fx │ %10.2fx │ %7.2fx ║",
		baseSearch/float64(bruteScalar.SearchTime),
		baseSearch/float64(hnswNone.SearchTime),
		baseSearch/float64(hnswScalar.SearchTime))
	b.Logf("╚══════════════════╧═════════════╧═════════════╧═════════════╧══════════╝")
	b.Logf("")
}

// verifyHNSWBenefits logs HNSW speedup relative to brute force, warning when the
// dataset is too small for HNSW to outperform the linear scan.
func verifyHNSWBenefits(b *testing.B, bruteNone, hnswNone, bruteScalar, hnswScalar BenchmarkResults) {
	b.Helper()

	float32Speedup := float64(bruteNone.SearchTime) / float64(hnswNone.SearchTime)
	if float32Speedup >= 1.0 {
		b.Logf("✅ HNSW float32 search speedup vs brute: %.2fx", float32Speedup)
	} else {
		b.Logf("⚠️  HNSW float32 not faster than brute (dataset may be too small): %.2fx", float32Speedup)
	}

	scalarSpeedup := float64(bruteScalar.SearchTime) / float64(hnswScalar.SearchTime)
	if scalarSpeedup >= 1.0 {
		b.Logf("✅ HNSW scalar search speedup vs brute: %.2fx", scalarSpeedup)
	} else {
		b.Logf("⚠️  HNSW scalar not faster than brute: %.2fx", scalarSpeedup)
	}

	b.Logf("Search times — brute/none: %s  hnsw/none: %s  brute/scalar: %s  hnsw/scalar: %s",
		bruteNone.SearchTime.Round(time.Millisecond),
		hnswNone.SearchTime.Round(time.Millisecond),
		bruteScalar.SearchTime.Round(time.Millisecond),
		hnswScalar.SearchTime.Round(time.Millisecond))
}

// =============================================================================
// Benchmark functions
// =============================================================================

// BenchmarkHNSW_vs_Brute_Comprehensive runs the primary 4-way comparison on a
// 1000 × 256 dataset: brute/scalar, hnsw/none, hnsw/scalar all vs brute/none.
func BenchmarkHNSW_vs_Brute_Comprehensive(b *testing.B) {
	numVectors := 1000
	vectorDim := 256
	numSearches := 100
	topK := 10

	b.Logf("=== HNSW vs Brute-Force 4-Way Comparison ===")
	b.Logf("Dataset: %d vectors × %d dimensions, %d queries × top-%d",
		numVectors, vectorDim, numSearches, topK)
	b.Logf("")

	bruteNone := runBenchmarkWithIndex(b, "none", "brute", numVectors, vectorDim, numSearches, topK)
	bruteScalar := runBenchmarkWithIndex(b, "scalar", "brute", numVectors, vectorDim, numSearches, topK)
	hnswNone := runBenchmarkWithIndex(b, "none", "hnsw", numVectors, vectorDim, numSearches, topK)
	hnswScalar := runBenchmarkWithIndex(b, "scalar", "hnsw", numVectors, vectorDim, numSearches, topK)

	printFourWayTable(b, bruteNone, bruteScalar, hnswNone, hnswScalar, numVectors, vectorDim)
	verifyHNSWBenefits(b, bruteNone, hnswNone, bruteScalar, hnswScalar)
}

// BenchmarkHNSW_SearchLatency_ScaleUp demonstrates O(log n) HNSW scaling vs O(n)
// brute force across n = 100, 500, 1000, 5000 at 128 dimensions.
func BenchmarkHNSW_SearchLatency_ScaleUp(b *testing.B) {
	vectorDim := 128
	numSearches := 50
	topK := 10

	b.Logf("=== Search Latency Scaling: O(log n) vs O(n) ===")
	b.Logf("%-8s │ %-14s │ %-14s │ %s", "n", "brute", "hnsw", "speedup")
	b.Logf("%-8s─┼─%-14s─┼─%-14s─┼─%s", "--------", "--------------", "--------------", "-------")

	for _, n := range []int{100, 500, 1000, 5000} {
		brute := runBenchmarkWithIndex(b, "none", "brute", n, vectorDim, numSearches, topK)
		hnsw := runBenchmarkWithIndex(b, "none", "hnsw", n, vectorDim, numSearches, topK)

		speedup := float64(brute.SearchTime) / float64(hnsw.SearchTime)
		b.Logf("%-8d │ %-14s │ %-14s │ %.2fx",
			n,
			brute.SearchTime.Round(time.Millisecond),
			hnsw.SearchTime.Round(time.Millisecond),
			speedup)
	}
}

// BenchmarkHNSW_Insert_BuildTime compares single-insert throughput for brute vs
// HNSW using the standard b.N loop pattern.
func BenchmarkHNSW_Insert_BuildTime(b *testing.B) {
	vectorDim := 128

	b.Run("brute", func(b *testing.B) {
		engine, wal := newHNSWEngineForBench(b, "none", "brute")
		defer wal.Close() //nolint:errcheck // benchmark cleanup
		vectors := generateTestVectors(b.N, vectorDim)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := engine.Insert(fmt.Sprintf("v%d", i), vectors[i], core.SparseVector{}, nil); err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}
	})

	b.Run("hnsw", func(b *testing.B) {
		engine, wal := newHNSWEngineForBench(b, "none", "hnsw")
		defer wal.Close() //nolint:errcheck // benchmark cleanup
		vectors := generateTestVectors(b.N, vectorDim)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := engine.Insert(fmt.Sprintf("v%d", i), vectors[i], core.SparseVector{}, nil); err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}
	})
}

// BenchmarkHNSW_Search_HighDimension tests both engines on 500 vectors of 1536
// dimensions (real embedding size). Skipped in -short mode.
func BenchmarkHNSW_Search_HighDimension(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping high-dimension benchmark in short mode")
	}

	numVectors := 500
	vectorDim := 1536
	numSearches := 50
	topK := 10

	b.Logf("=== High-Dimension Search (1536 dims) ===")
	b.Logf("Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("")

	brute := runBenchmarkWithIndex(b, "none", "brute", numVectors, vectorDim, numSearches, topK)
	hnsw := runBenchmarkWithIndex(b, "none", "hnsw", numVectors, vectorDim, numSearches, topK)

	speedup := float64(brute.SearchTime) / float64(hnsw.SearchTime)
	b.Logf("brute/none=%s  hnsw/none=%s  speedup=%.2fx",
		brute.SearchTime.Round(time.Millisecond),
		hnsw.SearchTime.Round(time.Millisecond),
		speedup)
}

// BenchmarkHNSW_Recall_vs_DatasetSize measures recall@10 at n = 200, 500, 1000,
// using brute-force as ground truth at each scale.
func BenchmarkHNSW_Recall_vs_DatasetSize(b *testing.B) {
	vectorDim := 128
	topK := 10
	numQueries := 20

	b.Logf("=== HNSW recall@%d vs Dataset Size ===", topK)
	b.Logf("%-8s │ %s", "n", fmt.Sprintf("recall@%d", topK))
	b.Logf("%-8s─┼─%s", "--------", "---------")

	for _, n := range []int{200, 500, 1000} {
		bruteEngine, bruteWal := newHNSWEngineForBench(b, "none", "brute")
		hnswEngine, hnswWal := newHNSWEngineForBench(b, "none", "hnsw")

		vectors := generateTestVectors(n, vectorDim)
		for i, v := range vectors {
			id := fmt.Sprintf("v%d", i)
			if err := bruteEngine.Insert(id, v, core.SparseVector{}, nil); err != nil {
				b.Fatalf("brute Insert failed: %v", err)
			}
			if err := hnswEngine.Insert(id, v, core.SparseVector{}, nil); err != nil {
				b.Fatalf("hnsw Insert failed: %v", err)
			}
		}

		queries := generateTestVectors(numQueries, vectorDim)
		totalRecall := 0.0
		for _, q := range queries {
			gt, err := bruteEngine.Search(q, core.SparseVector{}, topK, nil)
			if err != nil {
				b.Fatalf("brute Search failed: %v", err)
			}
			approx, err := hnswEngine.Search(q, core.SparseVector{}, topK, nil)
			if err != nil {
				b.Fatalf("hnsw Search failed: %v", err)
			}
			totalRecall += measureRecall(gt, approx, topK)
		}

		avgRecall := totalRecall / float64(numQueries)
		b.Logf("%-8d │ %.3f", n, avgRecall)

		_ = bruteWal.Close() //nolint:errcheck // benchmark cleanup
		_ = hnswWal.Close()  //nolint:errcheck // benchmark cleanup
	}
}
