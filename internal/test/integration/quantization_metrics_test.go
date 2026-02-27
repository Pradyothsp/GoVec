package integration

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// AdvancedMetrics holds detailed performance and accuracy metrics
type AdvancedMetrics struct {
	Quantization string

	// Throughput metrics
	InsertThroughput float64 // inserts/sec
	SearchThroughput float64 // searches/sec

	// Latency metrics (microseconds)
	InsertP50 float64
	InsertP95 float64
	InsertP99 float64
	SearchP50 float64
	SearchP95 float64
	SearchP99 float64

	// Accuracy metrics
	RecallAt1  float64 // What % of top-1 results match
	RecallAt5  float64 // What % of top-5 results match
	RecallAt10 float64 // What % of top-10 results match
	MAP        float64 // Mean Average Precision
	NDCG       float64 // Normalized Discounted Cumulative Gain

	// Resource metrics
	PeakMemoryMB    float64
	GCCount         uint32
	TotalAllocMB    float64
	CPUTimeMs       float64
	AllocsPerInsert uint64
	AllocsPerSearch uint64
	BytesPerInsert  uint64
	BytesPerSearch  uint64

	// Quality metrics
	MeanScoreError   float64
	StdDevScoreError float64
	MaxScoreError    float64
	ScoreCorrelation float64 // Spearman correlation of rankings
}

func BenchmarkQuantizationMetrics_Comprehensive(b *testing.B) {
	numVectors := 2000
	vectorDim := 768
	numQueries := 100
	topK := 10

	b.Logf("=== Advanced Metrics Comparison ===")
	b.Logf("Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("Queries: %d searches × top-%d results", numQueries, topK)
	b.Logf("")

	metricsNone := collectAdvancedMetrics(b, "none", numVectors, vectorDim, numQueries, topK)
	metricsScalar := collectAdvancedMetrics(b, "scalar", numVectors, vectorDim, numQueries, topK)

	printAdvancedComparison(b, metricsNone, metricsScalar)
}

func collectAdvancedMetrics(b *testing.B, quantization string, numVectors, vectorDim, numQueries, topK int) AdvancedMetrics {
	tmpDir := b.TempDir()
	walPath := tmpDir + "/test.wal"

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

	vectors := generateDiverseVectors(numVectors, vectorDim)
	queries := generateDiverseVectors(numQueries, vectorDim)

	// Measure insert performance with detailed metrics
	runtime.GC()
	var memStatsBefore, memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	insertLatencies := make([]time.Duration, numVectors)
	insertStart := time.Now()

	for i, vec := range vectors {
		start := time.Now()
		if err := engine.Insert(context.Background(), fmt.Sprintf("vec%d", i), vec, core.SparseVector{}, nil); err != nil {
			b.Fatalf("Insert failed: %v", err)
		}
		insertLatencies[i] = time.Since(start)
	}

	insertTotalTime := time.Since(insertStart)
	runtime.ReadMemStats(&memStatsAfter)

	// Measure search performance with detailed metrics
	searchLatencies := make([]time.Duration, numQueries)
	searchStart := time.Now()

	for i, query := range queries {
		start := time.Now()
		if _, err := engine.Search(context.Background(), query, core.SparseVector{}, topK, nil); err != nil {
			b.Fatalf("Search failed: %v", err)
		}
		searchLatencies[i] = time.Since(start)
	}

	searchTotalTime := time.Since(searchStart)

	// Calculate throughput
	insertThroughput := float64(numVectors) / insertTotalTime.Seconds()
	searchThroughput := float64(numQueries) / searchTotalTime.Seconds()

	// Calculate latency percentiles
	insertP50 := calculatePercentile(insertLatencies, 0.5)
	insertP95 := calculatePercentile(insertLatencies, 0.95)
	insertP99 := calculatePercentile(insertLatencies, 0.99)
	searchP50 := calculatePercentile(searchLatencies, 0.5)
	searchP95 := calculatePercentile(searchLatencies, 0.95)
	searchP99 := calculatePercentile(searchLatencies, 0.99)

	// Memory metrics
	peakMemory := float64(memStatsAfter.Sys) / 1024 / 1024
	totalAlloc := float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc) / 1024 / 1024
	gcCount := memStatsAfter.NumGC - memStatsBefore.NumGC
	allocsPerInsert := (memStatsAfter.Mallocs - memStatsBefore.Mallocs) / uint64(numVectors)
	bytesPerInsert := (memStatsAfter.TotalAlloc - memStatsBefore.TotalAlloc) / uint64(numVectors)

	// For search metrics, we'd need to measure separately
	runtime.ReadMemStats(&memStatsBefore)
	for _, query := range queries[:10] { // Sample 10 searches
		engine.Search(context.Background(), query, core.SparseVector{}, topK, nil)
	}
	runtime.ReadMemStats(&memStatsAfter)
	allocsPerSearch := (memStatsAfter.Mallocs - memStatsBefore.Mallocs) / 10
	bytesPerSearch := (memStatsAfter.TotalAlloc - memStatsBefore.TotalAlloc) / 10

	return AdvancedMetrics{
		Quantization:     quantization,
		InsertThroughput: insertThroughput,
		SearchThroughput: searchThroughput,
		InsertP50:        insertP50,
		InsertP95:        insertP95,
		InsertP99:        insertP99,
		SearchP50:        searchP50,
		SearchP95:        searchP95,
		SearchP99:        searchP99,
		PeakMemoryMB:     peakMemory,
		GCCount:          gcCount,
		TotalAllocMB:     totalAlloc,
		AllocsPerInsert:  allocsPerInsert,
		AllocsPerSearch:  allocsPerSearch,
		BytesPerInsert:   bytesPerInsert,
		BytesPerSearch:   bytesPerSearch,
	}
}

func BenchmarkQuantizationMetrics_Accuracy(b *testing.B) {
	numVectors := 1000
	vectorDim := 512
	numQueries := 50
	topK := 20

	b.Logf("=== Accuracy Metrics Comparison ===")
	b.Logf("Dataset: %d vectors × %d dimensions", numVectors, vectorDim)
	b.Logf("")

	// Create both engines with same data
	tmpDir := b.TempDir()

	wal1, _ := index.NewWAL(tmpDir + "/none.wal")
	defer wal1.Close()
	engineNone, _ := index.NewEngine(config.EngineConfig{Quantization: "none", DistanceMetric: "cosine"}, config.StorageConfig{}, wal1)

	wal2, _ := index.NewWAL(tmpDir + "/scalar.wal")
	defer wal2.Close()
	engineScalar, _ := index.NewEngine(config.EngineConfig{Quantization: "scalar", DistanceMetric: "cosine"}, config.StorageConfig{}, wal2)

	// Insert same vectors
	vectors := generateDiverseVectors(numVectors, vectorDim)
	for i, vec := range vectors {
		id := fmt.Sprintf("vec%d", i)
		engineNone.Insert(context.Background(), id, vec, core.SparseVector{}, nil)
		engineScalar.Insert(context.Background(), id, vec, core.SparseVector{

			// Run queries and compare
		}, nil)
	}

	queries := generateDiverseVectors(numQueries, vectorDim)

	var (
		recallAt1Sum  float64
		recallAt5Sum  float64
		recallAt10Sum float64
		mapSum        float64
		ndcgSum       float64
		scoreErrors   []float64
	)

	for _, query := range queries {
		resultsNone, _ := engineNone.Search(context.Background(), query, core.SparseVector{}, topK, nil)
		resultsScalar, _ := engineScalar.Search(context.Background(), query, core.SparseVector{

			// Calculate recall@K
		}, topK, nil)

		recallAt1Sum += calculateRecall(resultsNone[:min(1, len(resultsNone))], resultsScalar[:min(1, len(resultsScalar))])
		recallAt5Sum += calculateRecall(resultsNone[:min(5, len(resultsNone))], resultsScalar[:min(5, len(resultsScalar))])
		recallAt10Sum += calculateRecall(resultsNone[:min(10, len(resultsNone))], resultsScalar[:min(10, len(resultsScalar))])

		// Calculate MAP and NDCG
		mapSum += calculateMAP(resultsNone, resultsScalar)
		ndcgSum += calculateNDCG(resultsNone, resultsScalar)

		// Calculate score errors
		for i := 0; i < min(len(resultsNone), len(resultsScalar)); i++ {
			if resultsNone[i].ID == resultsScalar[i].ID {
				scoreErrors = append(scoreErrors, math.Abs(float64(resultsNone[i].Score-resultsScalar[i].Score)))
			}
		}
	}

	numQueriesFloat := float64(numQueries)
	recallAt1 := recallAt1Sum / numQueriesFloat
	recallAt5 := recallAt5Sum / numQueriesFloat
	recallAt10 := recallAt10Sum / numQueriesFloat
	mapScore := mapSum / numQueriesFloat
	ndcgScore := ndcgSum / numQueriesFloat

	// Calculate score error statistics
	meanError, stdDevError, maxError := calculateStats(scoreErrors)

	// Print accuracy metrics
	b.Logf("╔════════════════════════════════════════════════╗")
	b.Logf("║         Accuracy Metrics (Scalar vs None)     ║")
	b.Logf("╠════════════════════════════════════════════════╣")
	b.Logf("║ Recall@1                    │ %13.1f%% ║", recallAt1*100)
	b.Logf("║ Recall@5                    │ %13.1f%% ║", recallAt5*100)
	b.Logf("║ Recall@10                   │ %13.1f%% ║", recallAt10*100)
	b.Logf("║ Mean Average Precision      │ %13.4f ║", mapScore)
	b.Logf("║ NDCG@%d                     │ %13.4f ║", topK, ndcgScore)
	b.Logf("╠════════════════════════════════════════════════╣")
	b.Logf("║ Mean Score Error            │ %13.6f ║", meanError)
	b.Logf("║ StdDev Score Error          │ %13.6f ║", stdDevError)
	b.Logf("║ Max Score Error             │ %13.6f ║", maxError)
	b.Logf("╚════════════════════════════════════════════════╝")
	b.Logf("")

	// Verify high accuracy
	if recallAt1 < 0.80 {
		b.Logf("⚠️  Recall@1 (%.1f%%) is below 80%%", recallAt1*100)
	} else {
		b.Logf("✅ Recall@1: %.1f%%", recallAt1*100)
	}

	if recallAt10 < 0.60 {
		b.Logf("⚠️  Recall@10 (%.1f%%) is below 60%%", recallAt10*100)
	} else {
		b.Logf("✅ Recall@10: %.1f%%", recallAt10*100)
	}
}

func BenchmarkQuantizationMetrics_Concurrency(b *testing.B) {
	numVectors := 500
	vectorDim := 384
	numWorkers := 10
	insertsPerWorker := 50
	searchesPerWorker := 100

	b.Logf("=== Concurrency Performance ===")
	b.Logf("Workers: %d", numWorkers)
	b.Logf("Inserts per worker: %d (total: %d)", insertsPerWorker, numWorkers*insertsPerWorker)
	b.Logf("Searches per worker: %d (total: %d)", searchesPerWorker, numWorkers*searchesPerWorker)
	b.Logf("")

	resultsNone := runConcurrencyBenchmark(b, "none", numVectors, vectorDim, numWorkers, insertsPerWorker, searchesPerWorker)
	resultsScalar := runConcurrencyBenchmark(b, "scalar", numVectors, vectorDim, numWorkers, insertsPerWorker, searchesPerWorker)

	b.Logf("╔════════════════════════════════════════════════════════════╗")
	b.Logf("║              Concurrent Operations Performance            ║")
	b.Logf("╠════════════════════════════════════════════════════════════╣")
	b.Logf("║ Metric                  │ None (float32) │ Scalar (int8) ║")
	b.Logf("╠═════════════════════════┼════════════════┼═══════════════╣")
	b.Logf("║ Insert Throughput       │ %9.0f ops/s │ %8.0f ops/s ║",
		resultsNone.InsertThroughput, resultsScalar.InsertThroughput)
	b.Logf("║ Search Throughput       │ %9.0f ops/s │ %8.0f ops/s ║",
		resultsNone.SearchThroughput, resultsScalar.SearchThroughput)
	b.Logf("╚════════════════════════════════════════════════════════════╝")
	b.Logf("")

	speedup := resultsScalar.SearchThroughput / resultsNone.SearchThroughput
	b.Logf("Search throughput speedup: %.2fx", speedup)
}

func runConcurrencyBenchmark(b *testing.B, quantization string, numVectors, vectorDim, numWorkers, insertsPerWorker, searchesPerWorker int) AdvancedMetrics {
	tmpDir := b.TempDir()
	walPath := tmpDir + "/test.wal"

	wal, _ := index.NewWAL(walPath)
	defer wal.Close()

	cfg := config.EngineConfig{Quantization: config.Quantization(quantization), DistanceMetric: "cosine"}
	engine, _ := index.NewEngine(cfg, config.StorageConfig{}, wal)

	// Pre-populate with initial vectors
	vectors := generateDiverseVectors(numVectors, vectorDim)
	for i, vec := range vectors {
		engine.Insert(context.Background(), fmt.Sprintf("initial%d", i), vec, core.SparseVector{

			// Concurrent inserts
		}, nil)
	}

	var wg sync.WaitGroup
	insertStart := time.Now()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < insertsPerWorker; i++ {
				vec := generateDiverseVectors(1, vectorDim)[0]
				id := fmt.Sprintf("worker%d_vec%d", workerID, i)
				engine.Insert(context.Background(), id, vec, core.SparseVector{}, nil)
			}
		}(w)
	}

	wg.Wait()
	insertDuration := time.Since(insertStart)
	insertThroughput := float64(numWorkers*insertsPerWorker) / insertDuration.Seconds()

	// Concurrent searches
	searchStart := time.Now()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < searchesPerWorker; i++ {
				query := generateDiverseVectors(1, vectorDim)[0]
				engine.Search(context.Background(), query, core.SparseVector{}, 10, nil)
			}
		}()
	}

	wg.Wait()
	searchDuration := time.Since(searchStart)
	searchThroughput := float64(numWorkers*searchesPerWorker) / searchDuration.Seconds()

	return AdvancedMetrics{
		Quantization:     quantization,
		InsertThroughput: insertThroughput,
		SearchThroughput: searchThroughput,
	}
}

// Helper functions

func generateDiverseVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
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

func calculatePercentile(durations []time.Duration, percentile float64) float64 {
	sorted := make([]time.Duration, len(durations))
	copy(sorted, durations)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(float64(len(sorted)) * percentile)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return float64(sorted[idx].Microseconds())
}

func calculateRecall(groundTruth, results []index.SearchResult) float64 {
	truthSet := make(map[string]bool)
	for _, r := range groundTruth {
		truthSet[r.ID] = true
	}

	matches := 0
	for _, r := range results {
		if truthSet[r.ID] {
			matches++
		}
	}

	if len(groundTruth) == 0 {
		return 1.0
	}
	return float64(matches) / float64(len(groundTruth))
}

func calculateMAP(groundTruth, results []index.SearchResult) float64 {
	truthSet := make(map[string]bool)
	for _, r := range groundTruth {
		truthSet[r.ID] = true
	}

	precisionSum := 0.0
	matches := 0

	for i, r := range results {
		if truthSet[r.ID] {
			matches++
			precisionSum += float64(matches) / float64(i+1)
		}
	}

	if len(groundTruth) == 0 {
		return 1.0
	}
	return precisionSum / float64(len(groundTruth))
}

func calculateNDCG(groundTruth, results []index.SearchResult) float64 {
	truthMap := make(map[string]float32)
	for i, r := range groundTruth {
		truthMap[r.ID] = float32(len(groundTruth) - i) // Relevance score
	}

	dcg := 0.0
	for i, r := range results {
		rel := float64(truthMap[r.ID])
		dcg += rel / math.Log2(float64(i+2))
	}

	// Calculate ideal DCG
	idcg := 0.0
	for i := range groundTruth {
		rel := float64(len(groundTruth) - i)
		idcg += rel / math.Log2(float64(i+2))
	}

	if idcg == 0 {
		return 1.0
	}
	return dcg / idcg
}

func calculateStats(values []float64) (mean, stdDev, max float64) {
	if len(values) == 0 {
		return 0, 0, 0
	}

	sum := 0.0
	max = values[0]
	for _, v := range values {
		sum += v
		if v > max {
			max = v
		}
	}
	mean = sum / float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	stdDev = math.Sqrt(variance / float64(len(values)))

	return mean, stdDev, max
}

func printAdvancedComparison(b *testing.B, none, scalar AdvancedMetrics) {
	b.Logf("╔═════════════════════════════════════════════════════════════════╗")
	b.Logf("║                   Throughput Performance                        ║")
	b.Logf("╠═════════════════════════════════════════════════════════════════╣")
	b.Logf("║ Insert Throughput  │ %10.0f ops/s │ %10.0f ops/s │ %.2fx ║",
		none.InsertThroughput, scalar.InsertThroughput,
		scalar.InsertThroughput/none.InsertThroughput)
	b.Logf("║ Search Throughput  │ %10.0f ops/s │ %10.0f ops/s │ %.2fx ║",
		none.SearchThroughput, scalar.SearchThroughput,
		scalar.SearchThroughput/none.SearchThroughput)
	b.Logf("╠═════════════════════════════════════════════════════════════════╣")
	b.Logf("║                  Latency (microseconds)                         ║")
	b.Logf("╠═════════════════════════════════════════════════════════════════╣")
	b.Logf("║ Insert P50         │ %10.0f μs    │ %10.0f μs    │ %.2fx ║",
		none.InsertP50, scalar.InsertP50, none.InsertP50/scalar.InsertP50)
	b.Logf("║ Insert P95         │ %10.0f μs    │ %10.0f μs    │ %.2fx ║",
		none.InsertP95, scalar.InsertP95, none.InsertP95/scalar.InsertP95)
	b.Logf("║ Insert P99         │ %10.0f μs    │ %10.0f μs    │ %.2fx ║",
		none.InsertP99, scalar.InsertP99, none.InsertP99/scalar.InsertP99)
	b.Logf("║ Search P50         │ %10.0f μs    │ %10.0f μs    │ %.2fx ║",
		none.SearchP50, scalar.SearchP50, none.SearchP50/scalar.SearchP50)
	b.Logf("║ Search P95         │ %10.0f μs    │ %10.0f μs    │ %.2fx ║",
		none.SearchP95, scalar.SearchP95, none.SearchP95/scalar.SearchP95)
	b.Logf("║ Search P99         │ %10.0f μs    │ %10.0f μs    │ %.2fx ║",
		none.SearchP99, scalar.SearchP99, none.SearchP99/scalar.SearchP99)
	b.Logf("╠═════════════════════════════════════════════════════════════════╣")
	b.Logf("║                    Memory & GC Metrics                          ║")
	b.Logf("╠═════════════════════════════════════════════════════════════════╣")
	b.Logf("║ Peak Memory        │ %10.2f MB    │ %10.2f MB    │ %.2fx ║",
		none.PeakMemoryMB, scalar.PeakMemoryMB, scalar.PeakMemoryMB/none.PeakMemoryMB)
	b.Logf("║ Total Alloc        │ %10.2f MB    │ %10.2f MB    │ %.2fx ║",
		none.TotalAllocMB, scalar.TotalAllocMB, scalar.TotalAllocMB/none.TotalAllocMB)
	b.Logf("║ GC Runs            │ %10d       │ %10d       │      ║",
		none.GCCount, scalar.GCCount)
	b.Logf("║ Allocs/Insert      │ %10d       │ %10d       │      ║",
		none.AllocsPerInsert, scalar.AllocsPerInsert)
	b.Logf("║ Bytes/Insert       │ %10d B     │ %10d B     │      ║",
		none.BytesPerInsert, scalar.BytesPerInsert)
	b.Logf("║ Allocs/Search      │ %10d       │ %10d       │      ║",
		none.AllocsPerSearch, scalar.AllocsPerSearch)
	b.Logf("║ Bytes/Search       │ %10d B     │ %10d B     │      ║",
		none.BytesPerSearch, scalar.BytesPerSearch)
	b.Logf("╚═════════════════════════════════════════════════════════════════╝")
	b.Logf("")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
