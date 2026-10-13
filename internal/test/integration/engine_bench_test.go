package integration

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
)

// These benchmarks time the engine through its public interface, on every
// index type and quantization, and fail when what they measure is wrong as
// well as slow: search reports recall against exact float32 results and fails
// below a floor. Run them with `task test:benchmark`; -short shrinks the index
// so a quick check takes seconds.
//
// The queries are drawn apart from the data: a query that is also a stored
// vector finds itself, which says nothing about recall.

const (
	benchDims    = 256
	benchK       = 10
	benchQueries = 200
)

type benchSetup struct {
	indexType config.IndexType
	quant     config.Quantization
	// minRecall is the lowest recall@10 against exact float32 results this
	// setup may report. Brute force over float32 is the exact answer, so 1.
	// The others sit a margin below what they measured when set -- HNSW
	// float32 about 0.999, either engine at int8 about 0.845 -- leaving room
	// for HNSW's time-seeded level assignment. int8 is the limit, not HNSW:
	// scalar quantization spends its 255 levels on [-1, 1], and a normalized
	// 256-dimension vector's components are mostly within ±0.1.
	minRecall float64
}

func (s benchSetup) String() string { return fmt.Sprintf("%s/%s", s.indexType, s.quant) }

var benchSetups = []benchSetup{
	{config.IndexTypeBrute, config.QuantizationNone, 1},
	{config.IndexTypeBrute, config.QuantizationScalar, 0.80},
	{config.IndexTypeHNSW, config.QuantizationNone, 0.97},
	{config.IndexTypeHNSW, config.QuantizationScalar, 0.80},
}

// benchSize is the number of vectors to index: 10,000, or 2,000 under -short.
func benchSize() int {
	if testing.Short() {
		return 2_000
	}
	return 10_000
}

// benchCenters is how many clusters the benchmark data is drawn around.
const benchCenters = 50

// randomVectors returns n seeded vectors of benchDims, each a random point
// near one of benchCenters fixed centres. Embeddings cluster by topic like
// this; uniform random vectors have no structure at all, the worst case for
// any approximate index, and would make HNSW look far worse than it does on
// real data. Every seed draws around the same centres, so queries come from
// the same distribution as the data without being copies of it.
func randomVectors(seed uint64, n int) [][]float32 {
	centerRNG := rand.New(rand.NewPCG(0, 0)) //nolint:gosec // deterministic benchmark data
	centers := make([][]float64, benchCenters)
	for c := range centers {
		centers[c] = make([]float64, benchDims)
		for d := range centers[c] {
			centers[c][d] = centerRNG.NormFloat64()
		}
	}

	rng := rand.New(rand.NewPCG(seed, 1)) //nolint:gosec // deterministic benchmark data
	vectors := make([][]float32, n)
	for i := range vectors {
		center := centers[rng.IntN(benchCenters)]
		v := make([]float32, benchDims)
		for d := range v {
			v[d] = float32(center[d] + 0.5*rng.NormFloat64())
		}
		vectors[i] = v
	}
	return vectors
}

// newBenchEngine builds an empty engine for setup, with a WAL in a temp dir.
func newBenchEngine(b *testing.B, setup benchSetup) index.FileEngine {
	b.Helper()
	level := zerolog.GlobalLevel()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	b.Cleanup(func() { zerolog.SetGlobalLevel(level) })

	wal, err := index.NewWAL(filepath.Join(b.TempDir(), "bench.wal"))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = wal.Close() })
	engine, err := index.NewEngine(config.EngineConfig{
		IndexType:      setup.indexType,
		Quantization:   setup.quant,
		DistanceMetric: config.DistanceMetricCosine,
	}, config.StorageConfig{}, wal)
	if err != nil {
		b.Fatal(err)
	}
	return engine
}

// fill batch-inserts vectors into engine as v0, v1, ...
func fill(b *testing.B, engine index.Engine, vectors [][]float32) {
	b.Helper()
	const chunk = 1_000
	for start := 0; start < len(vectors); start += chunk {
		items := make([]index.BatchInsertItem, 0, chunk)
		for i := start; i < min(start+chunk, len(vectors)); i++ {
			items = append(items, index.BatchInsertItem{ID: fmt.Sprintf("v%d", i), Vector: vectors[i]})
		}
		failures, err := engine.BatchInsert(context.Background(), items)
		if err != nil || len(failures) > 0 {
			b.Fatalf("fill: %v, %d failed items", err, len(failures))
		}
	}
}

// recallAt returns the share of truth's IDs that got contains, over queries.
func recallAt(b *testing.B, engine index.Engine, queries [][]float32, truth [][]string) float64 {
	b.Helper()
	found, total := 0, 0
	for i, q := range queries {
		results, err := engine.Search(context.Background(), q, core.SparseVector{}, benchK, nil)
		if err != nil {
			b.Fatal(err)
		}
		want := make(map[string]bool, len(truth[i]))
		for _, id := range truth[i] {
			want[id] = true
		}
		for _, r := range results {
			if want[r.ID] {
				found++
			}
		}
		total += len(truth[i])
	}
	return float64(found) / float64(total)
}

// BenchmarkSearch times one top-10 query and reports recall@10 against exact
// float32 results, failing below the setup's floor.
func BenchmarkSearch(b *testing.B) {
	vectors := randomVectors(1, benchSize())
	queries := randomVectors(2, benchQueries)

	exact := newBenchEngine(b, benchSetups[0])
	fill(b, exact, vectors)
	truth := make([][]string, len(queries))
	for i, q := range queries {
		results, err := exact.Search(context.Background(), q, core.SparseVector{}, benchK, nil)
		if err != nil {
			b.Fatal(err)
		}
		truth[i] = resultIDsOf(results)
	}

	for _, setup := range benchSetups {
		b.Run(setup.String(), func(b *testing.B) {
			engine := newBenchEngine(b, setup)
			fill(b, engine, vectors)

			recall := recallAt(b, engine, queries, truth)
			if recall < setup.minRecall {
				b.Errorf("recall@10 is %.3f, below the floor of %.2f", recall, setup.minRecall)
			}

			i := 0
			for b.Loop() {
				if _, err := engine.Search(context.Background(), queries[i%len(queries)], core.SparseVector{}, benchK, nil); err != nil {
					b.Fatal(err)
				}
				i++
			}
			// After the loop: b.Loop clears metrics reported before it starts.
			b.ReportMetric(recall, "recall@10")
		})
	}
}

// BenchmarkBuild times building a whole index through BatchInsert, the way a
// bulk load arrives: one WAL write per batch, and HNSW's parallel graph build.
func BenchmarkBuild(b *testing.B) {
	vectors := randomVectors(1, benchSize())
	for _, setup := range benchSetups {
		b.Run(setup.String(), func(b *testing.B) {
			for b.Loop() {
				b.StopTimer()
				engine := newBenchEngine(b, setup)
				b.StartTimer()
				fill(b, engine, vectors)
			}
			b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N*len(vectors)), "µs/vector")
		})
	}
}

// BenchmarkInsert times one insert into an index of benchSize vectors, the
// way a single REST or gRPC insert arrives: its own WAL write and fsync.
func BenchmarkInsert(b *testing.B) {
	vectors := randomVectors(1, benchSize())
	extra := randomVectors(3, 1_000)
	for _, setup := range benchSetups {
		b.Run(setup.String(), func(b *testing.B) {
			engine := newBenchEngine(b, setup)
			fill(b, engine, vectors)

			i := 0
			for b.Loop() {
				if err := engine.Insert(context.Background(), fmt.Sprintf("extra%d", i), extra[i%len(extra)], core.SparseVector{}, nil); err != nil {
					b.Fatal(err)
				}
				i++
			}
		})
	}
}

// BenchmarkSnapshot times saving and loading an index of benchSize vectors and
// reports the snapshot's size per vector. An int8 snapshot holding much more
// than a quarter of float32's bytes per vector fails: the vectors would not
// be stored as int8.
func BenchmarkSnapshot(b *testing.B) {
	vectors := randomVectors(1, benchSize())
	for _, setup := range benchSetups {
		b.Run(setup.String(), func(b *testing.B) {
			engine := newBenchEngine(b, setup)
			fill(b, engine, vectors)
			path := filepath.Join(b.TempDir(), "snapshot.bin")
			if err := engine.SaveToFile(context.Background(), path); err != nil {
				b.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				b.Fatal(err)
			}
			perVector := float64(info.Size()) / float64(len(vectors))
			bytesPerElement := map[config.Quantization]float64{config.QuantizationNone: 4, config.QuantizationScalar: 1}[setup.quant]
			if perVector > 1.5*bytesPerElement*benchDims+512 {
				b.Errorf("%.0f bytes per vector in the snapshot, expected about %.0f", perVector, bytesPerElement*benchDims)
			}

			b.Run("save", func(b *testing.B) {
				for b.Loop() {
					if err := engine.SaveToFile(context.Background(), path); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(perVector, "bytes/vector")
			})
			b.Run("load", func(b *testing.B) {
				for b.Loop() {
					b.StopTimer()
					loaded := newBenchEngine(b, setup)
					b.StartTimer()
					if err := loaded.LoadFromFile(context.Background(), path); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func resultIDsOf(results []index.SearchResult) []string {
	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = r.ID
	}
	return ids
}
