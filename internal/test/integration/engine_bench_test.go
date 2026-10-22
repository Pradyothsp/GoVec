package integration

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/wal"
)

// These benchmarks time the engine through its public interface, on every
// index type and quantization, and fail when what they measure is wrong as
// well as slow: search reports recall against exact float32 results and fails
// below a floor. Run them with `task test:benchmark`; -short shrinks the
// synthetic index so a quick check takes seconds. What they run on is in
// bench_data_test.go.
//
// The queries are apart from the data: a query that is also a stored vector
// finds itself, which says nothing about recall.

const benchK = 10

type benchSetup struct {
	indexType config.IndexType
	quant     config.Quantization
}

func (s benchSetup) String() string { return fmt.Sprintf("%s/%s", s.indexType, s.quant) }

var benchSetups = []benchSetup{
	{config.IndexTypeBrute, config.QuantizationNone},
	{config.IndexTypeBrute, config.QuantizationScalar},
	{config.IndexTypeHNSW, config.QuantizationNone},
	{config.IndexTypeHNSW, config.QuantizationScalar},
}

// newBenchEngine builds an empty engine for setup, with a WAL in a temp dir.
func newBenchEngine(b *testing.B, setup benchSetup) index.FileEngine {
	b.Helper()
	level := zerolog.GlobalLevel()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	b.Cleanup(func() { zerolog.SetGlobalLevel(level) })

	walLog, err := wal.Open(filepath.Join(b.TempDir(), "bench.wal"))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = walLog.Close() })
	engine, err := index.NewEngine(config.EngineConfig{
		IndexType:      setup.indexType,
		Quantization:   setup.quant,
		DistanceMetric: config.DistanceMetricCosine,
		// A fixed seed builds the same graph every run, so two runs differ
		// only by the code under test.
		HnswSeed: 1,
	}, config.StorageConfig{}, walLog)
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
// results, failing below the setup's floor for the data.
func BenchmarkSearch(b *testing.B) {
	data := benchDataset(b)

	for _, setup := range benchSetups {
		b.Run(data.name+"/"+setup.String(), func(b *testing.B) {
			engine := newBenchEngine(b, setup)
			fill(b, engine, data.vectors)

			recall := recallAt(b, engine, data.queries, data.truth)
			if floor := data.minRecall[setup]; recall < floor {
				b.Errorf("recall@10 is %.3f, below the floor of %.2f", recall, floor)
			}

			i := 0
			for b.Loop() {
				if _, err := engine.Search(context.Background(), data.queries[i%len(data.queries)], core.SparseVector{}, benchK, nil); err != nil {
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
	data := benchDataset(b)

	for _, setup := range benchSetups {
		b.Run(data.name+"/"+setup.String(), func(b *testing.B) {
			for b.Loop() {
				b.StopTimer()
				engine := newBenchEngine(b, setup)
				b.StartTimer()
				fill(b, engine, data.vectors)
			}
			b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N*len(data.vectors)), "µs/vector")
		})
	}
}

// BenchmarkInsert times one insert into the full index, the way a single REST
// or gRPC insert arrives: its own WAL write and fsync.
func BenchmarkInsert(b *testing.B) {
	data := benchDataset(b)

	for _, setup := range benchSetups {
		b.Run(data.name+"/"+setup.String(), func(b *testing.B) {
			engine := newBenchEngine(b, setup)
			fill(b, engine, data.vectors)

			i := 0
			for b.Loop() {
				if err := engine.Insert(context.Background(), fmt.Sprintf("extra%d", i), data.extra[i%len(data.extra)], core.SparseVector{}, nil); err != nil {
					b.Fatal(err)
				}
				i++
			}
		})
	}
}

// BenchmarkSnapshot times saving and loading the full index and reports the
// snapshot's size per vector. An int8 snapshot holding much more than a
// quarter of float32's bytes per vector fails: the vectors would not be stored
// as int8.
func BenchmarkSnapshot(b *testing.B) {
	data := benchDataset(b)

	for _, setup := range benchSetups {
		b.Run(data.name+"/"+setup.String(), func(b *testing.B) {
			engine := newBenchEngine(b, setup)
			fill(b, engine, data.vectors)
			path := filepath.Join(b.TempDir(), "snapshot.bin")
			if err := engine.SaveToFile(context.Background(), path); err != nil {
				b.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				b.Fatal(err)
			}
			perVector := float64(info.Size()) / float64(len(data.vectors))
			bytesPerElement := map[config.Quantization]float64{config.QuantizationNone: 4, config.QuantizationScalar: 1}[setup.quant]
			if perVector > 1.5*bytesPerElement*float64(data.dims)+512 {
				b.Errorf("%.0f bytes per vector in the snapshot, expected about %.0f", perVector, bytesPerElement*float64(data.dims))
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
