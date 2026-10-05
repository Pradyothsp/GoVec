package index

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"
)

// WAL benchmarks at the shape that made the WAL a problem (govec #15): real
// valued 1536-dimension embeddings with a little metadata, written in batches
// of 100 as the bench loads them. They use only WriteEntries and replay, so the
// same benchmarks measure any entry format.
//
//	go test ./internal/index/ -run '^$' -bench BenchmarkWAL -benchtime 3x

const (
	walBenchDims  = 1536
	walBenchBatch = 100
)

// quietLogs silences the WAL's info logs, which otherwise land in the
// benchmark output between result lines.
func quietLogs(b *testing.B) {
	level := zerolog.GlobalLevel()
	zerolog.SetGlobalLevel(zerolog.Disabled)
	b.Cleanup(func() { zerolog.SetGlobalLevel(level) })
}

func walBenchEntries(n int) []*WALEntry {
	rng := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // deterministic benchmark data
	entries := make([]*WALEntry, n)
	for i := range entries {
		vec := make([]float32, walBenchDims)
		for d := range vec {
			vec[d] = float32(rng.NormFloat64() * 0.025) // the scale of a unit-length 1536-d embedding
		}
		entries[i] = &WALEntry{
			Action: WALActionInsert,
			ID:     fmt.Sprintf("doc-%d", i),
			Vector: vec,
			Meta:   map[string]any{"source": "dbpedia", "chunk": float64(i % 7)},
		}
	}
	return entries
}

// BenchmarkWALWriteEntries reports the time per entry (encoding plus one fsync
// per batch of 100) and the WAL's size per entry.
func BenchmarkWALWriteEntries(b *testing.B) {
	quietLogs(b)
	entries := walBenchEntries(1000)
	var bytesPerEntry float64

	for b.Loop() {
		b.StopTimer()
		path := filepath.Join(b.TempDir(), "bench.wal")
		wal, err := NewWAL(path)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()

		for i := 0; i < len(entries); i += walBenchBatch {
			if err := wal.WriteEntries(context.Background(), entries[i:i+walBenchBatch]); err != nil {
				b.Fatal(err)
			}
		}

		b.StopTimer()
		info, err := os.Stat(path)
		if err != nil {
			b.Fatal(err)
		}
		bytesPerEntry = float64(info.Size()) / float64(len(entries))
		_ = wal.Close()
		b.StartTimer()
	}

	b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N*len(entries)), "µs/entry")
	b.ReportMetric(bytesPerEntry, "bytes/entry")
	b.ReportMetric(bytesPerEntry/(walBenchDims*4), "x-raw-vector")
}

// BenchmarkWALReplay reports the time per entry to read a WAL back.
func BenchmarkWALReplay(b *testing.B) {
	quietLogs(b)
	entries := walBenchEntries(1000)
	path := filepath.Join(b.TempDir(), "bench.wal")
	wal, err := NewWAL(path)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < len(entries); i += walBenchBatch {
		if err := wal.WriteEntries(context.Background(), entries[i:i+walBenchBatch]); err != nil {
			b.Fatal(err)
		}
	}
	_ = wal.Close()

	for b.Loop() {
		replayed := 0
		if err := replayWALFile(path, func(WALEntry) error { replayed++; return nil }); err != nil {
			b.Fatal(err)
		}
		if replayed != len(entries) {
			b.Fatalf("replayed %d of %d entries", replayed, len(entries))
		}
	}

	b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N*len(entries)), "µs/entry")
}
