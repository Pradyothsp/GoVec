package integration

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/Pradyothsp/govec/internal/config"
)

// The engine benchmarks run on real text embeddings when GOVEC_BENCH_DATA
// names a directory exported by govec-bench (`task data:export`), and on
// seeded synthetic vectors otherwise, so `task test:benchmark` still runs
// without a download. Synthetic data shows whether a change helps; how much
// it helps a user depends on real embeddings: at 1536 dimensions the distance
// arithmetic is a far larger share of a search than at 256.
//
// The dataset's name is the first level of every benchmark name, so results
// from different data can't be compared by accident.

const benchDataEnv = "GOVEC_BENCH_DATA"

// benchData is what the engine benchmarks index and query.
type benchData struct {
	name    string
	dims    int
	vectors [][]float32 // stored as v0, v1, ...
	queries [][]float32 // none of them in vectors
	// truth holds each query's exact top-benchK IDs, nearest first.
	truth [][]string
	// extra are vectors for single inserts into the full index.
	extra [][]float32
	// minRecall is the lowest recall@10 each setup may report on this data.
	minRecall map[benchSetup]float64
}

// loadBenchData reads the dataset once per test binary: at 100k vectors it is
// 600 MB, and every benchmark uses it.
var loadBenchData = sync.OnceValues(func() (benchData, error) {
	dir := os.Getenv(benchDataEnv)
	if dir == "" {
		return syntheticData(), nil
	}
	return readExportedData(dir)
})

func benchDataset(b *testing.B) benchData {
	b.Helper()
	data, err := loadBenchData()
	if err != nil {
		b.Fatalf("%s: %v", benchDataEnv, err)
	}
	return data
}

// The synthetic data is 256 dimensions, so this is how many clusters its
// vectors are drawn around.
const (
	syntheticDims    = 256
	syntheticCenters = 50
	syntheticQueries = 200
)

// syntheticData is 10,000 clustered vectors, or 2,000 under -short, with
// exact answers from a float32 brute-force scan.
func syntheticData() benchData {
	n := 10_000
	if testing.Short() {
		n = 2_000
	}

	data := benchData{
		name:    "synthetic",
		dims:    syntheticDims,
		vectors: randomVectors(1, n),
		queries: randomVectors(2, syntheticQueries),
		extra:   randomVectors(3, 1_000),
		// A margin below what each setup measured when set: HNSW float32
		// about 0.999, either engine at int8 about 0.845. int8 is the limit,
		// not HNSW: scalar quantization spends its 255 levels on [-1, 1], and
		// a normalized 256-dimension vector's components are mostly within
		// ±0.1.
		minRecall: map[benchSetup]float64{
			{config.IndexTypeBrute, config.QuantizationNone}:   1,
			{config.IndexTypeBrute, config.QuantizationScalar}: 0.80,
			{config.IndexTypeHNSW, config.QuantizationNone}:    0.97,
			{config.IndexTypeHNSW, config.QuantizationScalar}:  0.80,
		},
	}

	data.truth = make([][]string, len(data.queries))
	for i, q := range data.queries {
		data.truth[i] = exactNeighbors(data.vectors, q, benchK)
	}

	return data
}

// randomVectors returns n seeded vectors of syntheticDims, each a random point
// near one of syntheticCenters fixed centres. Embeddings cluster by topic like
// this; uniform random vectors have no structure at all, the worst case for
// any approximate index, and would make HNSW look far worse than it does on
// real data. Every seed draws around the same centres, so queries come from
// the same distribution as the data without being copies of it.
func randomVectors(seed uint64, n int) [][]float32 {
	centerRNG := rand.New(rand.NewPCG(0, 0)) //nolint:gosec // deterministic benchmark data
	centers := make([][]float64, syntheticCenters)
	for c := range centers {
		centers[c] = make([]float64, syntheticDims)
		for d := range centers[c] {
			centers[c][d] = centerRNG.NormFloat64()
		}
	}

	rng := rand.New(rand.NewPCG(seed, 1)) //nolint:gosec // deterministic benchmark data
	vectors := make([][]float32, n)
	for i := range vectors {
		center := centers[rng.IntN(syntheticCenters)]
		v := make([]float32, syntheticDims)
		for d := range v {
			v[d] = float32(center[d] + 0.5*rng.NormFloat64())
		}
		vectors[i] = v
	}
	return vectors
}

// exactNeighbors returns the IDs of q's k nearest vectors by cosine, nearest
// first, computed in float64 apart from any engine code.
func exactNeighbors(vectors [][]float32, q []float32, k int) []string {
	sims := make([]float64, len(vectors))
	for row, v := range vectors {
		sims[row] = cosine(q, v)
	}

	// A stable sort keeps exact ties in row order, as govec-bench's key does.
	rows := make([]int, len(vectors))
	for i := range rows {
		rows[i] = i
	}
	slices.SortStableFunc(rows, func(a, b int) int { return cmp.Compare(sims[b], sims[a]) })

	ids := make([]string, min(k, len(rows)))
	for i := range ids {
		ids[i] = fmt.Sprintf("v%d", rows[i])
	}
	return ids
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// readExportedData reads base.fvecs, query.fvecs and groundtruth.ivecs from
// dir, as govec-bench's export writes them. The answer key is govec-bench's
// float64 brute force, so recall doesn't depend on GoVec's own brute force
// being right. The data is used whole, -short or not: the key is for exactly
// these vectors.
func readExportedData(dir string) (benchData, error) {
	vectors, err := readVecs[float32](filepath.Join(dir, "base.fvecs"))
	if err != nil {
		return benchData{}, err
	}
	queries, err := readVecs[float32](filepath.Join(dir, "query.fvecs"))
	if err != nil {
		return benchData{}, err
	}
	truthRows, err := readVecs[int32](filepath.Join(dir, "groundtruth.ivecs"))
	if err != nil {
		return benchData{}, err
	}

	if len(vectors) == 0 || len(queries) == 0 {
		return benchData{}, fmt.Errorf("%s: no vectors or no queries", dir)
	}
	if len(truthRows) != len(queries) {
		return benchData{}, fmt.Errorf("%s: %d answer rows for %d queries", dir, len(truthRows), len(queries))
	}

	truth := make([][]string, len(truthRows))
	for i, row := range truthRows {
		if len(row) < benchK {
			return benchData{}, fmt.Errorf("%s: answer row %d has %d IDs, need %d", dir, i, len(row), benchK)
		}
		truth[i] = make([]string, benchK)
		for j, r := range row[:benchK] {
			truth[i][j] = fmt.Sprintf("v%d", r)
		}
	}

	return benchData{
		name:    filepath.Base(dir),
		dims:    len(vectors[0]),
		vectors: vectors,
		queries: queries,
		truth:   truth,
		// The queries are text embeddings too, and none is in the index.
		extra:     queries,
		minRecall: exportedFloors(filepath.Base(dir)),
	}, nil
}

// exportedFloors returns the recall@10 floors for an exported dataset: a
// margin below what each setup measured on it (2026-10-22, seed 1). Brute
// force over float32 is the exact answer on any data, so it must find every
// true neighbour even on a dataset with no floors of its own.
func exportedFloors(name string) map[benchSetup]float64 {
	floors := map[benchSetup]float64{
		{config.IndexTypeBrute, config.QuantizationNone}: 1,
	}

	measured := map[string]map[benchSetup]float64{
		// Measured: int8 0.901 (brute) and 0.895 (HNSW), HNSW float32 0.994.
		"dbpedia10k": {
			{config.IndexTypeBrute, config.QuantizationScalar}: 0.85,
			{config.IndexTypeHNSW, config.QuantizationNone}:    0.97,
			{config.IndexTypeHNSW, config.QuantizationScalar}:  0.85,
		},
		// Measured: int8 0.890 (brute) and 0.885 (HNSW), HNSW float32 0.984.
		"dbpedia100k": {
			{config.IndexTypeBrute, config.QuantizationScalar}: 0.84,
			{config.IndexTypeHNSW, config.QuantizationNone}:    0.95,
			{config.IndexTypeHNSW, config.QuantizationScalar}:  0.83,
		},
	}
	for setup, floor := range measured[name] {
		floors[setup] = floor
	}

	return floors
}

// readVecs reads an fvecs or ivecs file: per row, a little-endian int32 count
// and then that many 4-byte values.
func readVecs[T float32 | int32](path string) ([][]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var rows [][]T
	for {
		var n int32
		if err := binary.Read(f, binary.LittleEndian, &n); err == io.EOF {
			return rows, nil
		} else if err != nil {
			return nil, fmt.Errorf("%s: row %d: %w", path, len(rows), err)
		}

		row := make([]T, n)
		if err := binary.Read(f, binary.LittleEndian, row); err != nil {
			return nil, fmt.Errorf("%s: row %d: %w", path, len(rows), err)
		}
		rows = append(rows, row)
	}
}
