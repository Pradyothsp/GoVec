package index

import (
	"bytes"
	"context"
	"encoding/gob"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGobChunkStream_RoundTrip(t *testing.T) {
	sizes := map[string]int{
		"empty":                  0,
		"smaller than a chunk":   100,
		"exactly one chunk":      snapshotChunkSize,
		"one chunk and a byte":   snapshotChunkSize + 1,
		"several chunks, ragged": 3*snapshotChunkSize + 12345,
	}

	for name, size := range sizes {
		t.Run(name, func(t *testing.T) {
			data := make([]byte, size)
			for i := range data {
				data[i] = byte(i * 7)
			}

			var stream bytes.Buffer
			enc := gob.NewEncoder(&stream)
			w := newGobChunkWriter(enc)
			// Odd-sized writes, so chunk boundaries fall mid-write.
			for rest := data; len(rest) > 0; {
				n := min(len(rest), 7919)
				_, err := w.Write(rest[:n])
				require.NoError(t, err)
				rest = rest[n:]
			}
			require.NoError(t, w.Close())
			// A value after the chunks: the reader must stop exactly at the end marker.
			require.NoError(t, enc.Encode("after"))

			dec := gob.NewDecoder(&stream)
			got, err := io.ReadAll(newGobChunkReader(dec))
			require.NoError(t, err)
			assert.True(t, bytes.Equal(data, got), "round-tripped bytes differ")

			var after string
			require.NoError(t, dec.Decode(&after))
			assert.Equal(t, "after", after)
		})
	}
}

// TestLoadFromFile_OtherSnapshotVersion_RejectedWithClearError keeps an old
// file from failing later with an opaque gob error.
func TestLoadFromFile_OtherSnapshotVersion_RejectedWithClearError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.bin")
	var file bytes.Buffer
	require.NoError(t, gob.NewEncoder(&file).Encode(SnapshotHeader{Version: 1, Quantization: "none", DistanceMetric: "cosine", IndexType: "hnsw"}))
	require.NoError(t, os.WriteFile(path, file.Bytes(), 0o600))

	err := newTestHNSWIndex(t).LoadFromFile(context.Background(), path)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "snapshot format version 1")
}

// TestSaveToFile_HNSW_WritesVectorsOnceAndStreams guards the two properties
// that let a 100k x 1536 index save inside a 2 GB container:
//   - the file holds each vector once (the graph is topology only), where it
//     used to hold them about 2.1x;
//   - a save allocates a small fraction of the vectors' size, where it used to
//     build every section as one in-memory gob value first -- the copy that
//     got the container OOM-killed.
func TestSaveToFile_HNSW_WritesVectorsOnceAndStreams(t *testing.T) {
	const count, dims = 600, 1536
	rawBytes := float64(count * dims * 4)

	idx := newTestHNSWIndex(t)
	rng := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // deterministic test data
	items := make([]BatchInsertItem, count)
	for i := range items {
		vec := make([]float32, dims)
		for d := range vec {
			vec[d] = float32(rng.NormFloat64())
		}
		items[i] = BatchInsertItem{ID: fmt.Sprintf("v%d", i), Vector: vec}
	}
	failures, err := idx.BatchInsert(context.Background(), items)
	require.NoError(t, err)
	require.Empty(t, failures)

	path := filepath.Join(t.TempDir(), "snapshot.bin")
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	require.NoError(t, idx.SaveToFile(context.Background(), path))
	runtime.ReadMemStats(&after)

	info, err := os.Stat(path)
	require.NoError(t, err)
	fileRatio := float64(info.Size()) / rawBytes
	// The file buffer and the topology chunk buffer are fixed costs, not a
	// copy of the index; leave them out so the ratio means something at this size.
	fixedBuffers := uint64(2 * snapshotChunkSize)
	allocRatio := float64(after.TotalAlloc-before.TotalAlloc-fixedBuffers) / rawBytes
	t.Logf("snapshot %.2fx raw vectors; save allocated %.2fx raw vectors beyond its fixed buffers", fileRatio, allocRatio)

	assert.Less(t, fileRatio, 1.2, "the snapshot should hold each vector once")
	// The single-value format allocated over 2x; the race detector inflates
	// allocation, so the bound leaves it room while still catching that.
	assert.Less(t, allocRatio, 0.5, "a save should stream, not build the snapshot in memory")
}

func TestLoadedDimensions(t *testing.T) {
	cases := []struct {
		name                       string
		saved, configured, current int
		want                       int
		wantErr                    bool
	}{
		{name: "learned width comes back", saved: 1536, want: 1536},
		{name: "configured width that matches", saved: 1536, configured: 1536, current: 1536, want: 1536},
		{name: "empty snapshot keeps the configured width", configured: 768, current: 768, want: 768},
		{name: "empty snapshot, nothing configured: still unset", want: 0},
		{name: "configured width that disagrees with the data", saved: 1536, configured: 768, current: 768, wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := loadedDimensions(tc.saved, tc.configured, tc.current)
			if tc.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "snapshot dimensions mismatch")
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}
