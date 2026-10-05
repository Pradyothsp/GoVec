package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/rs/zerolog"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/storage"
)

func init() {
	// Register types for GOB encoding/decoding of metadata
	gob.Register(map[string]any{})
	gob.Register([]any{})
	// Register types for inverted index
	gob.Register(core.Posting{})
}

// SnapshotHeader contains metadata about the snapshot file format
type SnapshotHeader struct {
	Version        int    // snapshotVersion at save time; a load rejects any other
	Quantization   string // "none" or "scalar"
	DistanceMetric string // "cosine", etc.
	IndexType      string // "" or "brute" = VectorIndex; "hnsw" = HNSWIndex
	Mmap           bool   // vectors live in the mmap store, not inline in the snapshot
	Dimensions     int    // the index's width, configured or learned; 0 for an empty index that never had one
}

// snapshotVersion is written into every snapshot header, and a load rejects
// any other (checkSnapshotVersion). 2: streamed records, HNSW topology only
// (snapshot_stream.go). 1: each section as one gob value, vectors twice for HNSW.
const snapshotVersion = 2

// The inverted index is never written to a snapshot. It is derived data --
// every posting comes from a node's sparse vector, which the snapshot does
// store -- so each load path rebuilds it instead. A saved copy could only
// disagree with the store it was derived from.

// ensureParentDir creates the directory a snapshot is written into, so a
// nested data_path works on first save instead of failing every auto-save.
func ensureParentDir(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil { //nolint:gosec // 0o750 satisfies G301
		return fmt.Errorf("create snapshot directory: %w", err)
	}
	return nil
}

// publishSnapshot makes a fully written temporary snapshot the live one, and
// only then truncates the WAL.
//
// Until the new snapshot is durably on disk, the WAL is the only durable copy
// of every write since the previous snapshot. Truncating it first meant any
// failure in between -- out of memory, disk full, a crash -- lost all of those
// writes at the next restart, while the server kept serving them from memory.
// govec-bench hit exactly that: OOM-killed mid-save, right after "WAL cleared".
//
// The order, each step only if the one before succeeded:
//   - fsync the file, so the rename can't publish a snapshot whose bytes are
//     still only in the page cache;
//   - rename it over the old snapshot (atomic: readers see old or new, whole);
//   - fsync the directory, so the rename itself survives a power cut;
//   - truncate the WAL.
//
// A failure before the rename removes the temporary file and keeps the WAL, so
// recovery replays it over the previous snapshot. A failure after the rename
// keeps the WAL too; replaying entries the new snapshot already holds is safe
// because replay applies inserts as upserts and deletes of missing IDs as
// no-ops.
//
// PRECONDITION: the index write lock is held, so no write can land between the
// snapshot and the truncation.
func publishSnapshot(f *os.File, tmpPath, path string, wal *WAL) error {
	if err := f.Sync(); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; sync error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; sync error takes precedence
		return fmt.Errorf("sync snapshot: %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; close error takes precedence
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; rename error takes precedence
		return err
	}
	if err := syncDir(filepath.Dir(path)); err != nil {
		return fmt.Errorf("sync snapshot directory: %w", err)
	}

	if wal == nil {
		return nil
	}
	return wal.Clear()
}

// syncDir fsyncs a directory, making a rename inside it durable. Windows
// can't open a directory for syncing, and NTFS journals the rename itself.
func syncDir(dir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}

	d, err := os.Open(dir) //nolint:gosec // dir of the operator-configured data_path
	if err != nil {
		return err
	}
	if err := d.Sync(); err != nil {
		_ = d.Close() //nolint:errcheck // best-effort cleanup; sync error takes precedence
		return err
	}
	return d.Close()
}

// loadedDimensions decides an index's width after loading a snapshot that
// recorded savedDims. A learned width has to come back with the data: without
// it the restarted index takes its width from the next insert, whatever that
// is, and one wrong-width vector gets every real insert and query rejected.
// A configured width that disagrees with the data is an error, like a
// quantization or metric mismatch.
func loadedDimensions(savedDims, configuredDims, currentDims int) (int, error) {
	if savedDims == 0 {
		return currentDims, nil
	}
	if configuredDims > 0 && savedDims != configuredDims {
		return 0, fmt.Errorf("snapshot dimensions mismatch: config expects %d but snapshot holds %d-dimensional vectors. Delete data files or change engine.dimensions", configuredDims, savedDims)
	}
	return savedDims, nil
}

// SaveToFile serializes the index to a specific path.
// When mmap is enabled (vectorStore != nil) the vectors stay in the mmap store
// and the snapshot carries everything else; otherwise vectors are inline.
func (idx *VectorIndex[T]) SaveToFile(ctx context.Context, path string) error {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	var quantType string
	switch any(*new(T)).(type) {
	case []float32:
		quantType = "none"
	case []int8:
		quantType = "scalar"
	default:
		return fmt.Errorf("unknown vector type")
	}

	if err := ensureParentDir(path); err != nil {
		return err
	}

	err := idx.saveSnapshot(path, quantType)

	if err == nil {
		zerolog.Ctx(ctx).Info().
			Str("path", path).
			Int("vectors", len(idx.Store)).
			Dur("duration", time.Since(start)).
			Msg("snapshot saved to disk")
	}
	return err
}

// saveSnapshot writes the snapshot (layout in snapshot_stream.go) and
// publishes it. PRECONDITION: idx.mu.Lock() held.
func (idx *VectorIndex[T]) saveSnapshot(path, quantType string) error {
	if idx.vectorStore != nil {
		// The vectors live only in the mmap store, so they must be on disk before
		// publishSnapshot lets the WAL go.
		if err := idx.vectorStore.Sync(); err != nil {
			return fmt.Errorf("mmap sync: %w", err)
		}
	}

	w, err := newSnapshotWriter(path)
	if err != nil {
		return err
	}

	header := SnapshotHeader{
		Version:        snapshotVersion,
		Quantization:   quantType,
		DistanceMetric: idx.distanceMetric,
		Mmap:           idx.vectorStore != nil,
		Dimensions:     idx.dimensions,
	}
	if err := w.enc.Encode(header); err != nil {
		w.abort()
		return err
	}
	if err := idx.IDMapper.EncodeGOB(w.enc); err != nil {
		w.abort()
		return err
	}
	if err := writeNodeRecords(w.enc, idx.Store, idx.vectorStore == nil); err != nil {
		w.abort()
		return err
	}

	return w.publish(idx.wal)
}

// LoadFromFile reads the index from disk.
// A missing file is not an error — the server starts with an empty index.
func (idx *VectorIndex[T]) LoadFromFile(ctx context.Context, path string) (err error) {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path) //nolint:gosec // path from operator config
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	decoder := gob.NewDecoder(f)

	var header SnapshotHeader
	if err := decoder.Decode(&header); err != nil {
		return fmt.Errorf("failed to decode snapshot header: %w", err)
	}

	if header.IndexType == "hnsw" {
		return fmt.Errorf("snapshot was created with HNSW index. Delete data files or change index_type to 'hnsw' in config")
	}
	if err := checkSnapshotVersion(header); err != nil {
		return err
	}

	var expectedQuant string
	switch any(*new(T)).(type) {
	case []float32:
		expectedQuant = "none"
	case []int8:
		expectedQuant = "scalar"
	default:
		return fmt.Errorf("unknown vector type")
	}
	if header.Quantization != expectedQuant {
		return fmt.Errorf("snapshot quantization mismatch: config expects '%s' but snapshot is '%s'. Delete data files or change config", expectedQuant, header.Quantization)
	}
	if header.DistanceMetric != idx.distanceMetric {
		return fmt.Errorf("snapshot distance metric mismatch: config expects '%s' but snapshot is '%s'. Delete data files or change config", idx.distanceMetric, header.DistanceMetric)
	}
	dims, err := loadedDimensions(header.Dimensions, idx.configuredDimensions, idx.dimensions)
	if err != nil {
		return err
	}
	idx.dimensions = dims

	if err := idx.IDMapper.DecodeGOB(decoder); err != nil {
		return fmt.Errorf("failed to decode IDMapper: %w", err)
	}

	if header.Mmap && idx.vectorStore == nil {
		return fmt.Errorf("snapshot keeps its vectors in an mmap store, but storage.enable_mmap is off")
	}
	var mmapStore *storage.MmapStore
	if header.Mmap {
		mmapStore = idx.vectorStore
	}
	if err := readNodeRecords(decoder, idx.Store, mmapStore); err != nil {
		return err
	}

	if idx.metaIndex != nil {
		idx.metaIndex.Clear()
		for internalID, node := range idx.Store {
			idx.metaIndex.Add(internalID, node.Metadata)
		}
	}

	idx.rebuildInvertedIndex()

	if err == nil {
		zerolog.Ctx(ctx).Info().
			Str("path", path).
			Int("vectors", len(idx.Store)).
			Dur("duration", time.Since(start)).
			Msg("snapshot loaded from disk")
	}

	return err
}
