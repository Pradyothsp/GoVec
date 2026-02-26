package index

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/Pradyothsp/govec/internal/core"
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
	Version             int    // File format version (3=VectorIndex, 4=HNSWIndex, 5=VectorIndex+mmap, 6=HNSWIndex+mmap)
	Quantization        string // "none" or "scalar"
	DistanceMetric      string // "cosine", etc.
	HybridSearchEnabled bool   // Whether inverted index is included
	IndexType           string // "" or "brute" = VectorIndex; "hnsw" = HNSWIndex
}

// vectorNodeSnapshot is a vector-free snapshot of a VectorNode used when mmap
// is enabled.  The Vector field is intentionally omitted: vectors live in the
// mmap store and are restored by reading from that store at load time.
type vectorNodeSnapshot struct {
	InternalID uint32
	ExternalID string
	Sparse     core.SparseVector
	Metadata   map[string]any
}

// SaveToFile serializes the index to a specific path.
// When mmap is enabled (vectorStore != nil) it writes version 5 (no inline vectors).
// Otherwise it writes the standard version 3 format.
func (idx *VectorIndex[T]) SaveToFile(path string) error {
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

	if idx.vectorStore != nil {
		return idx.saveToFileMmap(path, quantType)
	}
	return idx.saveToFileHeap(path, quantType)
}

// saveToFileHeap writes the standard (heap-backed) v3 snapshot.
// PRECONDITION: idx.mu.Lock() held.
func (idx *VectorIndex[T]) saveToFileHeap(path, quantType string) error {
	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			return err
		}
	}

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path from operator config
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(f)
	header := SnapshotHeader{
		Version:             3,
		Quantization:        quantType,
		DistanceMetric:      "cosine",
		HybridSearchEnabled: idx.InvertedIndex != nil,
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := idx.IDMapper.EncodeGOB(encoder); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := encoder.Encode(idx.Store); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if idx.InvertedIndex != nil {
		if err := encoder.Encode(idx.InvertedIndex); err != nil {
			_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			return err
		}
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	return nil
}

// saveToFileMmap writes a v5 snapshot (no inline vectors; vectors live in mmap).
// PRECONDITION: idx.mu.Lock() held.
func (idx *VectorIndex[T]) saveToFileMmap(path, quantType string) error {
	// Flush mmap pages to disk before clearing WAL.
	if err := idx.vectorStore.Sync(); err != nil {
		return fmt.Errorf("mmap sync: %w", err)
	}

	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			return err
		}
	}

	// Build vector-free snapshots.
	snapshots := make([]vectorNodeSnapshot, 0, len(idx.Store))
	for _, node := range idx.Store {
		snapshots = append(snapshots, vectorNodeSnapshot{
			InternalID: node.InternalID,
			ExternalID: node.ExternalID,
			Sparse:     node.Sparse,
			Metadata:   node.Metadata,
		})
	}

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path from operator config
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(f)
	header := SnapshotHeader{
		Version:             5,
		Quantization:        quantType,
		DistanceMetric:      "cosine",
		HybridSearchEnabled: idx.InvertedIndex != nil,
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := idx.IDMapper.EncodeGOB(encoder); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := encoder.Encode(snapshots); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if idx.InvertedIndex != nil {
		if err := encoder.Encode(idx.InvertedIndex); err != nil {
			_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			return err
		}
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}
	return nil
}

// LoadFromFile reads the index from disk.
// A missing file is not an error — the server starts with an empty index.
func (idx *VectorIndex[T]) LoadFromFile(path string) (err error) {
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

	if header.Version < 1 || header.Version > 5 {
		return fmt.Errorf("unsupported snapshot version: %d (expected 1-5)", header.Version)
	}

	if header.IndexType == "hnsw" {
		return fmt.Errorf("snapshot was created with HNSW index. Delete data files or change index_type to 'hnsw' in config")
	}

	if header.Version < 3 {
		return fmt.Errorf("snapshot version %d uses string IDs (incompatible with v3 uint32 IDs). Delete data files and re-insert data", header.Version)
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

	if err := idx.IDMapper.DecodeGOB(decoder); err != nil {
		return fmt.Errorf("failed to decode IDMapper: %w", err)
	}

	if header.Version == 5 {
		// Mmap format: vectors live in the mmap store, not in the GOB stream.
		return idx.loadFromFileMmap(decoder, &header)
	}

	// Standard heap format (v3).
	if err := decoder.Decode(&idx.Store); err != nil {
		return fmt.Errorf("failed to decode store: %w", err)
	}

	if idx.metaIndex != nil {
		idx.metaIndex.Clear()
		for internalID, node := range idx.Store {
			idx.metaIndex.Add(internalID, node.Metadata)
		}
	}

	if header.Version == 2 && header.HybridSearchEnabled {
		if idx.InvertedIndex != nil {
			var loadedIndex map[uint32][]core.Posting
			if err := decoder.Decode(&loadedIndex); err != nil {
				return fmt.Errorf("failed to decode inverted index: %w", err)
			}
			idx.InvertedIndex = loadedIndex
		} else {
			var discarded map[uint32][]core.Posting
			if err := decoder.Decode(&discarded); err != nil {
				return fmt.Errorf("failed to skip inverted index: %w", err)
			}
		}
	}

	return nil
}

// loadFromFileMmap reconstructs the Store from mmap-backed snapshots.
// PRECONDITION: idx.mu.Lock() held; IDMapper already decoded.
func (idx *VectorIndex[T]) loadFromFileMmap(decoder *gob.Decoder, header *SnapshotHeader) error {
	if idx.vectorStore == nil {
		return fmt.Errorf("snapshot is mmap format (v5) but mmap storage is not configured")
	}

	var snapshots []vectorNodeSnapshot
	if err := decoder.Decode(&snapshots); err != nil {
		return fmt.Errorf("failed to decode mmap snapshots: %w", err)
	}

	for _, snap := range snapshots {
		vec, ok := getFromMmapStore[T](idx.vectorStore, snap.InternalID)
		if !ok {
			return fmt.Errorf("mmap slot missing for internalID %d (externalID %q)", snap.InternalID, snap.ExternalID)
		}
		idx.Store[snap.InternalID] = &core.VectorNode[T]{
			InternalID: snap.InternalID,
			ExternalID: snap.ExternalID,
			Vector:     vec,
			Sparse:     snap.Sparse,
			Metadata:   snap.Metadata,
		}
	}

	if idx.metaIndex != nil {
		idx.metaIndex.Clear()
		for internalID, node := range idx.Store {
			idx.metaIndex.Add(internalID, node.Metadata)
		}
	}

	if header.HybridSearchEnabled {
		if idx.InvertedIndex != nil {
			var loadedIndex map[uint32][]core.Posting
			if err := decoder.Decode(&loadedIndex); err != nil {
				return fmt.Errorf("failed to decode inverted index: %w", err)
			}
			idx.InvertedIndex = loadedIndex
		} else {
			var discarded map[uint32][]core.Posting
			if err := decoder.Decode(&discarded); err != nil {
				return fmt.Errorf("failed to skip inverted index: %w", err)
			}
		}
	}

	return nil
}
