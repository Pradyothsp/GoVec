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
	Version             int    // File format version (currently 3: IDMapper + uint32 IDs)
	Quantization        string // "none" or "scalar"
	DistanceMetric      string // "cosine", etc.
	HybridSearchEnabled bool   // Whether inverted index is included
}

// SaveToFile serializes the index to a specific path
func (idx *VectorIndex[T]) SaveToFile(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Determine quantization type from T
	var quantType string
	switch any(*new(T)).(type) {
	case []float32:
		quantType = "none"
	case []int8:
		quantType = "scalar"
	default:
		return fmt.Errorf("unknown vector type")
	}

	// PHASE 1: Clear WAL first (fail fast if this fails)
	if idx.wal != nil {
		if err := idx.wal.Clear(); err != nil {
			return err
		}
	}

	// PHASE 2: Create temporary snapshot file
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path comes from operator config, not user input
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(f)

	// Write header first
	header := SnapshotHeader{
		Version:             3, // Bumped to v3 for IDMapper support
		Quantization:        quantType,
		DistanceMetric:      "cosine", // TODO: Make this configurable when we support multiple metrics
		HybridSearchEnabled: idx.InvertedIndex != nil,
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	// Write IDMapper state (v3+)
	if err := idx.IDMapper.EncodeGOB(encoder); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	// Write store (vector data with uint32 keys in v3)
	if err := encoder.Encode(idx.Store); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	// Write inverted index (if hybrid search is enabled)
	if idx.InvertedIndex != nil {
		if err := encoder.Encode(idx.InvertedIndex); err != nil {
			_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
			_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
			return err
		}
	}

	// Close before rename — required on some OSes and ensures flush
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	// PHASE 3: Atomic rename (commit)
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	return nil
}

// LoadFromFile reads the index from disk.
// A missing file is not an error — the server starts with an empty index.
func (idx *VectorIndex[T]) LoadFromFile(path string) (err error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path) //nolint:gosec // path comes from operator config, not user input
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

	// Read header
	var header SnapshotHeader
	if err := decoder.Decode(&header); err != nil {
		return fmt.Errorf("failed to decode snapshot header: %w", err)
	}

	// Validate header version (support v1, v2, and v3)
	if header.Version < 1 || header.Version > 3 {
		return fmt.Errorf("unsupported snapshot version: %d (expected 1-3)", header.Version)
	}

	// v1 and v2 used string IDs - incompatible with v3 (uint32 IDs)
	if header.Version < 3 {
		return fmt.Errorf("snapshot version %d uses string IDs (incompatible with v3 uint32 IDs). Delete data files and re-insert data", header.Version)
	}

	// Determine expected quantization type from T
	var expectedQuant string
	switch any(*new(T)).(type) {
	case []float32:
		expectedQuant = "none"
	case []int8:
		expectedQuant = "scalar"
	default:
		return fmt.Errorf("unknown vector type")
	}

	// Validate quantization matches
	if header.Quantization != expectedQuant {
		return fmt.Errorf("snapshot quantization mismatch: config expects '%s' but snapshot is '%s'. Delete data files or change config", expectedQuant, header.Quantization)
	}

	// Read IDMapper state (v3+)
	if header.Version >= 3 {
		if err := idx.IDMapper.DecodeGOB(decoder); err != nil {
			return fmt.Errorf("failed to decode IDMapper: %w", err)
		}
	}

	// Read store (vector data)
	if err := decoder.Decode(&idx.Store); err != nil {
		return fmt.Errorf("failed to decode store: %w", err)
	}

	// Read inverted index (if v2 and hybrid search was enabled)
	if header.Version == 2 && header.HybridSearchEnabled {
		// Only load if current index has inverted index enabled
		if idx.InvertedIndex != nil {
			var loadedIndex map[uint32][]core.Posting
			if err := decoder.Decode(&loadedIndex); err != nil {
				return fmt.Errorf("failed to decode inverted index: %w", err)
			}
			idx.InvertedIndex = loadedIndex
		} else {
			// Skip inverted index data if current config doesn't have hybrid search
			var discarded map[uint32][]core.Posting
			if err := decoder.Decode(&discarded); err != nil {
				return fmt.Errorf("failed to skip inverted index: %w", err)
			}
		}
	}

	return nil
}
