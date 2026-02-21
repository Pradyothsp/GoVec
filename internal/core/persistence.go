package core

import (
	"encoding/gob"
	"fmt"
	"os"
)

func init() {
	// Register types for GOB encoding/decoding of metadata
	gob.Register(map[string]any{})
	gob.Register([]any{})
}

// SnapshotHeader contains metadata about the snapshot file format
type SnapshotHeader struct {
	Version        int    // File format version (currently 1)
	Quantization   string // "none" or "scalar"
	DistanceMetric string // "cosine", etc.
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
		Version:        1,
		Quantization:   quantType,
		DistanceMetric: "cosine", // TODO: Make this configurable when we support multiple metrics
	}
	if err := encoder.Encode(header); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
	}

	// Write data
	if err := encoder.Encode(idx.Store); err != nil {
		_ = f.Close()          //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		_ = os.Remove(tmpPath) //nolint:errcheck // best-effort cleanup
		return err
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

	// Validate header version
	if header.Version != 1 {
		return fmt.Errorf("unsupported snapshot version: %d", header.Version)
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

	// Read data
	return decoder.Decode(&idx.Store)
}
