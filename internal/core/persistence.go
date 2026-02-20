package core

import (
	"encoding/gob"
	"os"
)

func init() {
	// Register types for GOB encoding/decoding of metadata
	gob.Register(map[string]any{})
	gob.Register([]any{})
}

// SaveToFile serializes the index to a specific path
func (idx *VectorIndex) SaveToFile(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

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

	// Encode the entire map to the file using GOB (Go Binary)
	encoder := gob.NewEncoder(f)
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
func (idx *VectorIndex) LoadFromFile(path string) (err error) {
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
	return decoder.Decode(&idx.Store)
}
