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

	// 1. Create a temporary file (safer than overwriting directly)
	f, err := os.Create(path + ".tmp") //nolint:gosec // path comes from operator config, not user input
	if err != nil {
		return err
	}

	// 2. Encode the entire map to the file using GOB (Go Binary)
	encoder := gob.NewEncoder(f)
	if err := encoder.Encode(idx.Store); err != nil {
		_ = f.Close() //nolint:errcheck // best-effort cleanup; encoding error takes precedence
		return err
	}

	// 3. Close before rename — required on some OSes and ensures flush
	if err := f.Close(); err != nil {
		return err
	}

	// 4. Rename temp file to actual file (Atomic operation)
	// This prevents corruption if the server crashes mid-write
	return os.Rename(path+".tmp", path)
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
