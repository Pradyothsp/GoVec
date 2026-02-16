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
func (idx *VectorIndex) SaveToFile(filepath string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// 1. Create a temporary file (safer than overwriting directly)
	f, err := os.Create(filepath + ".tmp")
	if err != nil {
		return err
	}
	defer f.Close()

	// 2. Encode the entire map to the file using GOB (Go Binary)
	encoder := gob.NewEncoder(f)
	if err := encoder.Encode(idx.Store); err != nil {
		return err
	}

	// 3. Rename temp file to actual file (Atomic operation)
	// This prevents corruption if the server crashes mid-write
	return os.Rename(filepath+".tmp", filepath)
}

// LoadFromFile reads the index from disk
// This usually runs ONLY at startup, so no locking needed (yet).
func (idx *VectorIndex) LoadFromFile(filepath string) error {
	f, err := os.Open(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			// File doesn't exist? That's fine, start with empty index.
			return nil
		}
		return err
	}
	defer f.Close()

	decoder := gob.NewDecoder(f)
	// We decode directly into the Store map
	return decoder.Decode(&idx.Store)
}
