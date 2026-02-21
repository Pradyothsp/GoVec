package core

import (
	"bufio"
	"encoding/json"
	"log"
	"os"
	"sync"
)

// WALAction represents the type of operation in a WAL entry
type WALAction string

const (
	// WALActionInsert represents an insert operation
	WALActionInsert WALAction = "INSERT"
	// WALActionDelete represents a delete operation
	WALActionDelete WALAction = "DELETE"
)

// WALEntry represents a single write-ahead log entry
// Vector is always []float32 (canonical format from API), regardless of internal storage type
type WALEntry struct {
	Action WALAction      `json:"action"`
	ID     string         `json:"id"`
	Vector []float32      `json:"vec,omitempty"`
	Meta   map[string]any `json:"meta,omitempty"`
}

// WAL is a write-ahead log for durability
type WAL struct {
	mu   sync.Mutex
	file *os.File
}

// NewWAL creates a new WAL instance at the specified filepath
func NewWAL(filepath string) (*WAL, error) {
	// Open file in Append mode (create if not exists)
	f, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644) //nolint:gosec // WAL file permissions are intentionally 0o644
	if err != nil {
		return nil, err
	}
	return &WAL{file: f}, nil
}

// WriteEntry saves an operation to the disk immediately
func (w *WAL) WriteEntry(entry WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Encode to JSON
	bytes, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	// Add newline so we can read it line-by-line later
	_, err = w.file.Write(append(bytes, '\n'))
	return err
}

// Clear wipes the WAL (used after a successful Snapshot)
func (w *WAL) Clear() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Truncate file to 0 bytes
	if err := w.file.Truncate(0); err != nil {
		return err
	}

	// Reset file pointer to start (defensive - some platforms may not do this)
	_, err := w.file.Seek(0, 0)
	return err
}

// Close closes the WAL file handle
func (w *WAL) Close() error {
	return w.file.Close()
}

// ReplayWAL reads the WAL file and applies changes to the index.
// Uses locking to ensure thread-safety during concurrent operations.
func (idx *VectorIndex[T]) ReplayWAL(filepath string) error {
	f, err := os.Open(filepath) //nolint:gosec // filepath comes from config, not user input
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close() //nolint:errcheck // read-only operation, error on close is not critical

	scanner := bufio.NewScanner(f)
	count := 0
	lineNum := 0 // Track line number for logging

	for scanner.Scan() {
		lineNum++
		var entry WALEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			log.Printf("WARNING: Skipping malformed WAL entry at line %d: %v", lineNum, err)
			continue
		}

		switch entry.Action {
		case WALActionInsert:
			// Encode WAL vector ([]float32) to storage format (T)
			idx.mu.Lock()
			idx.Store[entry.ID] = &VectorNode[T]{
				ID:       entry.ID,
				Vector:   idx.encodeFunc(entry.Vector),
				Metadata: entry.Meta,
			}
			idx.mu.Unlock()
		case WALActionDelete:
			idx.mu.Lock()
			delete(idx.Store, entry.ID)
			idx.mu.Unlock()
		}
		count++
	}

	log.Printf("Replayed %d WAL entries", count)

	return nil
}
