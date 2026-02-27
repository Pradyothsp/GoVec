package index

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/core"
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
// Sparse vector is stored in canonical format (uint32 indices + float32 values)
type WALEntry struct {
	Action WALAction         `json:"action"`
	ID     string            `json:"id"`
	Vector []float32         `json:"vec,omitempty"`
	Sparse core.SparseVector `json:"sparse,omitempty"`
	Meta   map[string]any    `json:"meta,omitempty"`
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
	log.Info().Str("path", filepath).Msg("WAL opened")
	return &WAL{file: f}, nil
}

// WriteEntry saves an operation to the disk immediately.
// ctx is accepted for API consistency and future use (e.g. deadline-aware writes).
func (w *WAL) WriteEntry(_ context.Context, entry *WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Encode to JSON
	bytes, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	// Add newline so we can read it line-by-line later
	if _, err = w.file.Write(append(bytes, '\n')); err != nil {
		return err
	}
	return w.file.Sync()
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
	if err == nil {
		log.Info().Msg("WAL cleared")
	}
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
	// Increase buffer to 4MB to handle large vector WAL entries (default 64KB is too small)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	count := 0
	lineNum := 0 // Track line number for logging

	for scanner.Scan() {
		lineNum++
		var entry WALEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			log.Warn().Int("line", lineNum).Err(err).Msg("skipping malformed WAL entry")
			continue
		}

		switch entry.Action {
		case WALActionInsert:
			idx.mu.Lock()

			err := idx.insertInternal(entry.ID, entry.Vector, entry.Sparse, entry.Meta)
			if err != nil {
				log.Warn().Str("id", entry.ID).Int("line", lineNum).Err(err).Msg("failed to replay insert")
				idx.mu.Unlock()
				continue
			}

			idx.mu.Unlock()

		case WALActionDelete:
			idx.mu.Lock()

			err := idx.deleteInternal(entry.ID)
			if err != nil {
				// Not found is OK during replay (idempotent)
				if errors.Is(err, core.ErrNotFound) {
					idx.mu.Unlock()
					continue
				}

				log.Warn().Str("id", entry.ID).Int("line", lineNum).Err(err).Msg("failed to replay delete")
				idx.mu.Unlock()
				continue
			}

			idx.mu.Unlock()
		}
		count++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("WAL scanner error at line %d: %w", lineNum, err)
	}

	log.Info().Int("count", count).Msg("WAL replay complete")

	return nil
}
