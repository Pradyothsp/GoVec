package index

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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

// NewWAL creates a new WAL instance at the specified path, creating its
// directory if needed.
func NewWAL(walPath string) (*WAL, error) {
	if err := os.MkdirAll(filepath.Dir(walPath), 0o750); err != nil { //nolint:gosec // 0o750 satisfies G301
		return nil, fmt.Errorf("create WAL directory: %w", err)
	}

	// Open file in Append mode (create if not exists)
	f, err := os.OpenFile(walPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644) //nolint:gosec // WAL file permissions are intentionally 0o644
	if err != nil {
		return nil, err
	}
	log.Info().Str("path", walPath).Msg("WAL opened")
	return &WAL{file: f}, nil
}

// WriteEntry saves an operation to the disk immediately.
// ctx is accepted for API consistency and future use (e.g. deadline-aware writes).
func (w *WAL) WriteEntry(_ context.Context, entry *WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Encode to JSON
	encoded, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	// Add newline so we can read it line-by-line later
	if _, err = w.file.Write(append(encoded, '\n')); err != nil {
		return err
	}
	return w.file.Sync()
}

// WriteEntries saves multiple operations to disk as a single durable unit: one
// buffered write plus one fsync for the whole batch, instead of one fsync per
// entry. All entries are marshaled into memory first, so a marshal failure on
// any entry aborts before anything touches the file — the WAL is left exactly
// as it was, never containing a partial batch.
// ctx is accepted for API consistency and future use (e.g. deadline-aware writes).
func (w *WAL) WriteEntries(_ context.Context, entries []*WALEntry) error {
	if len(entries) == 0 {
		return nil
	}

	var buf bytes.Buffer
	for _, entry := range entries {
		b, err := json.Marshal(entry)
		if err != nil {
			return err
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if _, err := w.file.Write(buf.Bytes()); err != nil {
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

// ReplayWAL reads the WAL file and applies changes to the index. See
// replayWALFile for what counts as a torn write and what stops recovery.
func (idx *VectorIndex[T]) ReplayWAL(walPath string) error {
	return replayWALFile(walPath, func(entry WALEntry) error {
		idx.mu.Lock()
		defer idx.mu.Unlock()
		switch entry.Action {
		case WALActionInsert:
			return idx.insertInternal(entry.ID, entry.Vector, entry.Sparse, entry.Meta)
		case WALActionDelete:
			return ignoreNotFound(idx.deleteInternal(entry.ID))
		default:
			return fmt.Errorf("unknown WAL action %q", entry.Action)
		}
	})
}

// maxWALLine is the longest WAL line replay will read: 4MB, well above any
// realistic vector entry (the default 64KB is too small).
const maxWALLine = 4 * 1024 * 1024

// replayWALFile reads the WAL at walPath and calls apply for each entry, in
// order. Recovery keeps the longest valid prefix of the log, as Redis and etcd
// do:
//
//   - A missing file is an empty WAL.
//   - A last line that is unterminated or isn't valid JSON is a write torn by a
//     crash. It was never acknowledged, so it is cut off the file -- otherwise
//     the next append would merge into it and be lost on the following replay.
//   - A bad line before the end, or an entry apply rejects, is corruption.
//     Replay stops with an error naming the line, and the file is left as found.
//
// A delete of an ID that doesn't exist is not an error: replay is idempotent.
func replayWALFile(walPath string, apply func(WALEntry) error) error {
	f, err := os.Open(walPath) //nolint:gosec // walPath comes from config, not user input
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close() //nolint:errcheck // read-only operation, error on close is not critical

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, maxWALLine), maxWALLine)
	scanner.Split(scanLinesKeepingNewline)

	type badLine struct {
		num    int
		offset int64 // where the line starts, so a torn tail can be cut off
		err    error
	}
	var (
		bad     *badLine
		offset  int64
		lineNum int
		count   int
	)
	for scanner.Scan() {
		line := scanner.Bytes()
		lineNum++
		// A bad line is only a torn write if nothing follows it.
		if bad != nil {
			return fmt.Errorf("WAL is corrupt at line %d, before its last line: %w", bad.num, bad.err)
		}

		lineStart := offset
		offset += int64(len(line))
		body, terminated := bytes.CutSuffix(line, []byte("\n"))
		if !terminated {
			bad = &badLine{num: lineNum, offset: lineStart, err: errors.New("unterminated line")}
			continue
		}
		var entry WALEntry
		if err := json.Unmarshal(body, &entry); err != nil {
			bad = &badLine{num: lineNum, offset: lineStart, err: err}
			continue
		}
		if err := apply(entry); err != nil {
			return fmt.Errorf("WAL line %d: replay %s %q: %w", lineNum, entry.Action, entry.ID, err)
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("WAL read error after line %d: %w", lineNum, err)
	}

	if bad != nil {
		if err := os.Truncate(walPath, bad.offset); err != nil {
			return fmt.Errorf("cut torn last line %d off the WAL: %w", bad.num, err)
		}
		log.Warn().Int("line", bad.num).Int64("bytes", offset-bad.offset).Err(bad.err).
			Msg("cut a torn last line off the WAL (a write interrupted by a crash)")
	}

	log.Info().Int("count", count).Msg("WAL replay complete")
	return nil
}

// scanLinesKeepingNewline is bufio.ScanLines, except each token keeps its
// trailing newline, so the caller can tell a complete last line from a torn
// one and count bytes exactly.
func scanLinesKeepingNewline(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		return i + 1, data[:i+1], nil
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

// ignoreNotFound treats deleting a missing ID as success, which keeps WAL
// replay idempotent.
func ignoreNotFound(err error) error {
	if errors.Is(err, core.ErrNotFound) {
		return nil
	}
	return err
}
