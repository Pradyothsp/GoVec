package index

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
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

// WALEntry is one write-ahead log entry, encoded as a binary record by
// appendWALRecord. Vector is always []float32 (the API's canonical format),
// whatever the index stores internally; Sparse is in canonical format too
// (uint32 indices + float32 values).
type WALEntry struct {
	Action WALAction
	ID     string
	Vector []float32
	Sparse core.SparseVector
	Meta   map[string]any
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

	record, err := appendWALRecord(nil, entry)
	if err != nil {
		return err
	}

	if _, err = w.file.Write(record); err != nil {
		return err
	}
	return w.file.Sync()
}

// WriteEntries saves multiple operations to disk as a single durable unit: one
// buffered write plus one fsync for the whole batch, instead of one fsync per
// entry. All entries are encoded into memory first, so an encoding failure on
// any entry aborts before anything touches the file — the WAL is left exactly
// as it was, never containing a partial batch.
// ctx is accepted for API consistency and future use (e.g. deadline-aware writes).
func (w *WAL) WriteEntries(_ context.Context, entries []*WALEntry) error {
	if len(entries) == 0 {
		return nil
	}

	var buf []byte
	for _, entry := range entries {
		var err error
		if buf, err = appendWALRecord(buf, entry); err != nil {
			return err
		}
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if _, err := w.file.Write(buf); err != nil {
		return err
	}
	return w.file.Sync()
}

// Clear wipes the WAL, once a snapshot or reset has made its entries redundant.
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

// replayWALFile reads the WAL at walPath and calls apply for each entry, in
// order. Recovery keeps the longest valid prefix of the log, as Redis and etcd
// do:
//
//   - A missing file is an empty WAL.
//   - A bad record -- incomplete, failing its CRC, zero-filled, or garbage --
//     with no valid record anywhere after it is a write torn by a crash. It
//     was never acknowledged, so it is cut off the file; otherwise the next
//     append would follow it and be lost on the following replay. Garbage
//     counts too: some filesystems leave stale bytes at the end of a file
//     after a crash, and refusing to start over those would be worse.
//   - A bad record with a valid one after it is corruption, as is an entry
//     apply rejects. Replay stops with an error naming the record, and the
//     file is left as found.
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

	info, err := f.Stat()
	if err != nil {
		return err
	}
	size := info.Size()
	r := bufio.NewReaderSize(f, 1<<20)

	if size > 0 {
		first, err := r.Peek(1)
		if err != nil {
			return fmt.Errorf("WAL read error: %w", err)
		}
		if first[0] == '{' {
			return errLegacyWAL
		}
	}

	var (
		offset int64
		count  int
		header = make([]byte, walFrameHeaderLen)
		buf    []byte
	)
	for record := 1; offset < size; record++ {
		// badRecord handles a record at offset that can't be read: a torn tail
		// if no valid record follows it, corruption otherwise. Its header can't
		// be trusted, so neither can its length: look for a valid record at
		// every byte after its start. Only this failure path reads the rest of
		// the file into memory.
		badRecord := func(reason string) error {
			rest, err := readFrom(walPath, offset)
			if err != nil {
				return fmt.Errorf("WAL read error at record %d: %w", record, err)
			}
			if containsValidRecord(rest[1:]) {
				return fmt.Errorf("WAL is corrupt at record %d (byte %d), before its end: %s", record, offset, reason)
			}
			// Nothing valid at all, from the first byte: only cut it if it looks
			// like a WAL whose first write was torn (it starts like a record, or
			// it's zero-filled). Anything else isn't a GoVec WAL -- wal_path may
			// name the wrong file -- and must not be truncated.
			if offset == 0 && rest[0] != walRecordVersion && !isAllZero(rest) {
				return fmt.Errorf("file doesn't look like a GoVec WAL (first byte 0x%02x); left as found", rest[0])
			}
			if err := os.Truncate(walPath, offset); err != nil {
				return fmt.Errorf("cut torn last record %d off the WAL: %w", record, err)
			}
			log.Warn().Int("record", record).Int64("bytes", size-offset).Str("reason", reason).
				Msg("cut a torn last record off the WAL (a write interrupted by a crash)")
			return nil
		}

		if _, err := io.ReadFull(r, header); err != nil {
			return badRecord("incomplete record header")
		}
		version := header[0]
		length := int64(binary.LittleEndian.Uint32(header[1:5]))
		checksum := binary.LittleEndian.Uint32(header[5:9])
		end := offset + walFrameHeaderLen + length
		if version != walRecordVersion || length == 0 || length > maxWALRecord || end > size {
			return badRecord(fmt.Sprintf("bad record header (version %d, length %d)", version, length))
		}

		if int64(cap(buf)) < length {
			buf = make([]byte, length)
		}
		payload := buf[:length]
		if _, err := io.ReadFull(r, payload); err != nil {
			return fmt.Errorf("WAL read error at record %d: %w", record, err)
		}
		if crc32.Checksum(payload, walCRCTable) != checksum {
			return badRecord("CRC mismatch")
		}

		entry, err := decodeWALPayload(payload)
		if err != nil {
			return fmt.Errorf("WAL record %d: %w", record, err)
		}
		if err := apply(entry); err != nil {
			return fmt.Errorf("WAL record %d: replay %s %q: %w", record, entry.Action, entry.ID, err)
		}

		offset = end
		count++
	}

	log.Info().Int("count", count).Msg("WAL replay complete")
	return nil
}

// readFrom returns the file's bytes from offset to its end.
func readFrom(path string, offset int64) ([]byte, error) {
	f, err := os.Open(path) //nolint:gosec // path comes from config, not user input
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only operation, error on close is not critical

	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}
	return io.ReadAll(f)
}

// isAllZero reports whether b holds only zero bytes: the tail a filesystem
// can leave when a crash extends a file without writing its contents.
func isAllZero(b []byte) bool {
	for _, x := range b {
		if x != 0 {
			return false
		}
	}
	return true
}

// containsValidRecord reports whether a complete record with a matching CRC
// starts at any byte of b.
func containsValidRecord(b []byte) bool {
	for i := 0; i+walFrameHeaderLen <= len(b); i++ {
		if b[i] != walRecordVersion {
			continue
		}
		length := int(binary.LittleEndian.Uint32(b[i+1:]))
		end := i + walFrameHeaderLen + length
		if length == 0 || length > maxWALRecord || end > len(b) {
			continue
		}
		if crc32.Checksum(b[i+walFrameHeaderLen:end], walCRCTable) == binary.LittleEndian.Uint32(b[i+5:]) {
			return true
		}
	}
	return false
}

// ignoreNotFound treats deleting a missing ID as success, which keeps WAL
// replay idempotent.
func ignoreNotFound(err error) error {
	if errors.Is(err, core.ErrNotFound) {
		return nil
	}
	return err
}
