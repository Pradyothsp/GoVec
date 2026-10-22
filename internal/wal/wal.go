// Package wal is GoVec's write-ahead log: an append-only file of binary
// records, each fsynced before the write it describes is applied, and replayed
// in order on startup. It knows nothing about engines: what an entry means is
// up to the caller of Replay.
package wal

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/core"
)

// Action is the kind of operation an entry records.
type Action string

const (
	// ActionInsert records an insert or an overwrite.
	ActionInsert Action = "INSERT"
	// ActionDelete records a delete.
	ActionDelete Action = "DELETE"
)

// Entry is one write-ahead log entry, encoded as a binary record by
// appendRecord. Vector is always []float32 (the API's canonical format),
// whatever the index stores internally; Sparse is in canonical format too
// (uint32 indices + float32 values).
type Entry struct {
	Action Action
	ID     string
	Vector []float32
	Sparse core.SparseVector
	Meta   map[string]any
}

// Log is an open write-ahead log file.
type Log struct {
	mu   sync.Mutex
	file *os.File
}

// Open opens the log at path for appending, creating it and its directory if
// needed.
func Open(path string) (*Log, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil { //nolint:gosec // 0o750 satisfies G301
		return nil, fmt.Errorf("create WAL directory: %w", err)
	}

	// Open file in Append mode (create if not exists)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644) //nolint:gosec // WAL file permissions are intentionally 0o644
	if err != nil {
		return nil, err
	}
	log.Info().Str("path", path).Msg("WAL opened")
	return &Log{file: f}, nil
}

// Path returns the file the log appends to.
func (l *Log) Path() string {
	return l.file.Name()
}

// WriteEntry saves an operation to the disk immediately.
// ctx is accepted for API consistency and future use (e.g. deadline-aware writes).
func (l *Log) WriteEntry(_ context.Context, entry *Entry) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	record, err := appendRecord(nil, entry)
	if err != nil {
		return err
	}

	if _, err = l.file.Write(record); err != nil {
		return err
	}
	return l.file.Sync()
}

// WriteEntries saves multiple operations to disk as a single durable unit: one
// buffered write plus one fsync for the whole batch, instead of one fsync per
// entry. All entries are encoded into memory first, so an encoding failure on
// any entry aborts before anything touches the file — the log is left exactly
// as it was, never containing a partial batch.
// ctx is accepted for API consistency and future use (e.g. deadline-aware writes).
func (l *Log) WriteEntries(_ context.Context, entries []*Entry) error {
	if len(entries) == 0 {
		return nil
	}

	var buf []byte
	for _, entry := range entries {
		var err error
		if buf, err = appendRecord(buf, entry); err != nil {
			return err
		}
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if _, err := l.file.Write(buf); err != nil {
		return err
	}
	return l.file.Sync()
}

// Clear wipes the log, once a snapshot or reset has made its entries redundant.
func (l *Log) Clear() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Truncate file to 0 bytes
	if err := l.file.Truncate(0); err != nil {
		return err
	}

	// Reset file pointer to start (defensive - some platforms may not do this)
	_, err := l.file.Seek(0, 0)
	if err == nil {
		log.Info().Msg("WAL cleared")
	}
	return err
}

// Close closes the log file.
func (l *Log) Close() error {
	return l.file.Close()
}

// Replay reads the log at path and calls apply for each entry, in order.
// Recovery keeps the longest valid prefix of the log, as Redis and etcd do:
//
//   - A missing file is an empty log.
//   - A bad record -- incomplete, failing its CRC, zero-filled, or garbage --
//     with no valid record anywhere after it is a write torn by a crash. It
//     was never acknowledged, so it is cut off the file; otherwise the next
//     append would follow it and be lost on the following replay. Garbage
//     counts too: some filesystems leave stale bytes at the end of a file
//     after a crash, and refusing to start over those would be worse.
//   - A bad record with a valid one after it is corruption, as is an entry
//     apply rejects. Replay stops with an error naming the record, and the
//     file is left as found.
func Replay(path string, apply func(Entry) error) error {
	f, err := os.Open(path) //nolint:gosec // path comes from config, not user input
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
			return errLegacyFormat
		}
	}

	var (
		offset int64
		count  int
		header = make([]byte, frameHeaderLen)
		buf    []byte
	)
	for record := 1; offset < size; record++ {
		// badRecord handles a record at offset that can't be read: a torn tail
		// if no valid record follows it, corruption otherwise. Its header can't
		// be trusted, so neither can its length: look for a valid record at
		// every byte after its start. Only this failure path reads the rest of
		// the file into memory.
		badRecord := func(reason string) error {
			rest, err := readFrom(path, offset)
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
			if offset == 0 && rest[0] != recordVersion && !isAllZero(rest) {
				return fmt.Errorf("file doesn't look like a GoVec WAL (first byte 0x%02x); left as found", rest[0])
			}
			if err := os.Truncate(path, offset); err != nil {
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
		end := offset + frameHeaderLen + length
		if version != recordVersion || length == 0 || length > maxRecord || end > size {
			return badRecord(fmt.Sprintf("bad record header (version %d, length %d)", version, length))
		}

		if int64(cap(buf)) < length {
			buf = make([]byte, length)
		}
		payload := buf[:length]
		if _, err := io.ReadFull(r, payload); err != nil {
			return fmt.Errorf("WAL read error at record %d: %w", record, err)
		}
		if crc32.Checksum(payload, crcTable) != checksum {
			return badRecord("CRC mismatch")
		}

		entry, err := decodePayload(payload)
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
	for i := 0; i+frameHeaderLen <= len(b); i++ {
		if b[i] != recordVersion {
			continue
		}
		length := int(binary.LittleEndian.Uint32(b[i+1:]))
		end := i + frameHeaderLen + length
		if length == 0 || length > maxRecord || end > len(b) {
			continue
		}
		if crc32.Checksum(b[i+frameHeaderLen:end], crcTable) == binary.LittleEndian.Uint32(b[i+5:]) {
			return true
		}
	}
	return false
}
