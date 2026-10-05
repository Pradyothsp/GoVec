package index

import (
	"bufio"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/storage"
)

// A snapshot is one gob stream, written and read as a sequence of small values:
//
//	header | ID mapper | node count | one record per node | [HNSW] graph topology chunks
//
// It used to hold each big section as a single gob value -- every node in one
// map, the whole graph in one []byte. gob builds a value completely in memory
// before writing any of it, so every save needed a second full copy of the
// index, and HNSW a third (Graph.Export wrote each vector again, into a
// bytes.Buffer). At 100k x 1536 dimensions that copy didn't fit beside the live
// index, and govec-bench's 2 GB container was OOM-killed 4.2 MB into the file.
// Now a save holds one record at a time, and vectors are written once: the
// HNSW graph is stored as topology only and gets its vectors from the records
// on load.
//
// A record stores its vector as raw little-endian bytes, not as a gob slice:
// gob writes each float32 as a byte-reversed float64 varint, about 8-9 bytes
// for a real-valued embedding instead of 4, which made snapshots 1.5x the
// vectors. Under mmap a record carries no vector; the vectors are already in
// the mmap store, which the save syncs first.

// snapshotChunkSize bounds how much of the graph topology is held in memory
// while it streams into the snapshot.
const snapshotChunkSize = 1 << 20

// snapshotWriter writes a snapshot into a temporary file next to its final
// path, buffered, and either publishes it (publishSnapshot) or removes it.
type snapshotWriter struct {
	f       *os.File
	buf     *bufio.Writer
	enc     *gob.Encoder
	tmpPath string
	path    string
}

func newSnapshotWriter(path string) (*snapshotWriter, error) {
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path from operator config
	if err != nil {
		return nil, err
	}

	buf := bufio.NewWriterSize(f, snapshotChunkSize)
	return &snapshotWriter{f: f, buf: buf, enc: gob.NewEncoder(buf), tmpPath: tmpPath, path: path}, nil
}

// abort discards a snapshot that failed part-way. The live snapshot and the
// WAL are untouched.
func (w *snapshotWriter) abort() {
	_ = w.f.Close()          //nolint:errcheck // best-effort cleanup; the write error takes precedence
	_ = os.Remove(w.tmpPath) //nolint:errcheck // best-effort cleanup; the write error takes precedence
}

// publish flushes the buffer and makes the snapshot the live one; see
// publishSnapshot for the ordering that keeps the WAL until then.
func (w *snapshotWriter) publish(wal *WAL) error {
	if err := w.buf.Flush(); err != nil {
		w.abort()
		return err
	}
	return publishSnapshot(w.f, w.tmpPath, w.path, wal)
}

// nodeRecord is one node as a snapshot stores it.
type nodeRecord struct {
	InternalID uint32
	ExternalID string
	Vector     []byte // appendVectorBytes; empty under mmap
	Sparse     core.SparseVector
	Metadata   map[string]any
}

// writeNodeRecords writes the node count, then one record per node. Under
// mmap (vectorsInline false) a record leaves the vector out.
func writeNodeRecords[T any](enc *gob.Encoder, nodes map[uint32]*core.VectorNode[T], vectorsInline bool) error {
	if err := enc.Encode(len(nodes)); err != nil {
		return fmt.Errorf("encode node count: %w", err)
	}

	// Reused for every record: Encode copies it into gob's buffer.
	var vecBuf []byte
	for _, node := range nodes {
		record := nodeRecord{
			InternalID: node.InternalID,
			ExternalID: node.ExternalID,
			Sparse:     node.Sparse,
			Metadata:   node.Metadata,
		}
		if vectorsInline {
			vecBuf = appendVectorBytes(vecBuf[:0], node.Vector)
			record.Vector = vecBuf
		}
		if err := enc.Encode(record); err != nil {
			return fmt.Errorf("encode node %q: %w", node.ExternalID, err)
		}
	}
	return nil
}

// readNodeRecords reads what writeNodeRecords wrote into nodes. With an mmap
// store, each record's vector comes from the store instead of the stream.
func readNodeRecords[T any](dec *gob.Decoder, nodes map[uint32]*core.VectorNode[T], mmapStore *storage.MmapStore) error {
	var count int
	if err := dec.Decode(&count); err != nil {
		return fmt.Errorf("decode node count: %w", err)
	}

	for i := range count {
		// A fresh record each time: gob leaves fields absent from the stream
		// (zero values) as they were, so a reused one would carry the previous
		// node's metadata into the next.
		var record nodeRecord
		if err := dec.Decode(&record); err != nil {
			return fmt.Errorf("decode node %d of %d: %w", i+1, count, err)
		}

		var vec T
		if mmapStore != nil {
			var ok bool
			vec, ok = getFromMmapStore[T](mmapStore, record.InternalID)
			if !ok {
				return fmt.Errorf("mmap slot missing for internalID %d (externalID %q)", record.InternalID, record.ExternalID)
			}
		} else {
			var err error
			vec, err = vectorFromBytes[T](record.Vector)
			if err != nil {
				return fmt.Errorf("decode node %q: %w", record.ExternalID, err)
			}
		}

		nodes[record.InternalID] = &core.VectorNode[T]{
			InternalID: record.InternalID,
			ExternalID: record.ExternalID,
			Vector:     vec,
			Sparse:     record.Sparse,
			Metadata:   record.Metadata,
		}
	}
	return nil
}

// appendVectorBytes appends vec to dst as little-endian float32s, or as one
// byte per int8.
func appendVectorBytes[T any](dst []byte, vec T) []byte {
	switch v := any(vec).(type) {
	case []float32:
		for _, f := range v {
			dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(f))
		}
	case []int8:
		for _, b := range v {
			dst = append(dst, byte(b))
		}
	}
	return dst
}

// vectorFromBytes reverses appendVectorBytes into a slice of exactly the
// vector's length.
func vectorFromBytes[T any](b []byte) (T, error) {
	var zero T
	switch any(zero).(type) {
	case []float32:
		if len(b)%4 != 0 {
			return zero, fmt.Errorf("float32 vector bytes not a multiple of 4: %d", len(b))
		}
		vec := make([]float32, len(b)/4)
		for i := range vec {
			vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return asVector[T](vec)
	case []int8:
		vec := make([]int8, len(b))
		for i, x := range b {
			vec[i] = int8(x)
		}
		return asVector[T](vec)
	default:
		return zero, fmt.Errorf("unsupported vector type %T", zero)
	}
}

func asVector[T any](vec any) (T, error) {
	out, ok := vec.(T)
	if !ok {
		var zero T
		return zero, fmt.Errorf("vector type %T is not %T", vec, zero)
	}
	return out, nil
}

// gobChunkWriter carries a byte stream inside a gob stream: what's written to
// it goes out as gob-encoded []byte chunks of up to snapshotChunkSize, and
// Close writes an empty chunk to end it. Raw bytes can't be interleaved with
// gob values directly -- the decoder reads ahead of the value it returns.
type gobChunkWriter struct {
	enc *gob.Encoder
	buf []byte
}

func newGobChunkWriter(enc *gob.Encoder) *gobChunkWriter {
	return &gobChunkWriter{enc: enc, buf: make([]byte, 0, snapshotChunkSize)}
}

func (w *gobChunkWriter) Write(p []byte) (int, error) {
	written := 0
	for len(p) > 0 {
		n := min(len(p), snapshotChunkSize-len(w.buf))
		w.buf = append(w.buf, p[:n]...)
		p = p[n:]
		written += n

		if len(w.buf) == snapshotChunkSize {
			if err := w.flush(); err != nil {
				return written, err
			}
		}
	}
	return written, nil
}

func (w *gobChunkWriter) flush() error {
	if len(w.buf) == 0 {
		return nil
	}
	if err := w.enc.Encode(w.buf); err != nil {
		return err
	}
	w.buf = w.buf[:0]
	return nil
}

// Close writes any buffered bytes and the empty end-of-stream chunk.
func (w *gobChunkWriter) Close() error {
	if err := w.flush(); err != nil {
		return err
	}
	return w.enc.Encode([]byte{})
}

// gobChunkReader reads back what a gobChunkWriter wrote, returning io.EOF at
// the empty end-of-stream chunk.
type gobChunkReader struct {
	dec  *gob.Decoder
	buf  []byte
	done bool
}

func newGobChunkReader(dec *gob.Decoder) *gobChunkReader {
	return &gobChunkReader{dec: dec}
}

func (r *gobChunkReader) Read(p []byte) (int, error) {
	for len(r.buf) == 0 {
		if r.done {
			return 0, io.EOF
		}
		if err := r.dec.Decode(&r.buf); err != nil {
			return 0, fmt.Errorf("decode topology chunk: %w", err)
		}
		if len(r.buf) == 0 {
			r.done = true
		}
	}

	n := copy(p, r.buf)
	r.buf = r.buf[n:]
	return n, nil
}

// checkSnapshotVersion rejects a snapshot in any other format than the one
// this build writes. There is no migration path: the format changed before
// anyone depended on it, and decoding an old file would only fail later with
// a less useful gob error.
func checkSnapshotVersion(header SnapshotHeader) error {
	if header.Version != snapshotVersion {
		return fmt.Errorf("snapshot format version %d, but this build reads version %d. Delete data files, or restore them with the GoVec version that wrote them", header.Version, snapshotVersion)
	}
	return nil
}
