package index

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"math"
)

// The WAL is a sequence of binary records, each framed so replay can tell a
// complete record from a torn or corrupted one:
//
//	version (1 byte) | payload length (uint32) | CRC-32C of payload (uint32) | payload
//
// and the payload, little-endian, lengths as uvarints:
//
//	action (1 byte) | id | vector (count + float32s) | sparse indices (count + uint32s)
//	| sparse values (count + float32s) | metadata (length + JSON, 0 for none)
//
// It used to be one JSON object per line. JSON writes each float32 as about 12
// characters of decimal text, so a 1536-dimension entry took about 19 KB for
// 6 KB of vector (govec #15): 100k vectors meant a 1.9 GB WAL between
// snapshots, slow to write and slower to replay. Vectors are now raw bytes.
// Metadata stays JSON inside the record: it's arbitrary map[string]any, usually
// small, and JSON round-trips it exactly as the API does.
//
// The version byte also identifies a WAL in the old format: its first byte is
// '{', never a valid version.

const (
	walRecordVersion  = 1
	walFrameHeaderLen = 9 // version byte, payload length, CRC-32C
	// maxWALRecord bounds one record's payload, matching the largest request
	// the gRPC server accepts by default. A larger length can only be damage.
	maxWALRecord = 64 << 20
)

const (
	walActionCodeInsert byte = 1
	walActionCodeDelete byte = 2
)

var walCRCTable = crc32.MakeTable(crc32.Castagnoli)

var errLegacyWAL = errors.New("WAL is in the JSON format of GoVec 0.1.0, which this build no longer reads. Replay it with the version that wrote it, or delete it")

// appendWALRecord appends entry to dst as one framed record.
func appendWALRecord(dst []byte, entry *WALEntry) ([]byte, error) {
	start := len(dst)
	dst = append(dst, make([]byte, walFrameHeaderLen)...)

	dst, err := appendWALPayload(dst, entry)
	if err != nil {
		return nil, err
	}

	payload := dst[start+walFrameHeaderLen:]
	if len(payload) > maxWALRecord {
		return nil, fmt.Errorf("WAL entry %q is %d bytes, over the %d-byte limit", entry.ID, len(payload), maxWALRecord)
	}
	dst[start] = walRecordVersion
	binary.LittleEndian.PutUint32(dst[start+1:], uint32(len(payload))) //nolint:gosec // bounded by maxWALRecord above
	binary.LittleEndian.PutUint32(dst[start+5:], crc32.Checksum(payload, walCRCTable))
	return dst, nil
}

func appendWALPayload(dst []byte, entry *WALEntry) ([]byte, error) {
	switch entry.Action {
	case WALActionInsert:
		dst = append(dst, walActionCodeInsert)
	case WALActionDelete:
		dst = append(dst, walActionCodeDelete)
	default:
		return nil, fmt.Errorf("unknown WAL action %q", entry.Action)
	}

	dst = binary.AppendUvarint(dst, uint64(len(entry.ID)))
	dst = append(dst, entry.ID...)

	dst = appendFloat32s(dst, entry.Vector)

	dst = binary.AppendUvarint(dst, uint64(len(entry.Sparse.Indices)))
	for _, i := range entry.Sparse.Indices {
		dst = binary.LittleEndian.AppendUint32(dst, i)
	}
	dst = appendFloat32s(dst, entry.Sparse.Values)

	if len(entry.Meta) == 0 {
		return binary.AppendUvarint(dst, 0), nil
	}
	meta, err := json.Marshal(entry.Meta)
	if err != nil {
		return nil, fmt.Errorf("encode metadata of %q: %w", entry.ID, err)
	}
	dst = binary.AppendUvarint(dst, uint64(len(meta)))
	return append(dst, meta...), nil
}

func appendFloat32s(dst []byte, values []float32) []byte {
	dst = binary.AppendUvarint(dst, uint64(len(values)))
	for _, v := range values {
		dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(v))
	}
	return dst
}

// decodeWALPayload reverses appendWALPayload. Its input already passed the
// record's CRC, so an error here means a bug or a format mismatch, not a torn
// write.
func decodeWALPayload(payload []byte) (WALEntry, error) {
	r := walPayloadReader{buf: payload}
	var entry WALEntry

	switch code := r.byte(); code {
	case walActionCodeInsert:
		entry.Action = WALActionInsert
	case walActionCodeDelete:
		entry.Action = WALActionDelete
	default:
		if r.err == nil {
			r.err = fmt.Errorf("unknown WAL action code %d", code)
		}
	}

	entry.ID = string(r.bytes(r.uvarint()))
	entry.Vector = r.float32s()

	if n := r.uvarint(); n > 0 {
		raw := r.bytes(n * 4)
		if r.err == nil {
			entry.Sparse.Indices = make([]uint32, n)
			for i := range entry.Sparse.Indices {
				entry.Sparse.Indices[i] = binary.LittleEndian.Uint32(raw[i*4:])
			}
		}
	}
	entry.Sparse.Values = r.float32s()

	if meta := r.bytes(r.uvarint()); len(meta) > 0 && r.err == nil {
		if err := json.Unmarshal(meta, &entry.Meta); err != nil {
			r.err = fmt.Errorf("decode metadata: %w", err)
		}
	}

	if r.err == nil && len(r.buf) > 0 {
		r.err = fmt.Errorf("%d bytes left over after the entry", len(r.buf))
	}
	return entry, r.err
}

// walPayloadReader reads a payload front to back. The first failure sticks:
// later reads return zero values, so decodeWALPayload checks once at the end.
type walPayloadReader struct {
	buf []byte
	err error
}

func (r *walPayloadReader) fail(what string) {
	if r.err == nil {
		r.err = fmt.Errorf("WAL payload truncated reading %s", what)
	}
	r.buf = nil
}

func (r *walPayloadReader) byte() byte {
	if len(r.buf) < 1 {
		r.fail("action")
		return 0
	}
	b := r.buf[0]
	r.buf = r.buf[1:]
	return b
}

func (r *walPayloadReader) uvarint() int {
	v, n := binary.Uvarint(r.buf)
	if n <= 0 || v > maxWALRecord {
		r.fail("a length")
		return 0
	}
	r.buf = r.buf[n:]
	return int(v) //nolint:gosec // bounded by maxWALRecord above
}

func (r *walPayloadReader) bytes(n int) []byte {
	if n > len(r.buf) {
		r.fail("a field")
		return nil
	}
	b := r.buf[:n]
	r.buf = r.buf[n:]
	return b
}

func (r *walPayloadReader) float32s() []float32 {
	n := r.uvarint()
	if n == 0 {
		return nil
	}
	raw := r.bytes(n * 4)
	if r.err != nil {
		return nil
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values
}
