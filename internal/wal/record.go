package wal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"math"
)

// Encoding and decoding of one record; the layout is in format.go.

// appendRecord appends entry to dst as one framed record.
func appendRecord(dst []byte, entry *Entry) ([]byte, error) {
	start := len(dst)
	dst = append(dst, make([]byte, frameHeaderLen)...)

	dst, err := appendPayload(dst, entry)
	if err != nil {
		return nil, err
	}

	payload := dst[start+frameHeaderLen:]
	if len(payload) > maxRecord {
		return nil, fmt.Errorf("WAL entry %q is %d bytes, over the %d-byte limit", entry.ID, len(payload), maxRecord)
	}
	dst[start] = recordVersion
	binary.LittleEndian.PutUint32(dst[start+1:], uint32(len(payload))) //nolint:gosec // bounded by maxRecord above
	binary.LittleEndian.PutUint32(dst[start+5:], crc32.Checksum(payload, crcTable))
	return dst, nil
}

func appendPayload(dst []byte, entry *Entry) ([]byte, error) {
	switch entry.Action {
	case ActionInsert:
		dst = append(dst, actionCodeInsert)
	case ActionDelete:
		dst = append(dst, actionCodeDelete)
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

// decodePayload reverses appendPayload. Its input already passed the
// record's CRC, so an error here means a bug or a format mismatch, not a torn
// write.
func decodePayload(payload []byte) (Entry, error) {
	r := payloadReader{buf: payload}
	var entry Entry

	switch code := r.byte(); code {
	case actionCodeInsert:
		entry.Action = ActionInsert
	case actionCodeDelete:
		entry.Action = ActionDelete
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

// payloadReader reads a payload front to back. The first failure sticks:
// later reads return zero values, so decodePayload checks once at the end.
type payloadReader struct {
	buf []byte
	err error
}

func (r *payloadReader) fail(what string) {
	if r.err == nil {
		r.err = fmt.Errorf("WAL payload truncated reading %s", what)
	}
	r.buf = nil
}

func (r *payloadReader) byte() byte {
	if len(r.buf) < 1 {
		r.fail("action")
		return 0
	}
	b := r.buf[0]
	r.buf = r.buf[1:]
	return b
}

func (r *payloadReader) uvarint() int {
	v, n := binary.Uvarint(r.buf)
	if n <= 0 || v > maxRecord {
		r.fail("a length")
		return 0
	}
	r.buf = r.buf[n:]
	return int(v) //nolint:gosec // bounded by maxRecord above
}

func (r *payloadReader) bytes(n int) []byte {
	if n > len(r.buf) {
		r.fail("a field")
		return nil
	}
	b := r.buf[:n]
	r.buf = r.buf[n:]
	return b
}

func (r *payloadReader) float32s() []float32 {
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
