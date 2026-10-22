package wal

import (
	"errors"
	"hash/crc32"
)

// The on-disk format. These are not settings: changing any of them changes
// what a log file means, so a log written before the change can't be replayed
// after it. A new format takes a new recordVersion.
//
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
// 6 KB of vector: 100k vectors meant a 1.9 GB WAL between
// snapshots, slow to write and slower to replay. Vectors are now raw bytes.
// Metadata stays JSON inside the record: it's arbitrary map[string]any, usually
// small, and JSON round-trips it exactly as the API does.
//
// The version byte also identifies a WAL in the old format: its first byte is
// '{', never a valid version.

const (
	recordVersion  = 1
	frameHeaderLen = 9 // version byte, payload length, CRC-32C
	// maxRecord bounds one record's payload, matching the largest request
	// the gRPC server accepts by default. A larger length can only be damage.
	maxRecord = 64 << 20
)

const (
	actionCodeInsert byte = 1
	actionCodeDelete byte = 2
)

var crcTable = crc32.MakeTable(crc32.Castagnoli)

var errLegacyFormat = errors.New("WAL is in the JSON format of GoVec 0.1.0, which this build no longer reads. Replay it with the version that wrote it, or delete it")
