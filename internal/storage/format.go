//go:build unix

package storage

// The on-disk format of the mmap store. These are not settings: changing any
// of them changes what an existing store's files mean, so a store written
// before the change would be misread after it. A new layout takes a new
// headerVersion.
//
// The store is a directory of chunk files, chunk_0.vec, chunk_1.vec, ..., each
// ChunkFileSize bytes. chunk_0 starts with a headerSize-byte header; later
// chunks hold vectors from byte 0. Slot N is the Nth fixed-size vector across
// them, so a vector's place follows from its slot number alone.
//
// The header, little-endian:
//
//	magic "GOVECMM0" (8 bytes) | version (uint32) | bytes per vector (uint32)
//	| quantization (uint8: 0 float32, 1 int8) | ... | slot count (uint64, at 20)
//	| chunk size (uint64, at 28; 0 in stores written before it was recorded)

const (
	headerSize  = 4096       // bytes reserved at the start of every chunk for the header
	headerMagic = "GOVECMM0" // 8-byte magic string identifying the file format

	headerOffMagic    = 0  // [8]byte
	headerOffVersion  = 8  // uint32
	headerOffElemSize = 12 // uint32
	headerOffQuantTag = 16 // uint8  (0=float32, 1=int8)
	headerOffCount    = 20 // uint64 (total vectors ever Put; only in chunk_0)
	headerOffChunk    = 28 // uint64 (ChunkFileSize; only in chunk_0)

	headerVersion = uint32(1)
	quantFloat32  = uint8(0)
	quantInt8     = uint8(1)
)

// ChunkFileSize is the size in bytes pre-allocated per chunk file. Slots are
// addressed from it, so it is part of the format too: the header records it,
// and a store written with a different size won't open. It is a variable only
// so tests can use smaller chunks.
var ChunkFileSize = int64(1 << 30) // 1 GiB

// legacyChunkFileSize is the chunk size of a store whose header predates the
// chunk-size field (it reads 0 there). 1 GiB is the only size ever shipped.
const legacyChunkFileSize = 1 << 30
