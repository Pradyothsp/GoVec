//go:build unix

package storage

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

const (
	headerSize  = 4096       // bytes reserved at the start of every chunk for the header
	headerMagic = "GOVECMM0" // 8-byte magic string identifying the file format

	headerOffMagic    = 0  // [8]byte
	headerOffVersion  = 8  // uint32
	headerOffElemSize = 12 // uint32
	headerOffQuantTag = 16 // uint8  (0=float32, 1=int8)
	headerOffCount    = 20 // uint64 (total vectors ever Put; only in chunk_0)

	headerVersion = uint32(1)
	quantFloat32  = uint8(0)
	quantInt8     = uint8(1)
)

// ChunkFileSize is the size in bytes pre-allocated per chunk file.
// It can be overridden in tests to use smaller chunks.
var ChunkFileSize = int64(1 << 30) // 1 GiB

// mmapChunk represents one pre-allocated, memory-mapped chunk file.
type mmapChunk struct {
	data []byte // mmap-backed slice; len == ChunkFileSize
	f    *os.File
}

// MmapStore is a chunked, memory-mapped vector store.
// Vectors are addressed by a uint32 slot ID.  Writes go through Put(id, data);
// reads through Get(id). Both are protected by mu; Get only needs a read lock.
//
// Addressing: chunk_0 reserves headerSize bytes for its header, so it holds
// chunk0Cap vectors (data at [headerSize:]). Chunks 1+ start data at byte 0
// and hold chunkCap vectors — no wasted header space in subsequent files.
type MmapStore struct {
	dir       string
	elemSize  int // bytes per vector: dims*4 (float32) or dims*1 (int8)
	isInt8    bool
	chunk0Cap int // vector slots in chunk_0  = (ChunkFileSize - headerSize) / elemSize
	chunkCap  int // vector slots in chunk 1+ = ChunkFileSize / elemSize

	mu     sync.RWMutex
	chunks []*mmapChunk
	count  atomic.Uint64 // high-water mark: max(id+1) across all Put calls
}

// NewMmapStore opens or creates a chunked mmap store at dir.
// dims is the number of dimensions per vector; isInt8 selects the element type.
// If dir already contains chunk files, they are opened and the stored header is
// validated against the requested configuration.
func NewMmapStore(dir string, dims int, isInt8 bool) (*MmapStore, error) {
	if dims <= 0 {
		return nil, fmt.Errorf("mmap_store: dims must be > 0, got %d", dims)
	}

	elemSize := dims * 4
	if isInt8 {
		elemSize = dims
	}

	chunk0Cap := int((ChunkFileSize - headerSize) / int64(elemSize))
	chunkCap := int(ChunkFileSize / int64(elemSize))
	if chunk0Cap <= 0 {
		return nil, fmt.Errorf("mmap_store: elemSize %d is too large for ChunkFileSize %d", elemSize, ChunkFileSize)
	}

	if err := os.MkdirAll(dir, 0o750); err != nil { //nolint:gosec // 0o750 satisfies G301
		return nil, fmt.Errorf("mmap_store: create dir %q: %w", dir, err)
	}

	s := &MmapStore{
		dir:       dir,
		elemSize:  elemSize,
		isInt8:    isInt8,
		chunk0Cap: chunk0Cap,
		chunkCap:  chunkCap,
	}

	// Open existing chunks in order (chunk_0.vec, chunk_1.vec, ...).
	for i := 0; ; i++ {
		path := s.chunkPath(i)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			break
		}
		chunk, err := s.openChunk(i)
		if err != nil {
			_ = s.Close() //nolint:errcheck // best-effort cleanup
			return nil, err
		}
		s.chunks = append(s.chunks, chunk)
	}

	if len(s.chunks) > 0 {
		// Validate the header in chunk_0 and restore the count.
		if err := s.validateHeader(s.chunks[0].data); err != nil {
			_ = s.Close() //nolint:errcheck // best-effort cleanup
			return nil, fmt.Errorf("mmap_store: header validation failed: %w", err)
		}
		savedCount := binary.LittleEndian.Uint64(s.chunks[0].data[headerOffCount:])
		s.count.Store(savedCount)
	}

	return s, nil
}

// chunkPath returns the file path for chunk index i.
func (s *MmapStore) chunkPath(i int) string {
	return filepath.Join(s.dir, fmt.Sprintf("chunk_%d.vec", i))
}

// validateHeader checks the header in data against the store's configured
// elemSize and quant tag.  It returns a descriptive error on any mismatch.
func (s *MmapStore) validateHeader(data []byte) error {
	if string(data[headerOffMagic:headerOffMagic+8]) != headerMagic {
		return fmt.Errorf("invalid magic: expected %q, got %q", headerMagic, data[headerOffMagic:headerOffMagic+8])
	}

	if v := binary.LittleEndian.Uint32(data[headerOffVersion:]); v != headerVersion {
		return fmt.Errorf("unsupported version: expected %d, got %d", headerVersion, v)
	}

	if stored := binary.LittleEndian.Uint32(data[headerOffElemSize:]); stored != uint32(s.elemSize) {
		return fmt.Errorf("elemSize mismatch: store has %d bytes/vector, config expects %d", stored, s.elemSize)
	}

	expectedTag := quantFloat32
	if s.isInt8 {
		expectedTag = quantInt8
	}
	if tag := data[headerOffQuantTag]; tag != expectedTag {
		return fmt.Errorf("quantization mismatch: store tag=%d, config tag=%d", tag, expectedTag)
	}

	return nil
}

// writeHeader writes the 4096-byte header into the first bytes of data.
// Called once when chunk_0 is first created.
func (s *MmapStore) writeHeader(data []byte) {
	copy(data[headerOffMagic:], headerMagic)
	binary.LittleEndian.PutUint32(data[headerOffVersion:], headerVersion)
	binary.LittleEndian.PutUint32(data[headerOffElemSize:], uint32(s.elemSize))
	quantTag := quantFloat32
	if s.isInt8 {
		quantTag = quantInt8
	}
	data[headerOffQuantTag] = quantTag
	// count starts at 0 — already zero from ftruncate-zeroed pages.
}

// newChunk creates, pre-allocates, and mmaps a new chunk file.
// PRECONDITION: s.mu.Lock() is held by caller.
func (s *MmapStore) newChunk(idx int) (*mmapChunk, error) {
	path := s.chunkPath(idx)
	f, err := os.Create(path) //nolint:gosec // path constructed from operator config
	if err != nil {
		return nil, fmt.Errorf("mmap_store: create chunk %d: %w", idx, err)
	}

	if err := f.Truncate(ChunkFileSize); err != nil {
		_ = f.Close() //nolint:errcheck // best-effort cleanup
		return nil, fmt.Errorf("mmap_store: truncate chunk %d: %w", idx, err)
	}

	data, err := unix.Mmap(int(f.Fd()), 0, int(ChunkFileSize),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = f.Close() //nolint:errcheck // best-effort cleanup
		return nil, fmt.Errorf("mmap_store: mmap chunk %d: %w", idx, err)
	}

	if idx == 0 {
		s.writeHeader(data)
	}

	return &mmapChunk{data: data, f: f}, nil
}

// openChunk memory-maps an existing chunk file (read-write, shared).
func (s *MmapStore) openChunk(idx int) (*mmapChunk, error) {
	path := s.chunkPath(idx)
	f, err := os.OpenFile(path, os.O_RDWR, 0o644) //nolint:gosec // path from operator config
	if err != nil {
		return nil, fmt.Errorf("mmap_store: open chunk %d: %w", idx, err)
	}

	fi, err := f.Stat()
	if err != nil {
		_ = f.Close() //nolint:errcheck // best-effort cleanup
		return nil, fmt.Errorf("mmap_store: stat chunk %d: %w", idx, err)
	}

	size := fi.Size()
	if size == 0 {
		// Extend empty file to full chunk size (should not normally happen).
		if err := f.Truncate(ChunkFileSize); err != nil {
			_ = f.Close() //nolint:errcheck // best-effort cleanup
			return nil, fmt.Errorf("mmap_store: truncate chunk %d: %w", idx, err)
		}
		size = ChunkFileSize
	}

	data, err := unix.Mmap(int(f.Fd()), 0, int(size),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = f.Close() //nolint:errcheck // best-effort cleanup
		return nil, fmt.Errorf("mmap_store: mmap chunk %d: %w", idx, err)
	}

	return &mmapChunk{data: data, f: f}, nil
}

// idToChunkOffset maps a global slot ID to (chunkIndex, byteOffset within that chunk).
//
// chunk_0: data starts at headerSize  (capacity = chunk0Cap)
// chunk 1+: data starts at 0          (capacity = chunkCap, no wasted header space)
func (s *MmapStore) idToChunkOffset(id uint32) (chunkIdx, byteOffset int) {
	n := int(id)
	if n < s.chunk0Cap {
		return 0, headerSize + n*s.elemSize
	}
	n -= s.chunk0Cap
	chunkIdx = 1 + n/s.chunkCap
	byteOffset = (n % s.chunkCap) * s.elemSize
	return
}

// ensureChunks grows s.chunks until chunkIdx is a valid index.
// PRECONDITION: s.mu.Lock() is held by caller.
func (s *MmapStore) ensureChunks(chunkIdx int) error {
	for len(s.chunks) <= chunkIdx {
		chunk, err := s.newChunk(len(s.chunks))
		if err != nil {
			return err
		}
		s.chunks = append(s.chunks, chunk)
	}
	return nil
}

// Put writes data into slot id, creating chunks as needed.
// It returns a mmap-backed slice aliasing the stored bytes.
// For updates (same id), it overwrites the existing slot in-place.
func (s *MmapStore) Put(id uint32, data []byte) ([]byte, error) {
	if len(data) != s.elemSize {
		return nil, fmt.Errorf("mmap_store: Put size mismatch: expected %d bytes, got %d", s.elemSize, len(data))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	chunkIdx, offset := s.idToChunkOffset(id)
	if err := s.ensureChunks(chunkIdx); err != nil {
		return nil, err
	}

	dst := s.chunks[chunkIdx].data[offset : offset+s.elemSize]
	copy(dst, data)

	// Update high-water mark.
	for {
		old := s.count.Load()
		want := uint64(id) + 1
		if want <= old || s.count.CompareAndSwap(old, want) {
			break
		}
	}

	return dst, nil
}

// Get returns the mmap-backed byte slice for slot id.
// Returns nil if the slot is beyond the current count.
func (s *MmapStore) Get(id uint32) []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	chunkIdx, offset := s.idToChunkOffset(id)
	if chunkIdx >= len(s.chunks) {
		return nil
	}
	return s.chunks[chunkIdx].data[offset : offset+s.elemSize]
}

// Sync flushes all dirty mmap pages to disk (MS_SYNC) and persists the
// high-water-mark count in the chunk_0 header.
func (s *MmapStore) Sync() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.chunks) == 0 {
		return nil
	}

	// Persist count in chunk_0 header before flushing pages.
	binary.LittleEndian.PutUint64(s.chunks[0].data[headerOffCount:], s.count.Load())

	for i, chunk := range s.chunks {
		if err := unix.Msync(chunk.data, unix.MS_SYNC); err != nil {
			return fmt.Errorf("mmap_store: msync chunk %d: %w", i, err)
		}
	}
	return nil
}

// Madvise applies an madvise hint to all mapped chunks.
// Use unix.MADV_SEQUENTIAL for brute-force scan, unix.MADV_RANDOM for HNSW.
func (s *MmapStore) Madvise(advice int) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for i, chunk := range s.chunks {
		if err := unix.Madvise(chunk.data, advice); err != nil {
			return fmt.Errorf("mmap_store: madvise chunk %d: %w", i, err)
		}
	}
	return nil
}

// Close unmaps all chunks and closes the underlying files.
func (s *MmapStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var firstErr error
	for i, chunk := range s.chunks {
		if err := unix.Munmap(chunk.data); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("mmap_store: munmap chunk %d: %w", i, err)
		}
		if err := chunk.f.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("mmap_store: close chunk %d: %w", i, err)
		}
	}
	s.chunks = nil
	return firstErr
}

// Count returns the high-water mark: the number of vector slots ever allocated.
func (s *MmapStore) Count() uint64 {
	return s.count.Load()
}

// Reset closes and deletes all chunk files, then re-initializes the store.
// Used by Clear() to wipe all mmap data after the in-memory structures are reset.
func (s *MmapStore) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Unmap and close all chunks.
	for _, chunk := range s.chunks {
		_ = unix.Munmap(chunk.data) //nolint:errcheck // best-effort cleanup
		_ = chunk.f.Close()         //nolint:errcheck // best-effort cleanup
	}
	s.chunks = nil
	s.count.Store(0)

	// Remove all chunk files.
	for i := 0; ; i++ {
		path := s.chunkPath(i)
		if err := os.Remove(path); os.IsNotExist(err) {
			break
		} else if err != nil {
			return fmt.Errorf("mmap_store: remove chunk %d: %w", i, err)
		}
	}
	return nil
}
