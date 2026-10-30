//go:build unix

package storage

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testDims is the width of every test vector: 4 float32s, 16 bytes.
const testDims = 4

// TestMain shrinks chunk files from 1 GiB to 64 KiB, so tests can cross chunk
// boundaries without writing gigabytes. ChunkFileSize is package state, which
// is why these tests don't run in parallel.
func TestMain(m *testing.M) {
	ChunkFileSize = 1 << 16
	os.Exit(m.Run())
}

// slots returns how many vectors chunk_0 and each later chunk hold.
func slots(elemSize int) (chunk0, later int) {
	return int((ChunkFileSize - headerSize) / int64(elemSize)), int(ChunkFileSize / int64(elemSize))
}

// vectorBytes returns a recognisable vector for id, as the bytes Put takes.
func vectorBytes(id uint32, isInt8 bool) []byte {
	if isInt8 {
		return VectorToBytes([]int8{int8(id), int8(id >> 8), -1, 1})
	}

	return VectorToBytes([]float32{float32(id), float32(id) + 0.5, -1, 1})
}

func openStore(t *testing.T, dir string, isInt8 bool) *MmapStore {
	t.Helper()

	s, err := NewMmapStore(dir, testDims, isInt8)
	require.NoError(t, err)

	return s
}

func chunkFiles(t *testing.T, dir string) []string {
	t.Helper()

	files, err := filepath.Glob(filepath.Join(dir, "chunk_*.vec"))
	require.NoError(t, err)

	return files
}

// Slots are addressed across chunk files: chunk_0 loses headerSize bytes to
// its header, later chunks don't. A slot on either side of each boundary must
// land in its own place.
func TestMmapStore_PutGet_AcrossChunks(t *testing.T) {
	for _, isInt8 := range []bool{false, true} {
		t.Run(fmt.Sprintf("int8=%v", isInt8), func(t *testing.T) {
			// Arrange
			dir := t.TempDir()
			s := openStore(t, dir, isInt8)
			t.Cleanup(func() { _ = s.Close() })

			elemSize := testDims * 4
			if isInt8 {
				elemSize = testDims
			}

			chunk0, later := slots(elemSize)
			ids := []uint32{0, uint32(chunk0 - 1), uint32(chunk0), uint32(chunk0 + later - 1), uint32(chunk0 + later + 5)}

			// Act
			for _, id := range ids {
				_, err := s.Put(id, vectorBytes(id, isInt8))
				require.NoError(t, err)
			}

			// Assert
			for _, id := range ids {
				assert.Equal(t, vectorBytes(id, isInt8), s.Get(id), "slot %d", id)
			}

			assert.Len(t, chunkFiles(t, dir), 3)
			assert.Equal(t, uint64(ids[len(ids)-1])+1, s.Count())
		})
	}
}

// Put returns the stored bytes themselves, not a copy: the index keeps that
// slice as the vector, so a later overwrite of the slot must show through it.
func TestMmapStore_Put_ReturnsTheStoredBytes(t *testing.T) {
	// Arrange
	s := openStore(t, t.TempDir(), false)
	t.Cleanup(func() { _ = s.Close() })

	stored, err := s.Put(7, vectorBytes(1, false))
	require.NoError(t, err)

	// Act
	_, err = s.Put(7, vectorBytes(2, false))

	// Assert
	require.NoError(t, err)
	assert.Equal(t, vectorBytes(2, false), stored)
}

func TestMmapStore_Put_WrongSize_Rejected(t *testing.T) {
	// Arrange
	s := openStore(t, t.TempDir(), false)
	t.Cleanup(func() { _ = s.Close() })

	// Act
	_, err := s.Put(0, make([]byte, testDims*4-1))

	// Assert
	require.Error(t, err)
	assert.Contains(t, err.Error(), "size mismatch")
	assert.Zero(t, s.Count())
}

// Count is a high-water mark, not the number of vectors: the index uses it as
// the number of slots ever handed out.
func TestMmapStore_Count_IsHighWaterMark(t *testing.T) {
	// Arrange
	s := openStore(t, t.TempDir(), false)
	t.Cleanup(func() { _ = s.Close() })

	// Act
	for _, id := range []uint32{10, 3, 10} {
		_, err := s.Put(id, vectorBytes(id, false))
		require.NoError(t, err)
	}

	// Assert
	assert.Equal(t, uint64(11), s.Count())
}

// Get answers for slots below the count, the slots ever handed out, and nil
// at or past it: snapshot loading relies on nil to notice lost vectors.
func TestMmapStore_Get(t *testing.T) {
	_, later := slots(testDims * 4)

	tests := []struct {
		name string
		id   uint32
		want []byte
	}{
		{name: "a written slot", id: 2, want: vectorBytes(2, false)},
		{name: "an unwritten slot below the count reads as zeros", id: 1, want: make([]byte, testDims*4)},
		{name: "the count itself, inside an existing chunk", id: 3, want: nil},
		{name: "past every chunk", id: uint32(10 * later), want: nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			s := openStore(t, t.TempDir(), false)
			t.Cleanup(func() { _ = s.Close() })

			for _, id := range []uint32{0, 2} {
				_, err := s.Put(id, vectorBytes(id, false))
				require.NoError(t, err)
			}

			// Act
			got := s.Get(tt.id)

			// Assert
			assert.Equal(t, tt.want, got)
		})
	}
}

// Under mmap the snapshot keeps only slot numbers, so a restart must find
// every vector, and the count, where it left them.
func TestMmapStore_Reopen_KeepsVectorsAndCount(t *testing.T) {
	for _, isInt8 := range []bool{false, true} {
		t.Run(fmt.Sprintf("int8=%v", isInt8), func(t *testing.T) {
			// Arrange
			dir := t.TempDir()
			s := openStore(t, dir, isInt8)

			chunk0, _ := slots(testDims * 4)
			ids := []uint32{0, 5, uint32(chunk0 + 1)}

			for _, id := range ids {
				_, err := s.Put(id, vectorBytes(id, isInt8))
				require.NoError(t, err)
			}

			require.NoError(t, s.Sync())
			require.NoError(t, s.Close())

			// Act
			reopened := openStore(t, dir, isInt8)
			t.Cleanup(func() { _ = reopened.Close() })

			// Assert
			for _, id := range ids {
				assert.Equal(t, vectorBytes(id, isInt8), reopened.Get(id), "slot %d", id)
			}

			assert.Equal(t, uint64(chunk0+2), reopened.Count())
		})
	}
}

// A store opened with settings that don't match its files would read every
// vector at the wrong width or type, so it must refuse to open.
func TestNewMmapStore_FilesThatDontMatch_Refused(t *testing.T) {
	tests := []struct {
		name   string
		dims   int
		isInt8 bool
		damage func(header []byte)
		want   string
	}{
		{name: "different width", dims: testDims * 2, want: "elemSize mismatch"},
		{name: "float32 store opened as int8", dims: testDims * 4, isInt8: true, want: "quantization mismatch"},
		{name: "damaged magic", dims: testDims, damage: func(h []byte) { h[headerOffMagic] = 'X' }, want: "invalid magic"},
		{name: "unknown version", dims: testDims, damage: func(h []byte) {
			binary.LittleEndian.PutUint32(h[headerOffVersion:], 99)
		}, want: "unsupported version"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			dir := t.TempDir()
			s := openStore(t, dir, false)

			_, err := s.Put(0, vectorBytes(0, false))
			require.NoError(t, err)
			require.NoError(t, s.Close())

			if tt.damage != nil {
				path := filepath.Join(dir, "chunk_0.vec")
				data, err := os.ReadFile(path)
				require.NoError(t, err)

				tt.damage(data)
				require.NoError(t, os.WriteFile(path, data, 0o600))
			}

			// Act
			_, err = NewMmapStore(dir, tt.dims, tt.isInt8)

			// Assert
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

func TestNewMmapStore_InvalidSettings_Refused(t *testing.T) {
	tests := []struct {
		name string
		dims int
		want string
	}{
		{name: "zero dims", dims: 0, want: "dims must be > 0"},
		{name: "a vector bigger than a chunk", dims: int(ChunkFileSize), want: "too large"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			_, err := NewMmapStore(t.TempDir(), tt.dims, false)

			// Assert
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

// Reset removes every chunk file and leaves a store that works from empty,
// with a fresh header a later open accepts.
func TestMmapStore_Reset_RemovesChunksAndStartsOver(t *testing.T) {
	// Arrange
	dir := t.TempDir()
	s := openStore(t, dir, false)

	chunk0, _ := slots(testDims * 4)
	for _, id := range []uint32{0, uint32(chunk0)} {
		_, err := s.Put(id, vectorBytes(id, false))
		require.NoError(t, err)
	}

	require.Len(t, chunkFiles(t, dir), 2)

	// Act
	err := s.Reset()

	// Assert
	require.NoError(t, err)
	assert.Empty(t, chunkFiles(t, dir))
	assert.Zero(t, s.Count())

	_, err = s.Put(1, vectorBytes(1, false))
	require.NoError(t, err)
	require.NoError(t, s.Sync())
	require.NoError(t, s.Close())

	reopened := openStore(t, dir, false)
	t.Cleanup(func() { _ = reopened.Close() })
	assert.Equal(t, vectorBytes(1, false), reopened.Get(1))
	assert.Equal(t, uint64(2), reopened.Count())
}

// Puts grow the chunk list under the write lock while Gets read it under the
// read lock. Run under -race; the assertions check every slot landed whole.
func TestMmapStore_ConcurrentPutAndGet(t *testing.T) {
	// Arrange
	s := openStore(t, t.TempDir(), false)
	t.Cleanup(func() { _ = s.Close() })

	chunk0, later := slots(testDims * 4)
	n := chunk0 + 2*later
	const writers = 4

	// Act
	var wg sync.WaitGroup
	for w := range writers {
		wg.Go(func() {
			for id := w; id < n; id += writers {
				if _, err := s.Put(uint32(id), vectorBytes(uint32(id), false)); err != nil {
					t.Error(err)
					return
				}

				_ = s.Get(uint32(id / 2))
			}
		})
	}

	wg.Wait()

	// Assert
	for id := range n {
		require.Equal(t, vectorBytes(uint32(id), false), s.Get(uint32(id)), "slot %d", id)
	}

	assert.Equal(t, uint64(n), s.Count())
}

// recordDirSyncs replaces the directory sync with one that records each call
// and still syncs.
func recordDirSyncs(t *testing.T) *[]string {
	t.Helper()

	var synced []string
	syncDirHook = func(dir string) error {
		synced = append(synced, dir)
		return syncDir(dir)
	}

	t.Cleanup(func() { syncDirHook = syncDir })

	return &synced
}

// A new chunk file must be durable by name, not just by contents, before
// vectors go into it; so must a reset's removal of them. A power cut can't be
// simulated here, so this checks the directory was synced.
func TestMmapStore_DirectoryIsSynced(t *testing.T) {
	chunk0, _ := slots(testDims * 4)

	tests := []struct {
		name string
		act  func(t *testing.T, s *MmapStore)
		want int
	}{
		{
			name: "each new chunk file",
			act: func(t *testing.T, s *MmapStore) {
				for _, id := range []uint32{0, 1, uint32(chunk0)} {
					_, err := s.Put(id, vectorBytes(id, false))
					require.NoError(t, err)
				}
			},
			want: 2,
		},
		{
			name: "reset",
			act: func(t *testing.T, s *MmapStore) {
				require.NoError(t, s.Reset())
			},
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			dir := t.TempDir()
			s := openStore(t, dir, false)
			t.Cleanup(func() { _ = s.Close() })

			synced := recordDirSyncs(t)

			// Act
			tt.act(t, s)

			// Assert
			assert.Len(t, *synced, tt.want)

			for _, d := range *synced {
				assert.Equal(t, dir, d)
			}
		})
	}
}

// If a chunk in the middle is lost, opening must fail rather than quietly use
// the chunks before it; and the chunk files after the gap must survive, where
// growing the store used to recreate, and so empty, them.
func TestNewMmapStore_MissingChunk_RefusedAndLaterChunksKept(t *testing.T) {
	// Arrange
	dir := t.TempDir()
	s := openStore(t, dir, false)

	chunk0, later := slots(testDims * 4)
	lastID := uint32(chunk0 + later)

	for _, id := range []uint32{0, uint32(chunk0), lastID} {
		_, err := s.Put(id, vectorBytes(id, false))
		require.NoError(t, err)
	}

	require.NoError(t, s.Sync())
	require.NoError(t, s.Close())

	lastChunk := filepath.Join(dir, "chunk_2.vec")
	before, err := os.ReadFile(lastChunk)
	require.NoError(t, err)

	require.NoError(t, os.Remove(filepath.Join(dir, "chunk_1.vec")))

	// Act
	_, err = NewMmapStore(dir, testDims, false)

	// Assert
	require.Error(t, err)
	assert.Contains(t, err.Error(), "chunk_1.vec is missing")

	after, err := os.ReadFile(lastChunk)
	require.NoError(t, err)
	assert.Equal(t, before, after)
}

// withChunkFileSize sets ChunkFileSize for one test and puts it back after.
func withChunkFileSize(t *testing.T, size int64) {
	t.Helper()

	old := ChunkFileSize
	ChunkFileSize = size
	t.Cleanup(func() { ChunkFileSize = old })
}

// Slots are addressed from the chunk size, so a store written with one size
// and opened with another would read every later vector from the wrong place.
func TestNewMmapStore_DifferentChunkSize_Refused(t *testing.T) {
	// Arrange
	dir := t.TempDir()
	s := openStore(t, dir, false)

	_, err := s.Put(0, vectorBytes(0, false))
	require.NoError(t, err)
	require.NoError(t, s.Close())

	withChunkFileSize(t, 2*ChunkFileSize)

	// Act
	_, err = NewMmapStore(dir, testDims, false)

	// Assert
	require.Error(t, err)
	assert.Contains(t, err.Error(), "chunk size mismatch")
}

// Stores written before the header recorded the chunk size hold 0 there, and
// were all written with 1 GiB chunks.
func TestValidateHeader_NoRecordedChunkSize_ReadAsOneGiB(t *testing.T) {
	tests := []struct {
		name      string
		chunkSize int64
		wantErr   bool
	}{
		{name: "this build uses 1 GiB", chunkSize: 1 << 30},
		{name: "this build uses another size", chunkSize: 1 << 16, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			withChunkFileSize(t, tt.chunkSize)

			s := &MmapStore{elemSize: testDims * 4}
			header := make([]byte, headerSize)
			s.writeHeader(header)
			binary.LittleEndian.PutUint64(header[headerOffChunk:], 0)

			// Act
			err := s.validateHeader(header)

			// Assert
			if tt.wantErr {
				assert.ErrorContains(t, err, "chunk size mismatch")
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestNewMmapStore_ChunkOfWrongSize_Refused(t *testing.T) {
	// Arrange
	dir := t.TempDir()
	s := openStore(t, dir, false)

	chunk0, _ := slots(testDims * 4)
	for _, id := range []uint32{0, uint32(chunk0)} {
		_, err := s.Put(id, vectorBytes(id, false))
		require.NoError(t, err)
	}

	require.NoError(t, s.Sync())
	require.NoError(t, s.Close())
	require.NoError(t, os.Truncate(filepath.Join(dir, "chunk_1.vec"), ChunkFileSize/2))

	// Act
	_, err := NewMmapStore(dir, testDims, false)

	// Assert
	require.Error(t, err)
	assert.Contains(t, err.Error(), "chunk_1.vec is")
}
