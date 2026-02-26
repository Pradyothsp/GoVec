package index

import (
	"fmt"

	"github.com/Pradyothsp/govec/internal/storage"
)

// putToMmapStore writes an encoded vector of type T to the mmap store at slot id
// and returns the mmap-backed slice.  T must be []float32 or []int8.
func putToMmapStore[T any](vs *storage.MmapStore, id uint32, encoded T) (T, error) {
	switch v := any(encoded).(type) {
	case []float32:
		raw := storage.VectorToBytes(v)
		mmapSlice, err := vs.Put(id, raw)
		if err != nil {
			return encoded, err
		}
		return any(storage.BytesToVector[[]float32](mmapSlice)).(T), nil //nolint:errcheck // type assertion is safe: branch only executes when T=[]float32
	case []int8:
		raw := storage.VectorToBytes(v)
		mmapSlice, err := vs.Put(id, raw)
		if err != nil {
			return encoded, err
		}
		return any(storage.BytesToVector[[]int8](mmapSlice)).(T), nil //nolint:errcheck // type assertion is safe: branch only executes when T=[]int8
	default:
		return encoded, fmt.Errorf("mmap: unsupported vector type %T", encoded)
	}
}

// getFromMmapStore retrieves slot id from the mmap store and reinterprets it as T.
// Returns (zero, false) if the slot is out of range.
func getFromMmapStore[T any](vs *storage.MmapStore, id uint32) (T, bool) {
	raw := vs.Get(id)
	if raw == nil {
		var zero T
		return zero, false
	}
	switch any(*new(T)).(type) {
	case []float32:
		return any(storage.BytesToVector[[]float32](raw)).(T), true //nolint:errcheck // type assertion is safe: branch only executes when T=[]float32
	case []int8:
		return any(storage.BytesToVector[[]int8](raw)).(T), true //nolint:errcheck // type assertion is safe: branch only executes when T=[]int8
	default:
		var zero T
		return zero, false
	}
}
