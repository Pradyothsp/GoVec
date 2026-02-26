package storage

import "unsafe"

// SliceVec is the union of supported vector slice types.
type SliceVec interface {
	[]float32 | []int8
}

// VectorToBytes reinterprets a typed vector slice as raw bytes — zero copy.
// The returned slice aliases the same memory; the caller must not use it
// after the source vector has been freed or invalidated.
func VectorToBytes[T SliceVec](v T) []byte {
	if len(v) == 0 {
		return nil
	}
	switch sv := any(v).(type) {
	case []float32:
		return unsafe.Slice((*byte)(unsafe.Pointer(&sv[0])), len(sv)*4) //nolint:gosec // unsafe.Slice is intentional: zero-copy alias into mmap region
	case []int8:
		return unsafe.Slice((*byte)(unsafe.Pointer(&sv[0])), len(sv)*1) //nolint:gosec // unsafe.Slice is intentional: zero-copy alias into mmap region
	default:
		panic("storage: unsupported vector type")
	}
}

// BytesToVector reinterprets raw bytes as a typed vector slice — zero copy.
// The returned slice aliases the same memory; typically the caller holds an
// mmap region for the lifetime of all returned slices.
func BytesToVector[T SliceVec](b []byte) T {
	if len(b) == 0 {
		return nil
	}
	var zero T
	switch any(zero).(type) {
	case []float32:
		n := len(b) / 4
		sl := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n) //nolint:gosec // unsafe.Slice is intentional: zero-copy alias into mmap region
		return any(sl).(T)                                       //nolint:errcheck // type assertion is safe: branch only executes when T=[]float32
	case []int8:
		n := len(b)
		sl := unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), n) //nolint:gosec // unsafe.Slice is intentional: zero-copy alias into mmap region
		return any(sl).(T)                                    //nolint:errcheck // type assertion is safe: branch only executes when T=[]int8
	default:
		panic("storage: unsupported vector type")
	}
}
