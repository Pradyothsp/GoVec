//go:build !unix

package storage

import "errors"

var errNotSupported = errors.New("mmap storage is not supported on this platform")

// MmapStore is a stub for non-Unix platforms.
type MmapStore struct{}

// NewMmapStore always returns an error on non-Unix platforms.
func NewMmapStore(dir string, dims int, isInt8 bool) (*MmapStore, error) {
	return nil, errNotSupported
}

// Put is a stub that always returns an error.
func (s *MmapStore) Put(_ uint32, _ []byte) ([]byte, error) {
	return nil, errNotSupported
}

// Get is a stub that always returns nil.
func (s *MmapStore) Get(_ uint32) []byte { return nil }

// Sync is a stub.
func (s *MmapStore) Sync() error { return errNotSupported }

// Close is a stub.
func (s *MmapStore) Close() error { return nil }

// SetAccessPattern is a stub.
func (s *MmapStore) SetAccessPattern(_ AccessPattern) error { return errNotSupported }

// AccessPattern returns AccessNormal.
func (s *MmapStore) AccessPattern() AccessPattern { return AccessNormal }

// Count returns 0.
func (s *MmapStore) Count() uint64 { return 0 }

// Reset is a stub.
func (s *MmapStore) Reset() error { return nil }
