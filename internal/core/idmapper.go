package core

import (
	"encoding/gob"
	"errors"
	"sync"
)

var (
	// ErrNotFound indicates that the requested identifier does not exist or has been deleted.
	ErrNotFound = errors.New("ID not found")
	// ErrIDExhausted indicates that the uint32 ID space has been fully utilized and no more IDs can be assigned.
	ErrIDExhausted = errors.New("uint32 ID space exhausted")
)

// IDMapper translates between external string IDs and internal uint32 IDs.
// It maintains bidirectional mappings and supports tombstone-based deletion
// (deleted IDs are never recycled to prevent race conditions).
type IDMapper struct {
	mu         sync.RWMutex
	nextID     uint32              // Monotonically increasing counter
	stringToID map[string]uint32   // "doc_abc" → 0
	idToString map[uint32]string   // 0 → "doc_abc"
	tombstones map[uint32]struct{} // Deleted IDs (never reused)
}

// NewIDMapper creates a new IDMapper with empty mappings.
func NewIDMapper() *IDMapper {
	return &IDMapper{
		stringToID: make(map[string]uint32),
		idToString: make(map[uint32]string),
		tombstones: make(map[uint32]struct{}),
	}
}

// GetOrCreate returns the existing uint32 ID for a string ID, or creates a new
// sequential ID if not found. Returns ErrIDExhausted if uint32 space is full.
func (m *IDMapper) GetOrCreate(stringID string) (uint32, error) {
	// Fast path: check if ID already exists (read lock)
	m.mu.RLock()
	if id, exists := m.stringToID[stringID]; exists {
		m.mu.RUnlock()
		return id, nil
	}
	m.mu.RUnlock()

	// Slow path: create new ID (write lock)
	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check after acquiring write lock (another goroutine may have created it)
	if id, exists := m.stringToID[stringID]; exists {
		return id, nil
	}

	// Check for uint32 overflow
	if m.nextID == ^uint32(0) {
		return 0, ErrIDExhausted
	}

	newID := m.nextID
	m.nextID++

	m.stringToID[stringID] = newID
	m.idToString[newID] = stringID

	return newID, nil
}

// ToStringID translates a uint32 ID back to its original string ID.
// Returns ErrNotFound if the ID doesn't exist or has been deleted.
func (m *IDMapper) ToStringID(id uint32) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stringID, exists := m.idToString[id]
	if !exists {
		return "", ErrNotFound
	}

	return stringID, nil
}

// ToUint32ID translates a string ID to its uint32 ID.
// Returns ErrNotFound if the string ID doesn't exist or has been deleted.
func (m *IDMapper) ToUint32ID(stringID string) (uint32, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	id, exists := m.stringToID[stringID]
	if !exists {
		return 0, ErrNotFound
	}

	return id, nil
}

// Delete marks a string ID as deleted (tombstone). The uint32 ID is never
// recycled to prevent race conditions. Returns ErrNotFound if ID doesn't exist.
func (m *IDMapper) Delete(stringID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	id, exists := m.stringToID[stringID]
	if !exists {
		return ErrNotFound
	}

	// Mark as tombstone (never recycle)
	m.tombstones[id] = struct{}{}
	delete(m.stringToID, stringID)
	delete(m.idToString, id)

	return nil
}

// Clear resets the mapper to its initial empty state: all string/uint32
// mappings and tombstones are dropped, and nextID restarts at 0. This is
// only safe when the entire collection is being wiped alongside it (e.g.
// Engine.Reset) -- unlike Delete, it does not preserve the never-recycle
// guarantee, since there is no live data left for a recycled ID to collide
// with.
func (m *IDMapper) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.nextID = 0
	m.stringToID = make(map[string]uint32)
	m.idToString = make(map[uint32]string)
	m.tombstones = make(map[uint32]struct{})
}

// IsTombstone checks if a uint32 ID has been deleted.
func (m *IDMapper) IsTombstone(id uint32) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	_, isTombstone := m.tombstones[id]
	return isTombstone
}

// Count returns the number of active ID mappings (excludes tombstones).
func (m *IDMapper) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.stringToID)
}

// NextID returns the next uint32 ID that will be assigned.
func (m *IDMapper) NextID() uint32 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.nextID
}

// EncodeGOB serializes the IDMapper state to a GOB encoder.
func (m *IDMapper) EncodeGOB(encoder *gob.Encoder) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Encode in order: nextID, stringToID, idToString, tombstones
	if err := encoder.Encode(m.nextID); err != nil {
		return err
	}
	if err := encoder.Encode(m.stringToID); err != nil {
		return err
	}
	if err := encoder.Encode(m.idToString); err != nil {
		return err
	}
	if err := encoder.Encode(m.tombstones); err != nil {
		return err
	}

	return nil
}

// DecodeGOB deserializes the IDMapper state from a GOB decoder.
func (m *IDMapper) DecodeGOB(decoder *gob.Decoder) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Decode in same order: nextID, stringToID, idToString, tombstones
	if err := decoder.Decode(&m.nextID); err != nil {
		return err
	}
	if err := decoder.Decode(&m.stringToID); err != nil {
		return err
	}
	if err := decoder.Decode(&m.idToString); err != nil {
		return err
	}
	if err := decoder.Decode(&m.tombstones); err != nil {
		return err
	}

	return nil
}
