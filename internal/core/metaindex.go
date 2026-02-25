package core

import (
	"fmt"
	"strconv"
)

// filterKey encodes a single metadata field:value pair as a flat, comparable map key.
// Storing value as a normalised string avoids using `any` as a map key, which is
// untyped and risks runtime panics on non-comparable values.
type filterKey struct {
	field string
	value string // normalised string representation of the original value
}

// MetadataIndex is an in-memory inverted index over metadata fields.
// It maps field:value pairs to the list of internal document IDs that carry them,
// enabling O(1) allowlist computation for filtered HNSW search.
//
// NOT thread-safe — callers must hold the owning index's write lock.
type MetadataIndex struct {
	entries map[filterKey][]uint32
}

// NewMetadataIndex returns an empty MetadataIndex ready for use.
func NewMetadataIndex() *MetadataIndex {
	return &MetadataIndex{
		entries: make(map[filterKey][]uint32),
	}
}

// normalise converts a metadata value to a consistent string for use as a map key.
// Numeric types (int, float32, float64, etc.) are unified so that int(2020) and
// float64(2020) produce the same key — matching the behaviour of toFloat in filters.go.
func normalise(v any) string {
	if f, ok := toFloat(v); ok {
		return strconv.FormatFloat(f, 'f', -1, 64)
	}
	return fmt.Sprintf("%v", v)
}

// Add records all metadata field:value pairs for internalID.
// If internalID is being updated, Remove must be called first with the old metadata.
func (m *MetadataIndex) Add(internalID uint32, meta map[string]any) {
	for field, val := range meta {
		key := filterKey{field: field, value: normalise(val)}
		m.entries[key] = append(m.entries[key], internalID)
	}
}

// Remove deletes internalID from every field:value list it was recorded under.
// Idempotent: removing an ID not present is a no-op.
func (m *MetadataIndex) Remove(internalID uint32, meta map[string]any) {
	for field, val := range meta {
		key := filterKey{field: field, value: normalise(val)}
		ids := m.entries[key]
		for i, id := range ids {
			if id == internalID {
				m.entries[key] = append(ids[:i], ids[i+1:]...)
				break
			}
		}
		if len(m.entries[key]) == 0 {
			delete(m.entries, key)
		}
	}
}

// Allowlist returns the set of internal IDs matching ALL provided filters (AND logic).
//   - Returns nil if filters is nil (no filter — all documents qualify).
//   - Returns an empty map if filters is non-nil but no documents match.
func (m *MetadataIndex) Allowlist(filters map[string]any) map[uint32]struct{} {
	if filters == nil {
		return nil
	}

	var result map[uint32]struct{}

	for field, val := range filters {
		key := filterKey{field: field, value: normalise(val)}
		ids := m.entries[key]

		if len(ids) == 0 {
			return map[uint32]struct{}{} // short-circuit: this filter matches nothing
		}

		if result == nil {
			// Seed the working set from the first filter.
			result = make(map[uint32]struct{}, len(ids))
			for _, id := range ids {
				result[id] = struct{}{}
			}
			continue
		}

		// Intersect: keep only IDs that appear in both the working set and this filter's list.
		// Iterate over ids (the candidate list) and probe result (the working set).
		next := make(map[uint32]struct{})
		for _, id := range ids {
			if _, ok := result[id]; ok {
				next[id] = struct{}{}
			}
		}
		result = next

		if len(result) == 0 {
			return result // short-circuit: the intersection is already empty
		}
	}

	if result == nil {
		// filters was a non-nil but empty map — treat as "no filter applied".
		return map[uint32]struct{}{}
	}

	return result
}

// Clear resets the index to empty.
func (m *MetadataIndex) Clear() {
	m.entries = make(map[filterKey][]uint32)
}
