package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type ids = map[uint32]struct{}

func TestMetadataIndex_Allowlist(t *testing.T) {
	// Arrange: one index shared by every case.
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi", "year": 2020, "tier": "premium"})
	m.Add(2, map[string]any{"genre": "sci-fi", "year": 2019})
	m.Add(3, map[string]any{"genre": "comedy", "year": 2020})

	tests := []struct {
		name    string
		filters map[string]any
		want    ids
	}{
		// nil means "no filter"; an empty map means "nothing matched".
		{"nil filter is no filter", nil, nil},
		{"single filter", map[string]any{"genre": "sci-fi"}, ids{1: {}, 2: {}}},
		{"filters intersect", map[string]any{"genre": "sci-fi", "year": 2020}, ids{1: {}}},
		{"no match is empty, not nil", map[string]any{"genre": "drama"}, ids{}},
		{"one filter matching nothing empties the result", map[string]any{"genre": "sci-fi", "tier": "free"}, ids{}},
		// JSON decodes numbers as float64; a stored int must still match.
		{"numbers normalise across types", map[string]any{"year": float64(2019)}, ids{2: {}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			got := m.Allowlist(tt.filters)

			// Assert
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMetadataIndex_Changes(t *testing.T) {
	sciFi := map[string]any{"genre": "sci-fi"}
	tests := []struct {
		name   string
		change func(m *MetadataIndex)
		query  map[string]any
		want   ids
	}{
		{"remove excludes the document", func(m *MetadataIndex) {
			m.Remove(1, sciFi)
		}, sciFi, ids{2: {}}},
		// Removing the last holder drops the value's list; removing again is harmless.
		{"remove every holder, then again", func(m *MetadataIndex) {
			m.Remove(1, sciFi)
			m.Remove(2, sciFi)
			m.Remove(2, sciFi)
		}, sciFi, ids{}},
		{"clear empties the index", func(m *MetadataIndex) {
			m.Clear()
		}, sciFi, ids{}},
		{"an update drops the old value", func(m *MetadataIndex) {
			m.Remove(1, sciFi)
			m.Add(1, map[string]any{"genre": "comedy"})
		}, map[string]any{"genre": "comedy"}, ids{1: {}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			m := NewMetadataIndex()
			m.Add(1, sciFi)
			m.Add(2, sciFi)

			// Act
			tt.change(m)

			// Assert
			assert.Equal(t, tt.want, m.Allowlist(tt.query))
		})
	}
}
