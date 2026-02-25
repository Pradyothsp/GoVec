package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMetadataIndex_NilFilter_ReturnsNil(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi"})

	result := m.Allowlist(nil)
	assert.Nil(t, result, "nil filter must return nil — caller interprets this as no filter applied")
}

func TestMetadataIndex_Allowlist_SingleFilter(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi"})
	m.Add(2, map[string]any{"genre": "comedy"})
	m.Add(3, map[string]any{"genre": "sci-fi"})

	result := m.Allowlist(map[string]any{"genre": "sci-fi"})
	require.NotNil(t, result)
	assert.Equal(t, map[uint32]struct{}{1: {}, 3: {}}, result)
}

func TestMetadataIndex_Allowlist_MultipleFilters_Intersection(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi", "year": 2020})
	m.Add(2, map[string]any{"genre": "sci-fi", "year": 2019})
	m.Add(3, map[string]any{"genre": "comedy", "year": 2020})

	// Only doc 1 is sci-fi AND 2020
	result := m.Allowlist(map[string]any{"genre": "sci-fi", "year": 2020})
	require.NotNil(t, result)
	assert.Equal(t, map[uint32]struct{}{1: {}}, result)
}

func TestMetadataIndex_Allowlist_NoMatch_ReturnsEmptyMap(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "comedy"})

	result := m.Allowlist(map[string]any{"genre": "sci-fi"})
	require.NotNil(t, result, "non-nil filter with no match must return empty map, not nil")
	assert.Empty(t, result)
}

func TestMetadataIndex_Allowlist_ShortCircuit_OnFirstEmptyFilter(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi", "tier": "premium"})

	// "genre" matches but "tier"="free" matches nothing — must short-circuit
	result := m.Allowlist(map[string]any{"genre": "sci-fi", "tier": "free"})
	require.NotNil(t, result)
	assert.Empty(t, result)
}

func TestMetadataIndex_Remove_ExcludesFromAllowlist(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi"})
	m.Add(2, map[string]any{"genre": "sci-fi"})

	m.Remove(1, map[string]any{"genre": "sci-fi"})

	result := m.Allowlist(map[string]any{"genre": "sci-fi"})
	assert.Equal(t, map[uint32]struct{}{2: {}}, result)
}

func TestMetadataIndex_Remove_Idempotent(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi"})

	// Double remove must not panic or corrupt state
	m.Remove(1, map[string]any{"genre": "sci-fi"})
	m.Remove(1, map[string]any{"genre": "sci-fi"})

	result := m.Allowlist(map[string]any{"genre": "sci-fi"})
	assert.Empty(t, result)
}

func TestMetadataIndex_NumericNormalisation(t *testing.T) {
	m := NewMetadataIndex()

	// Insert with int — JSON typically decodes numbers as float64
	m.Add(1, map[string]any{"year": int(2020)})

	// Query with float64 — must still match
	result := m.Allowlist(map[string]any{"year": float64(2020)})
	require.NotNil(t, result)
	assert.Equal(t, map[uint32]struct{}{1: {}}, result)
}

func TestMetadataIndex_Clear_ResetsState(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"genre": "sci-fi"})
	m.Add(2, map[string]any{"genre": "comedy"})

	m.Clear()

	result := m.Allowlist(map[string]any{"genre": "sci-fi"})
	require.NotNil(t, result)
	assert.Empty(t, result)
}

func TestMetadataIndex_Update_OldMetaRemoved(t *testing.T) {
	m := NewMetadataIndex()
	m.Add(1, map[string]any{"tier": "free"})

	// Simulate an update: remove old, add new
	m.Remove(1, map[string]any{"tier": "free"})
	m.Add(1, map[string]any{"tier": "premium"})

	free := m.Allowlist(map[string]any{"tier": "free"})
	assert.Empty(t, free, "old tier should no longer match after update")

	premium := m.Allowlist(map[string]any{"tier": "premium"})
	assert.Equal(t, map[uint32]struct{}{1: {}}, premium)
}
