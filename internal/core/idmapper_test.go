package core

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// GetOrCreate hands out IDs in order and returns the same ID for the same
// string; any string, however odd, maps both ways.
func TestIDMapper_GetOrCreate(t *testing.T) {
	ids := []string{"doc1", "doc2", "", strings.Repeat("a", 10_000), "user/profile:456", "unicode-日本語 🎉", "newline\n\ttab", `quotes'"mixed`}

	// Arrange
	m := NewIDMapper()

	// Act
	got := make([]uint32, len(ids))
	for i, id := range ids {
		var err error
		got[i], err = m.GetOrCreate(id)
		require.NoError(t, err)
	}
	again, err := m.GetOrCreate("doc1")

	// Assert
	require.NoError(t, err)
	assert.Equal(t, got[0], again, "the same string gets the same ID")
	assert.Equal(t, len(ids), m.Count())
	assert.Equal(t, uint32(len(ids)), m.NextID())
	for i, id := range ids {
		assert.Equal(t, uint32(i), got[i], "IDs are handed out in order")
		internal, err := m.ToUint32ID(id)
		require.NoError(t, err)
		assert.Equal(t, got[i], internal)
		external, err := m.ToStringID(got[i])
		require.NoError(t, err)
		assert.Equal(t, id, external)
	}
	_, err = m.ToUint32ID("unknown")
	assert.ErrorIs(t, err, ErrNotFound)
	_, err = m.ToStringID(999)
	assert.ErrorIs(t, err, ErrNotFound)
}

// A deleted ID is tombstoned and never handed out again: HNSW graph nodes
// and snapshot records are keyed by it, so reusing one would attach a new
// vector to a stale node.
func TestIDMapper_Delete(t *testing.T) {
	// Arrange
	m := NewIDMapper()
	deletedID, _ := m.GetOrCreate("doc1")
	keptID, _ := m.GetOrCreate("doc2")

	// Act
	err := m.Delete("doc1")
	newID, createErr := m.GetOrCreate("doc3")

	// Assert
	require.NoError(t, err)
	require.NoError(t, createErr)
	assert.Equal(t, 2, m.Count())
	_, err = m.ToUint32ID("doc1")
	assert.ErrorIs(t, err, ErrNotFound)
	_, err = m.ToStringID(deletedID)
	assert.ErrorIs(t, err, ErrNotFound)
	assert.True(t, m.IsTombstone(deletedID))
	assert.False(t, m.IsTombstone(keptID))
	assert.False(t, m.IsTombstone(999), "an ID never handed out is not a tombstone")
	assert.Equal(t, uint32(2), newID, "a deleted ID must not be recycled")
	assert.ErrorIs(t, m.Delete("unknown"), ErrNotFound)
}

func TestIDMapper_Exhausted(t *testing.T) {
	// Arrange
	m := NewIDMapper()
	m.nextID = ^uint32(0)

	// Act
	_, err := m.GetOrCreate("overflow")

	// Assert
	assert.ErrorIs(t, err, ErrIDExhausted)
	assert.Zero(t, m.Count())
}

// Clear forgets everything, tombstones included, so IDs start again at 0.
// That is safe only because Clear goes with wiping the whole collection: no
// live data is left for a reused ID to collide with.
func TestIDMapper_Clear(t *testing.T) {
	// Arrange
	m := NewIDMapper()
	deletedID, _ := m.GetOrCreate("doc1")
	_, _ = m.GetOrCreate("doc2")
	require.NoError(t, m.Delete("doc1"))

	// Act
	m.Clear()
	newID, err := m.GetOrCreate("doc3")

	// Assert
	require.NoError(t, err)
	assert.Equal(t, uint32(0), newID)
	assert.Equal(t, 1, m.Count())
	assert.False(t, m.IsTombstone(deletedID))
}

func TestIDMapper_Concurrent(t *testing.T) {
	t.Run("distinct strings each get an ID", func(t *testing.T) {
		// Arrange
		m := NewIDMapper()
		const goroutines, each = 50, 100

		// Act
		var wg sync.WaitGroup
		for g := range goroutines {
			wg.Go(func() {
				for i := range each {
					_, err := m.GetOrCreate(fmt.Sprintf("doc_%d_%d", g, i))
					assert.NoError(t, err)
				}
			})
		}
		wg.Wait()

		// Assert
		assert.Equal(t, goroutines*each, m.Count())
		assert.Equal(t, uint32(goroutines*each), m.NextID(), "no ID may be skipped or handed out twice")
	})

	t.Run("the same string gets one ID", func(t *testing.T) {
		// Arrange
		m := NewIDMapper()
		ids := make([]uint32, 50)

		// Act
		var wg sync.WaitGroup
		for i := range ids {
			wg.Go(func() {
				id, err := m.GetOrCreate("same")
				assert.NoError(t, err)
				ids[i] = id
			})
		}
		wg.Wait()

		// Assert
		assert.Equal(t, 1, m.Count())
		for _, id := range ids {
			assert.Equal(t, ids[0], id)
		}
	})
}

// A snapshot stores the mapper with gob; decoding it, and encoding and
// decoding that again, must give back every mapping, the counter and the
// tombstones.
func TestIDMapper_GOBRoundTrip(t *testing.T) {
	tests := []struct {
		name    string
		created int
		deleted func(i int) bool
	}{
		{"empty", 0, nil},
		{"with deletes", 100, func(i int) bool { return i%3 == 0 }},
	}
	roundTrip := func(t *testing.T, m *IDMapper) *IDMapper {
		t.Helper()
		var buf bytes.Buffer
		require.NoError(t, m.EncodeGOB(gob.NewEncoder(&buf)))
		decoded := NewIDMapper()
		require.NoError(t, decoded.DecodeGOB(gob.NewDecoder(&buf)))
		return decoded
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			m := NewIDMapper()
			for i := range tt.created {
				_, err := m.GetOrCreate(fmt.Sprintf("doc%d", i))
				require.NoError(t, err)
			}
			for i := range tt.created {
				if tt.deleted(i) {
					require.NoError(t, m.Delete(fmt.Sprintf("doc%d", i)))
				}
			}

			// Act
			decoded := roundTrip(t, roundTrip(t, m))

			// Assert
			assert.Equal(t, m.Count(), decoded.Count())
			assert.Equal(t, m.NextID(), decoded.NextID())
			for i := range tt.created {
				id := fmt.Sprintf("doc%d", i)
				internal, err := decoded.ToUint32ID(id)
				if tt.deleted(i) {
					assert.ErrorIs(t, err, ErrNotFound, id)
					assert.True(t, decoded.IsTombstone(uint32(i)), id)
					continue
				}
				require.NoError(t, err)
				assert.Equal(t, uint32(i), internal)
			}
		})
	}
}
