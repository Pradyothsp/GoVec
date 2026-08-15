package core

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"sync"
	"testing"
)

// TestNewIDMapper tests basic initialization
func TestNewIDMapper(t *testing.T) {
	mapper := NewIDMapper()

	if mapper == nil {
		t.Fatal("NewIDMapper returned nil")
	}

	if mapper.Count() != 0 {
		t.Errorf("Expected Count() = 0, got %d", mapper.Count())
	}

	if mapper.NextID() != 0 {
		t.Errorf("Expected NextID() = 0, got %d", mapper.NextID())
	}
}

// TestGetOrCreate_Basic tests basic ID creation
func TestGetOrCreate_Basic(t *testing.T) {
	mapper := NewIDMapper()

	// Create first ID
	id1, err := mapper.GetOrCreate("doc1")
	if err != nil {
		t.Fatalf("GetOrCreate failed: %v", err)
	}
	if id1 != 0 {
		t.Errorf("Expected first ID = 0, got %d", id1)
	}

	// Create second ID
	id2, err := mapper.GetOrCreate("doc2")
	if err != nil {
		t.Fatalf("GetOrCreate failed: %v", err)
	}
	if id2 != 1 {
		t.Errorf("Expected second ID = 1, got %d", id2)
	}

	// Get existing ID
	id1Again, err := mapper.GetOrCreate("doc1")
	if err != nil {
		t.Fatalf("GetOrCreate failed: %v", err)
	}
	if id1Again != id1 {
		t.Errorf("Expected same ID for 'doc1', got %d (original: %d)", id1Again, id1)
	}

	if mapper.Count() != 2 {
		t.Errorf("Expected Count() = 2, got %d", mapper.Count())
	}
}

// TestToStringID tests uint32 → string translation
func TestToStringID(t *testing.T) {
	mapper := NewIDMapper()

	// Create IDs
	mapper.GetOrCreate("doc1")
	mapper.GetOrCreate("doc2")

	// Translate back
	str, err := mapper.ToStringID(0)
	if err != nil {
		t.Fatalf("ToStringID failed: %v", err)
	}
	if str != "doc1" {
		t.Errorf("Expected 'doc1', got '%s'", str)
	}

	str, err = mapper.ToStringID(1)
	if err != nil {
		t.Fatalf("ToStringID failed: %v", err)
	}
	if str != "doc2" {
		t.Errorf("Expected 'doc2', got '%s'", str)
	}

	// Non-existent ID
	_, err = mapper.ToStringID(999)
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

// TestToUint32ID tests string → uint32 translation
func TestToUint32ID(t *testing.T) {
	mapper := NewIDMapper()

	// Create IDs
	mapper.GetOrCreate("doc1")
	mapper.GetOrCreate("doc2")

	// Translate string to uint32
	id, err := mapper.ToUint32ID("doc1")
	if err != nil {
		t.Fatalf("ToUint32ID failed: %v", err)
	}
	if id != 0 {
		t.Errorf("Expected ID = 0, got %d", id)
	}

	id, err = mapper.ToUint32ID("doc2")
	if err != nil {
		t.Fatalf("ToUint32ID failed: %v", err)
	}
	if id != 1 {
		t.Errorf("Expected ID = 1, got %d", id)
	}

	// Non-existent string ID
	_, err = mapper.ToUint32ID("nonexistent")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

// TestDelete tests deletion and tombstone behavior
func TestDelete(t *testing.T) {
	mapper := NewIDMapper()

	// Create IDs
	id1, _ := mapper.GetOrCreate("doc1")
	mapper.GetOrCreate("doc2")

	// Delete doc1
	err := mapper.Delete("doc1")
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Verify count decreased
	if mapper.Count() != 1 {
		t.Errorf("Expected Count() = 1 after delete, got %d", mapper.Count())
	}

	// Verify ToStringID fails
	_, err = mapper.ToStringID(id1)
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound for deleted ID, got %v", err)
	}

	// Verify ToUint32ID fails
	_, err = mapper.ToUint32ID("doc1")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound for deleted string ID, got %v", err)
	}

	// Verify tombstone is marked
	if !mapper.IsTombstone(id1) {
		t.Error("Expected ID to be tombstoned")
	}

	// Delete non-existent ID
	err = mapper.Delete("nonexistent")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound when deleting non-existent ID, got %v", err)
	}
}

// TestTombstoneNoRecycle tests that deleted IDs are never recycled
func TestTombstoneNoRecycle(t *testing.T) {
	mapper := NewIDMapper()

	// Create and delete doc1 (gets ID 0)
	id1, _ := mapper.GetOrCreate("doc1")
	mapper.Delete("doc1")

	// Create new doc (should get ID 1, not reuse ID 0)
	id2, _ := mapper.GetOrCreate("doc2")

	if id2 == id1 {
		t.Errorf("ID recycled! doc2 got ID %d (same as deleted doc1)", id2)
	}

	if id2 != 1 {
		t.Errorf("Expected doc2 to get ID 1, got %d", id2)
	}

	// Verify ID 0 is still tombstoned
	if !mapper.IsTombstone(id1) {
		t.Error("Tombstone was cleared (should never be cleared)")
	}
}

// TestEdgeCases_EmptyString tests empty string ID
func TestEdgeCases_EmptyString(t *testing.T) {
	mapper := NewIDMapper()

	// Empty string is valid
	id, err := mapper.GetOrCreate("")
	if err != nil {
		t.Fatalf("GetOrCreate with empty string failed: %v", err)
	}

	// Verify round-trip
	str, err := mapper.ToStringID(id)
	if err != nil {
		t.Fatalf("ToStringID failed: %v", err)
	}
	if str != "" {
		t.Errorf("Expected empty string, got '%s'", str)
	}
}

// TestEdgeCases_LongString tests very long string ID
func TestEdgeCases_LongString(t *testing.T) {
	mapper := NewIDMapper()

	longID := string(make([]byte, 10000))
	for i := range longID {
		longID = longID[:i] + "a" + longID[i+1:]
	}

	id, err := mapper.GetOrCreate(longID)
	if err != nil {
		t.Fatalf("GetOrCreate with long string failed: %v", err)
	}

	// Verify round-trip
	str, err := mapper.ToStringID(id)
	if err != nil {
		t.Fatalf("ToStringID failed: %v", err)
	}
	if str != longID {
		t.Error("Long string round-trip failed")
	}
}

// TestEdgeCases_SpecialCharacters tests special characters
func TestEdgeCases_SpecialCharacters(t *testing.T) {
	mapper := NewIDMapper()

	testCases := []string{
		"doc:123",
		"user/profile/456",
		"data\\\\path",
		"unicode-日本語",
		"emoji-🎉",
		"newline\n\ttab",
		"quotes'\"mixed",
	}

	for _, testID := range testCases {
		id, err := mapper.GetOrCreate(testID)
		if err != nil {
			t.Fatalf("GetOrCreate failed for '%s': %v", testID, err)
		}

		// Verify round-trip
		str, err := mapper.ToStringID(id)
		if err != nil {
			t.Fatalf("ToStringID failed for '%s': %v", testID, err)
		}
		if str != testID {
			t.Errorf("Round-trip failed: expected '%s', got '%s'", testID, str)
		}
	}
}

// TestConcurrency_ParallelGetOrCreate tests concurrent ID creation
func TestConcurrency_ParallelGetOrCreate(t *testing.T) {
	mapper := NewIDMapper()

	const numGoroutines = 100
	const numIDsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Each goroutine creates unique IDs
	for i := 0; i < numGoroutines; i++ {
		go func(goroutineID int) {
			defer wg.Done()
			for j := 0; j < numIDsPerGoroutine; j++ {
				stringID := fmt.Sprintf("doc_%d_%d", goroutineID, j)
				_, err := mapper.GetOrCreate(stringID)
				if err != nil {
					t.Errorf("GetOrCreate failed: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	expectedCount := numGoroutines * numIDsPerGoroutine
	if mapper.Count() != expectedCount {
		t.Errorf("Expected Count() = %d, got %d", expectedCount, mapper.Count())
	}
}

// TestConcurrency_SameID tests concurrent creation of same ID
func TestConcurrency_SameID(t *testing.T) {
	mapper := NewIDMapper()

	const numGoroutines = 100
	ids := make([]uint32, numGoroutines)

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// All goroutines try to create the same ID
	for i := 0; i < numGoroutines; i++ {
		go func(index int) {
			defer wg.Done()
			id, err := mapper.GetOrCreate("same_doc")
			if err != nil {
				t.Errorf("GetOrCreate failed: %v", err)
			}
			ids[index] = id
		}(i)
	}

	wg.Wait()

	// All goroutines should get the same ID
	firstID := ids[0]
	for i, id := range ids {
		if id != firstID {
			t.Errorf("Goroutine %d got different ID: %d (expected %d)", i, id, firstID)
		}
	}

	// Should only have one ID mapping
	if mapper.Count() != 1 {
		t.Errorf("Expected Count() = 1, got %d", mapper.Count())
	}
}

// TestIDExhaustion tests uint32 overflow handling
func TestIDExhaustion(t *testing.T) {
	mapper := NewIDMapper()

	// Manually set nextID to near max
	mapper.nextID = ^uint32(0) // Max uint32 value

	// Next creation should fail
	_, err := mapper.GetOrCreate("overflow_doc")
	if err != ErrIDExhausted {
		t.Errorf("Expected ErrIDExhausted, got %v", err)
	}

	// Count should still be 0
	if mapper.Count() != 0 {
		t.Errorf("Expected Count() = 0 after exhaustion, got %d", mapper.Count())
	}
}

// TestIsTombstone tests tombstone checking
func TestIsTombstone(t *testing.T) {
	mapper := NewIDMapper()

	id1, _ := mapper.GetOrCreate("doc1")
	id2, _ := mapper.GetOrCreate("doc2")

	// Initially not tombstoned
	if mapper.IsTombstone(id1) {
		t.Error("Expected ID 1 to not be tombstoned initially")
	}

	// Delete doc1
	mapper.Delete("doc1")

	// Now tombstoned
	if !mapper.IsTombstone(id1) {
		t.Error("Expected ID 1 to be tombstoned after delete")
	}

	// doc2 still not tombstoned
	if mapper.IsTombstone(id2) {
		t.Error("Expected ID 2 to not be tombstoned")
	}

	// Non-existent ID is not tombstoned
	if mapper.IsTombstone(999) {
		t.Error("Non-existent ID should not be tombstoned")
	}
}

// TestClear tests that Clear resets all mappings, tombstones, and the ID counter
func TestClear(t *testing.T) {
	mapper := NewIDMapper()

	id1, _ := mapper.GetOrCreate("doc1")
	mapper.GetOrCreate("doc2")
	mapper.Delete("doc1")

	if mapper.Count() != 1 {
		t.Fatalf("expected Count() = 1 before Clear, got %d", mapper.Count())
	}

	mapper.Clear()

	if mapper.Count() != 0 {
		t.Errorf("expected Count() = 0 after Clear, got %d", mapper.Count())
	}

	if mapper.NextID() != 0 {
		t.Errorf("expected NextID() = 0 after Clear, got %d", mapper.NextID())
	}

	if mapper.IsTombstone(id1) {
		t.Error("expected tombstones to be cleared after Clear")
	}

	// A fresh mapping after Clear should reuse ID 0 -- this is safe because
	// Clear implies the entire collection was wiped alongside it, so there is
	// no live data left for the recycled ID to collide with.
	newID, err := mapper.GetOrCreate("doc3")
	if err != nil {
		t.Fatalf("GetOrCreate after Clear failed: %v", err)
	}
	if newID != 0 {
		t.Errorf("expected first ID after Clear to be 0, got %d", newID)
	}
}

// TestPersistence_GOBEncoding tests GOB serialization
func TestPersistence_GOBEncoding(t *testing.T) {
	mapper := NewIDMapper()

	// Create some IDs
	mapper.GetOrCreate("doc1")
	mapper.GetOrCreate("doc2")
	mapper.GetOrCreate("doc3")

	// Delete one
	mapper.Delete("doc2")

	// Encode
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := mapper.EncodeGOB(encoder)
	if err != nil {
		t.Fatalf("EncodeGOB failed: %v", err)
	}

	// Create new mapper and decode
	mapper2 := NewIDMapper()
	decoder := gob.NewDecoder(&buf)
	err = mapper2.DecodeGOB(decoder)
	if err != nil {
		t.Fatalf("DecodeGOB failed: %v", err)
	}

	// Verify state matches
	if mapper2.Count() != mapper.Count() {
		t.Errorf("Count mismatch: original %d, decoded %d", mapper.Count(), mapper2.Count())
	}

	if mapper2.NextID() != mapper.NextID() {
		t.Errorf("NextID mismatch: original %d, decoded %d", mapper.NextID(), mapper2.NextID())
	}

	// Verify specific mappings
	id1, _ := mapper2.ToUint32ID("doc1")
	if id1 != 0 {
		t.Errorf("Expected doc1 → 0, got %d", id1)
	}

	id3, _ := mapper2.ToUint32ID("doc3")
	if id3 != 2 {
		t.Errorf("Expected doc3 → 2, got %d", id3)
	}

	// Verify doc2 is deleted
	_, err = mapper2.ToUint32ID("doc2")
	if err != ErrNotFound {
		t.Error("Expected doc2 to be deleted")
	}

	// Verify tombstone
	if !mapper2.IsTombstone(1) {
		t.Error("Expected ID 1 to be tombstoned")
	}
}

// TestPersistence_RoundTrip tests multiple encode/decode cycles
func TestPersistence_RoundTrip(t *testing.T) {
	mapper1 := NewIDMapper()

	// Create 100 IDs
	for i := 0; i < 100; i++ {
		mapper1.GetOrCreate(fmt.Sprintf("doc%d", i))
	}

	// Delete every 3rd ID
	for i := 0; i < 100; i += 3 {
		mapper1.Delete(fmt.Sprintf("doc%d", i))
	}

	// Encode
	var buf1 bytes.Buffer
	encoder1 := gob.NewEncoder(&buf1)
	mapper1.EncodeGOB(encoder1)

	// Decode to mapper2
	mapper2 := NewIDMapper()
	decoder1 := gob.NewDecoder(&buf1)
	mapper2.DecodeGOB(decoder1)

	// Encode mapper2
	var buf2 bytes.Buffer
	encoder2 := gob.NewEncoder(&buf2)
	mapper2.EncodeGOB(encoder2)

	// Decode to mapper3
	mapper3 := NewIDMapper()
	decoder2 := gob.NewDecoder(&buf2)
	mapper3.DecodeGOB(decoder2)

	// Verify all three have same state
	if mapper3.Count() != mapper1.Count() {
		t.Errorf("Count mismatch after round-trip: %d vs %d", mapper3.Count(), mapper1.Count())
	}

	if mapper3.NextID() != mapper1.NextID() {
		t.Errorf("NextID mismatch after round-trip: %d vs %d", mapper3.NextID(), mapper1.NextID())
	}
}

// TestPersistence_EmptyMapper tests encoding/decoding empty mapper
func TestPersistence_EmptyMapper(t *testing.T) {
	mapper := NewIDMapper()

	// Encode empty mapper
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := mapper.EncodeGOB(encoder)
	if err != nil {
		t.Fatalf("EncodeGOB failed: %v", err)
	}

	// Decode
	mapper2 := NewIDMapper()
	decoder := gob.NewDecoder(&buf)
	err = mapper2.DecodeGOB(decoder)
	if err != nil {
		t.Fatalf("DecodeGOB failed: %v", err)
	}

	// Verify empty state
	if mapper2.Count() != 0 {
		t.Errorf("Expected Count() = 0, got %d", mapper2.Count())
	}

	if mapper2.NextID() != 0 {
		t.Errorf("Expected NextID() = 0, got %d", mapper2.NextID())
	}
}
