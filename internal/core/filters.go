package core

import (
	"reflect"
)

// MatchFilter checks if the document matches the user's filter criteria
func MatchFilter(docMeta, filters map[string]interface{}) bool {
	for filterKey, filterVal := range filters {
		// 1. Check if the key exists in the document
		docVal, exists := docMeta[filterKey]
		if !exists {
			return false
		}

		// 2. Compare the values (handling types smartly)
		if !valuesEqual(docVal, filterVal) {
			return false
		}
	}
	return true
}

// valuesEqual handles the comparison of interface{} types safely
func valuesEqual(a, b interface{}) bool {
	// If they are exactly the same type and value, easy match
	if a == b {
		return true
	}

	// Handle the "Number Problem" (JSON numbers are float64)
	// We convert everything to float64 for comparison if possible
	valA, okA := toFloat(a)
	valB, okB := toFloat(b)
	if okA && okB {
		return valA == valB
	}

	// Handle standard types (DeepEqual helps with slices/maps if you ever need them)
	return reflect.DeepEqual(a, b)
}

// Helper to force numbers into float64
func toFloat(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case int:
		return float64(val), true
	case int8:
		return float64(val), true
	case int16:
		return float64(val), true
	case int32:
		return float64(val), true
	case int64:
		return float64(val), true
	case float32:
		return float64(val), true
	case float64:
		return val, true
	}
	return 0, false
}
