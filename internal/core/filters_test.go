package core

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatchFilter(t *testing.T) {
	tests := []struct {
		name     string
		docMeta  map[string]interface{}
		filters  map[string]interface{}
		expected bool
	}{
		{
			name:     "empty_filter_matches_any",
			docMeta:  map[string]interface{}{"a": 1},
			filters:  map[string]interface{}{},
			expected: true,
		},
		{
			name:     "empty_filter_matches_nil_meta",
			docMeta:  nil,
			filters:  map[string]interface{}{},
			expected: true,
		},
		{
			name:     "key_missing_returns_false",
			docMeta:  map[string]interface{}{"a": 1},
			filters:  map[string]interface{}{"b": 1},
			expected: false,
		},
		{
			name:     "value_mismatch_returns_false",
			docMeta:  map[string]interface{}{"a": 1},
			filters:  map[string]interface{}{"a": 2},
			expected: false,
		},
		{
			name:     "exact_string_match",
			docMeta:  map[string]interface{}{"k": "v"},
			filters:  map[string]interface{}{"k": "v"},
			expected: true,
		},
		{
			name:     "bool_match",
			docMeta:  map[string]interface{}{"active": true},
			filters:  map[string]interface{}{"active": true},
			expected: true,
		},
		{
			name:     "bool_mismatch",
			docMeta:  map[string]interface{}{"active": true},
			filters:  map[string]interface{}{"active": false},
			expected: false,
		},
		{
			name:     "all_filters_must_match",
			docMeta:  map[string]interface{}{"a": 1, "b": 2},
			filters:  map[string]interface{}{"a": 1, "b": 9},
			expected: false,
		},
		{
			name:     "nil_meta_non_empty_filter",
			docMeta:  nil,
			filters:  map[string]interface{}{"k": "v"},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MatchFilter(tt.docMeta, tt.filters)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestValuesEqual(t *testing.T) {
	tests := []struct {
		name     string
		a        interface{}
		b        interface{}
		expected bool
	}{
		{
			name:     "identical_strings",
			a:        "x",
			b:        "x",
			expected: true,
		},
		{
			name:     "different_strings",
			a:        "x",
			b:        "y",
			expected: false,
		},
		{
			name:     "int_vs_float64_same_value",
			a:        int(42),
			b:        float64(42),
			expected: true,
		},
		{
			name:     "float32_vs_float64_same_value",
			a:        float32(3.0),
			b:        float64(3.0),
			expected: true,
		},
		{
			name:     "int64_vs_float64_same_value",
			a:        int64(100),
			b:        float64(100),
			expected: true,
		},
		{
			name:     "numeric_values_different",
			a:        int(1),
			b:        float64(2),
			expected: false,
		},
		{
			name:     "bool_true_match",
			a:        true,
			b:        true,
			expected: true,
		},
		{
			name:     "bool_mismatch",
			a:        true,
			b:        false,
			expected: false,
		},
		{
			name:     "nil_vs_nil",
			a:        nil,
			b:        nil,
			expected: true,
		},
		{
			name:     "string_vs_int",
			a:        "42",
			b:        int(42),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := valuesEqual(tt.a, tt.b)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestToFloat(t *testing.T) {
	tests := []struct {
		name      string
		input     interface{}
		wantOk    bool
		wantValue float64
	}{
		{name: "int", input: int(5), wantOk: true, wantValue: 5.0},
		{name: "int8", input: int8(5), wantOk: true, wantValue: 5.0},
		{name: "int16", input: int16(5), wantOk: true, wantValue: 5.0},
		{name: "int32", input: int32(5), wantOk: true, wantValue: 5.0},
		{name: "int64", input: int64(5), wantOk: true, wantValue: 5.0},
		{name: "float32", input: float32(3.14), wantOk: true, wantValue: float64(float32(3.14))},
		{name: "float64", input: float64(2.71), wantOk: true, wantValue: 2.71},
		{name: "string", input: "5", wantOk: false},
		{name: "bool", input: true, wantOk: false},
		{name: "nil", input: nil, wantOk: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := toFloat(tt.input)
			assert.Equal(t, tt.wantOk, ok)
			if tt.wantOk {
				assert.InDelta(t, tt.wantValue, got, math.Abs(tt.wantValue)*1e-6+1e-9,
					"toFloat(%v) = %v, want %v", tt.input, got, tt.wantValue)
			}
		})
	}
}
