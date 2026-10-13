package convert_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/Pradyothsp/govec/internal/grpcserver/convert"
)

// Metadata crosses the wire as protobuf Values and must come back as the same
// map. JSON-shaped values only: numbers are float64 on the way back, as they
// are over REST.
func TestMetadata_RoundTrip(t *testing.T) {
	tests := []struct {
		name string
		meta map[string]any
	}{
		{"nil stays nil", nil},
		{"scalars", map[string]any{"str": "hello", "num": 42.5, "bool": true, "null": nil}},
		{"nested map", map[string]any{"nested": map[string]any{"x": 1.0, "deeper": map[string]any{"y": "z"}}}},
		{"list", map[string]any{"items": []any{1.0, "two", true}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			proto, err := convert.MetaToProto(tt.meta)
			back := convert.ProtoToMeta(proto)

			// Assert
			require.NoError(t, err)
			assert.Equal(t, tt.meta, back)
		})
	}
}

// Metadata stored from REST or the engine can hold any Go number type; every
// one becomes a protobuf number.
func TestMetaToProto_EveryNumericTypeIsANumber(t *testing.T) {
	// Arrange
	in := map[string]any{
		"float32": float32(1.5), "float64": 2.5, "int": 3, "int32": int32(4),
		"int64": int64(5), "uint": uint(6), "uint32": uint32(7), "uint64": uint64(8),
	}

	// Act
	out, err := convert.MetaToProto(in)

	// Assert
	require.NoError(t, err)
	require.Len(t, out, len(in))
	for k, v := range out {
		assert.IsType(t, &structpb.Value_NumberValue{}, v.Kind, k)
	}
}

func TestMetaToProto_UnsupportedType_Rejected(t *testing.T) {
	// Act
	_, err := convert.MetaToProto(map[string]any{"bad": struct{ x int }{x: 1}})

	// Assert
	assert.Error(t, err)
}
