package convert_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/Pradyothsp/govec/internal/grpcserver/convert"
)

func TestProtoToMeta_Nil(t *testing.T) {
	result := convert.ProtoToMeta(nil)
	assert.Nil(t, result)
}

func TestProtoToMeta_AllScalarTypes(t *testing.T) {
	in := map[string]*structpb.Value{
		"str":  structpb.NewStringValue("hello"),
		"num":  structpb.NewNumberValue(42.5),
		"bool": structpb.NewBoolValue(true),
		"null": structpb.NewNullValue(),
	}
	out := convert.ProtoToMeta(in)

	assert.Equal(t, "hello", out["str"])
	assert.Equal(t, 42.5, out["num"])
	assert.Equal(t, true, out["bool"])
	assert.Nil(t, out["null"])
}

func TestProtoToMeta_NestedMap(t *testing.T) {
	inner, _ := structpb.NewStruct(map[string]interface{}{"x": 1.0})
	in := map[string]*structpb.Value{
		"nested": structpb.NewStructValue(inner),
	}
	out := convert.ProtoToMeta(in)

	nested, ok := out["nested"].(map[string]interface{})
	require.True(t, ok, "nested should be a map")
	assert.Equal(t, 1.0, nested["x"])
}

func TestProtoToMeta_List(t *testing.T) {
	list, _ := structpb.NewList([]interface{}{1.0, "two", true})
	in := map[string]*structpb.Value{
		"items": structpb.NewListValue(list),
	}
	out := convert.ProtoToMeta(in)

	items, ok := out["items"].([]interface{})
	require.True(t, ok, "items should be a slice")
	assert.Len(t, items, 3)
}

func TestMetaToProto_Nil(t *testing.T) {
	result, err := convert.MetaToProto(nil)
	require.NoError(t, err)
	assert.Nil(t, result)
}

func TestMetaToProto_AllNumericTypes(t *testing.T) {
	in := map[string]interface{}{
		"float32": float32(1.5),
		"float64": float64(2.5),
		"int":     int(3),
		"int32":   int32(4),
		"int64":   int64(5),
		"uint":    uint(6),
		"uint32":  uint32(7),
		"uint64":  uint64(8),
	}
	out, err := convert.MetaToProto(in)
	require.NoError(t, err)
	assert.Len(t, out, 8)
	for k, v := range out {
		_, ok := v.Kind.(*structpb.Value_NumberValue)
		assert.True(t, ok, "key %q should be a number", k)
	}
}

func TestMetaToProto_UnsupportedType(t *testing.T) {
	in := map[string]interface{}{
		"bad": struct{ x int }{x: 1},
	}
	_, err := convert.MetaToProto(in)
	require.Error(t, err)
}

func TestMetaRoundTrip(t *testing.T) {
	original := map[string]interface{}{
		"name":   "test",
		"score":  float64(0.95),
		"active": true,
		"tags":   []interface{}{"a", "b"},
	}

	proto, err := convert.MetaToProto(original)
	require.NoError(t, err)

	back := convert.ProtoToMeta(proto)

	assert.Equal(t, original["name"], back["name"])
	assert.Equal(t, original["score"], back["score"])
	assert.Equal(t, original["active"], back["active"])

	tags, ok := back["tags"].([]interface{})
	require.True(t, ok)
	assert.Equal(t, []interface{}{"a", "b"}, tags)
}
