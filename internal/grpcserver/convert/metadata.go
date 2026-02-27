// Package convert provides type conversion between protobuf types and Go domain types.
package convert

import (
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

// ProtoToMeta converts a protobuf map[string]*structpb.Value to a Go map[string]interface{}.
// All numeric types are normalized to float64, consistent with the JSON/HTTP transport path.
func ProtoToMeta(in map[string]*structpb.Value) map[string]interface{} {
	if in == nil {
		return nil
	}
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = protoValueToInterface(v)
	}
	return out
}

// MetaToProto converts a Go map[string]interface{} to a protobuf map[string]*structpb.Value.
// Returns codes.InvalidArgument if an unsupported type is encountered.
func MetaToProto(in map[string]interface{}) (map[string]*structpb.Value, error) {
	if in == nil {
		return nil, nil
	}
	out := make(map[string]*structpb.Value, len(in))
	for k, v := range in {
		pbVal, err := interfaceToProtoValue(v)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "metadata key %q: %v", k, err)
		}
		out[k] = pbVal
	}
	return out, nil
}

// protoValueToInterface recursively converts a *structpb.Value to a Go native type.
// Numbers are always float64 to match JSON unmarshal behaviour.
func protoValueToInterface(v *structpb.Value) interface{} {
	if v == nil {
		return nil
	}
	return v.AsInterface()
}

// interfaceToProtoValue converts a Go value to a *structpb.Value.
// Supported types: nil, bool, string, all integer/float variants, []interface{}, map[string]interface{}.
func interfaceToProtoValue(v interface{}) (*structpb.Value, error) {
	switch t := v.(type) {
	case nil:
		return structpb.NewNullValue(), nil
	case bool:
		return structpb.NewBoolValue(t), nil
	case string:
		return structpb.NewStringValue(t), nil
	case float32:
		return structpb.NewNumberValue(float64(t)), nil
	case float64:
		return structpb.NewNumberValue(t), nil
	case int:
		return structpb.NewNumberValue(float64(t)), nil
	case int32:
		return structpb.NewNumberValue(float64(t)), nil
	case int64:
		return structpb.NewNumberValue(float64(t)), nil
	case uint:
		return structpb.NewNumberValue(float64(t)), nil
	case uint32:
		return structpb.NewNumberValue(float64(t)), nil
	case uint64:
		return structpb.NewNumberValue(float64(t)), nil
	case []interface{}:
		items := make([]*structpb.Value, len(t))
		for i, elem := range t {
			pbElem, err := interfaceToProtoValue(elem)
			if err != nil {
				return nil, fmt.Errorf("list[%d]: %w", i, err)
			}
			items[i] = pbElem
		}
		return structpb.NewListValue(&structpb.ListValue{Values: items}), nil
	case map[string]interface{}:
		fields := make(map[string]*structpb.Value, len(t))
		for k, val := range t {
			pbVal, err := interfaceToProtoValue(val)
			if err != nil {
				return nil, fmt.Errorf("map[%q]: %w", k, err)
			}
			fields[k] = pbVal
		}
		return structpb.NewStructValue(&structpb.Struct{Fields: fields}), nil
	default:
		return nil, fmt.Errorf("unsupported type %T", v)
	}
}
