package convert

import (
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/core"
)

// ProtoToSparse converts a protobuf SparseVector to a core.SparseVector.
// Returns an error if indices and values have different lengths (and both are non-nil).
func ProtoToSparse(in *pb.SparseVector) (core.SparseVector, error) {
	if in == nil {
		return core.SparseVector{}, nil
	}
	if len(in.Indices) != len(in.Values) {
		return core.SparseVector{}, status.Errorf(
			codes.InvalidArgument,
			"sparse vector: indices length %d != values length %d",
			len(in.Indices), len(in.Values),
		)
	}
	return core.SparseVector{
		Indices: in.Indices,
		Values:  in.Values,
	}, nil
}

// SparseToProto converts a core.SparseVector to a protobuf SparseVector.
// Returns nil for a nil or empty vector to avoid sending empty messages on the
// wire, which is how a dense-only record reports having no sparse component.
//
// Takes a pointer to match VectorRecord.SparseVector, where nil already carries
// that meaning; an empty struct is still accepted and treated the same way.
func SparseToProto(sv *core.SparseVector) *pb.SparseVector {
	if sv == nil || sv.IsEmpty() {
		return nil
	}
	return &pb.SparseVector{
		Indices: sv.Indices,
		Values:  sv.Values,
	}
}
