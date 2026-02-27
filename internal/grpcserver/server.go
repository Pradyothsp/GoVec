// Package grpcserver provides the gRPC transport layer for GoVec.
// Package name is grpcserver (not grpc) to avoid collision with google.golang.org/grpc.
package grpcserver

import (
	"context"
	"io"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/grpcserver/convert"
	"github.com/Pradyothsp/govec/internal/index"
)

// GoVecServer implements pb.GoVecServiceServer by delegating to an index.Engine.
type GoVecServer struct {
	engine   index.Engine
	dataPath string
}

// NewGoVecServer creates a new GoVecServer backed by the provided engine.
func NewGoVecServer(engine index.Engine, dataPath string) *GoVecServer {
	return &GoVecServer{engine: engine, dataPath: dataPath}
}

// Insert handles a single vector insert.
func (s *GoVecServer) Insert(ctx context.Context, req *pb.InsertRequest) (*pb.InsertResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}
	if len(req.Vector) == 0 {
		return nil, status.Error(codes.InvalidArgument, "vector is required")
	}
	sparse, err := convert.ProtoToSparse(req.Sparse)
	if err != nil {
		return nil, err
	}
	meta := convert.ProtoToMeta(req.Metadata)
	if err := s.engine.Insert(ctx, req.Id, req.Vector, sparse, meta); err != nil {
		return nil, status.Errorf(codes.Internal, "insert failed: %v", err)
	}
	return &pb.InsertResponse{Status: "ok"}, nil
}

// BatchInsert handles a client-streaming batch of insert requests.
func (s *GoVecServer) BatchInsert(stream pb.GoVecService_BatchInsertServer) error {
	ctx := stream.Context()
	var insertedCount int32
	var batchErrors []*pb.BatchError

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return status.Errorf(codes.Internal, "stream recv error: %v", err)
		}

		sparse, err := convert.ProtoToSparse(req.Sparse)
		if err != nil {
			batchErrors = append(batchErrors, &pb.BatchError{Id: req.Id, Error: err.Error()})
			continue
		}
		meta := convert.ProtoToMeta(req.Metadata)
		if err := s.engine.Insert(ctx, req.Id, req.Vector, sparse, meta); err != nil {
			batchErrors = append(batchErrors, &pb.BatchError{Id: req.Id, Error: err.Error()})
			continue
		}
		insertedCount++
	}

	return stream.SendAndClose(&pb.BatchInsertResponse{
		InsertedCount: insertedCount,
		Errors:        batchErrors,
	})
}

// Search finds the k nearest neighbors to the query vector.
func (s *GoVecServer) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	if len(req.QueryVector) == 0 {
		return nil, status.Error(codes.InvalidArgument, "query_vector is required")
	}
	if req.K <= 0 {
		return nil, status.Error(codes.InvalidArgument, "k must be positive")
	}
	sparse, err := convert.ProtoToSparse(req.SparseQuery)
	if err != nil {
		return nil, err
	}
	filters := convert.ProtoToMeta(req.Filters)
	results, err := s.engine.Search(ctx, req.QueryVector, sparse, int(req.K), filters)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "search failed: %v", err)
	}
	return &pb.SearchResponse{Results: convert.SearchResultsToProto(results)}, nil
}

// Delete removes a vector by ID.
func (s *GoVecServer) Delete(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteResponse, error) {
	found, err := s.engine.Delete(ctx, req.Id)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "delete failed: %v", err)
	}
	if !found {
		return nil, status.Errorf(codes.NotFound, "vector %q not found", req.Id)
	}
	return &pb.DeleteResponse{Status: "ok", Id: req.Id}, nil
}

// Stats returns the current vector count.
func (s *GoVecServer) Stats(_ context.Context, _ *pb.StatsRequest) (*pb.StatsResponse, error) {
	return &pb.StatsResponse{VectorCount: int32(s.engine.Len())}, nil //nolint:gosec // vector count fits in int32
}

// Info returns engine configuration details.
func (s *GoVecServer) Info(_ context.Context, _ *pb.InfoRequest) (*pb.InfoResponse, error) {
	info := s.engine.Info()
	return &pb.InfoResponse{
		Quantization:   info.Quantization,
		IndexType:      info.IndexType,
		DistanceMetric: info.DistanceMetric,
		Dimensions:     int32(info.Dimensions),  //nolint:gosec // dimension fits in int32
		VectorCount:    int32(info.VectorCount), //nolint:gosec // vector count fits in int32
	}, nil
}

// Flush persists the current index snapshot to disk.
func (s *GoVecServer) Flush(ctx context.Context, _ *pb.FlushRequest) (*pb.FlushResponse, error) {
	if err := s.engine.SaveToFile(ctx, s.dataPath); err != nil {
		return nil, status.Errorf(codes.Internal, "flush failed: %v", err)
	}
	return &pb.FlushResponse{Status: "ok"}, nil
}

// Health returns a static "ok" response — no engine interaction required.
func (s *GoVecServer) Health(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{Status: "ok"}, nil
}
