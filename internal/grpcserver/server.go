// Package grpcserver provides the gRPC transport layer for GoVec.
// Package name is grpcserver (not grpc) to avoid collision with google.golang.org/grpc.
package grpcserver

import (
	"context"
	"errors"
	"io"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/grpcserver/convert"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/release"
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
	sparse := convert.ProtoToSparse(req.Sparse)
	meta := convert.ProtoToMeta(req.Metadata)
	if err := s.engine.Insert(ctx, req.Id, req.Vector, sparse, meta); err != nil {
		// A mis-sized vector is the caller's mistake, not a server fault.
		if index.IsInvalidVectorError(err) {
			return nil, status.Errorf(codes.InvalidArgument, "insert failed: %v", err)
		}
		return nil, status.Errorf(codes.Internal, "insert failed: %v", err)
	}
	// "inserted", not "ok", to match what REST answers for the same call.
	// Picking a protocol is meant to change the wire format and nothing else,
	// and a caller testing `status == "ok"` used to pass over gRPC and fail
	// over REST. See statusStrings in server_test.go, which pins all four.
	return &pb.InsertResponse{Status: "inserted"}, nil
}

// batchChunk bounds how many streamed items are held before they go to the
// engine. A stream has no overall size limit, unlike a REST request body.
const batchChunk = 1000

// BatchInsert handles a client-streaming batch of insert requests. Items go
// to the engine as batches of up to batchChunk, each one WAL fsync, as a REST
// batch does: inserting them one by one cost an fsync per vector.
func (s *GoVecServer) BatchInsert(stream pb.GoVecService_BatchInsertServer) error {
	ctx := stream.Context()
	var insertedCount int32
	var batchErrors []*pb.BatchError
	items := make([]index.BatchInsertItem, 0, batchChunk)

	flush := func() error {
		if len(items) == 0 {
			return nil
		}
		failures, err := s.engine.BatchInsert(ctx, items)
		if err != nil {
			return status.Errorf(codes.Internal, "batch insert failed: %v", err)
		}
		for _, f := range failures {
			batchErrors = append(batchErrors, &pb.BatchError{Id: f.ID, Error: f.Err.Error()})
		}
		insertedCount += int32(len(items) - len(failures)) //nolint:gosec // at most batchChunk
		items = items[:0]
		return nil
	}

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return status.Errorf(codes.Internal, "stream recv error: %v", err)
		}

		items = append(items, index.BatchInsertItem{
			ID:     req.Id,
			Vector: req.Vector,
			Sparse: convert.ProtoToSparse(req.Sparse),
			Meta:   convert.ProtoToMeta(req.Metadata),
		})
		if len(items) == batchChunk {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	if err := flush(); err != nil {
		return err
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
	sparse := convert.ProtoToSparse(req.SparseQuery)
	filters := convert.ProtoToMeta(req.Filters)
	results, err := s.engine.Search(ctx, req.QueryVector, sparse, int(req.K), filters)
	if err != nil {
		// A query the index can't compare is the caller's mistake.
		if index.IsInvalidVectorError(err) {
			return nil, status.Errorf(codes.InvalidArgument, "search failed: %v", err)
		}
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
	return &pb.DeleteResponse{Status: "deleted", Id: req.Id}, nil
}

// GetByID returns the full stored record for a vector, mirroring REST's
// GET /api/v1/vectors/:id.
func (s *GoVecServer) GetByID(ctx context.Context, req *pb.GetByIDRequest) (*pb.GetByIDResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}

	record, err := s.engine.GetByID(ctx, req.Id)
	if err != nil {
		if errors.Is(err, core.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "vector %q not found", req.Id)
		}
		return nil, status.Errorf(codes.Internal, "get by id failed: %v", err)
	}

	// Unlike Search, which drops unconvertible metadata keys to keep a result
	// set flowing, a single-record fetch has no reason to hand back a record
	// that quietly lost fields.
	meta, err := convert.MetaToProto(record.Metadata)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "metadata conversion failed: %v", err)
	}

	return &pb.GetByIDResponse{
		Id:       record.ID,
		Vector:   record.Vector,
		Sparse:   convert.SparseToProto(record.SparseVector),
		Metadata: meta,
	}, nil
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
		EnableMmap:     info.EnableMmap,
		Version:        release.Version,
	}, nil
}

// Flush persists the current index snapshot to disk.
func (s *GoVecServer) Flush(ctx context.Context, _ *pb.FlushRequest) (*pb.FlushResponse, error) {
	if err := s.engine.SaveToFile(ctx, s.dataPath); err != nil {
		return nil, status.Errorf(codes.Internal, "flush failed: %v", err)
	}
	return &pb.FlushResponse{Status: "flushed"}, nil
}

// Reset clears every vector from the index, mirroring REST's
// POST /api/v1/admin/reset. Like the REST endpoint it only clears memory and
// the WAL -- call Flush afterward if the empty state should survive a restart.
func (s *GoVecServer) Reset(_ context.Context, _ *pb.ResetRequest) (*pb.ResetResponse, error) {
	s.engine.Clear()
	return &pb.ResetResponse{Status: "reset"}, nil
}

// Health returns a static "ok" response — no engine interaction required.
func (s *GoVecServer) Health(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{Status: "ok"}, nil
}
