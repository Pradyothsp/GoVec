package grpcserver_test

import (
	"context"
	"net"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/structpb"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

// bufconnDialer returns a grpc.WithContextDialer using a bufconn.Listener.
func bufconnDialer(lis *bufconn.Listener) func(context.Context, string) (net.Conn, error) {
	return func(ctx context.Context, _ string) (net.Conn, error) {
		return lis.DialContext(ctx)
	}
}

// GoVecServerSuite is the bufconn integration suite.
type GoVecServerSuite struct {
	suite.Suite
	lis    *bufconn.Listener
	conn   *grpc.ClientConn
	client pb.GoVecServiceClient
}

func TestGoVecServerSuite(t *testing.T) {
	suite.Run(t, new(GoVecServerSuite))
}

func (s *GoVecServerSuite) SetupTest() {
	engine := testutil.NewTestIndex(s.T())
	s.lis = bufconn.Listen(1024 * 1024)
	dataPath := filepath.Join(s.T().TempDir(), "data.bin")

	srv := grpcserver.SetupGRPCServer(engine, "", dataPath, 4)
	go srv.Serve(s.lis) //nolint:errcheck // test server
	s.T().Cleanup(srv.GracefulStop)

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufconnDialer(s.lis)),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(s.T(), err)
	s.conn = conn
	s.T().Cleanup(func() { _ = conn.Close() })
	s.client = pb.NewGoVecServiceClient(conn)
}

// --- Insert ---

func (s *GoVecServerSuite) TestInsert_HappyPath() {
	resp, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "vec-001",
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
}

func (s *GoVecServerSuite) TestInsert_MissingID() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
}

func (s *GoVecServerSuite) TestInsert_MissingVector() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id: "vec-002",
	})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
}

// A wrong-width vector is a bad request, not a server fault. The dimension
// guard rejects it; this pins the status code the client sees.
func (s *GoVecServerSuite) TestInsert_DimensionMismatch() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "vec-seed",
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.NoError(s.T(), err)

	_, err = s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "vec-wrong-width",
		Vector: []float32{1.0, 2.0},
	})

	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
	assert.Contains(s.T(), status.Convert(err).Message(), "index requires 3")
}

func (s *GoVecServerSuite) TestInsert_WithMetadata() {
	meta := map[string]*structpb.Value{
		"label": structpb.NewStringValue("test"),
		"score": structpb.NewNumberValue(0.99),
	}
	resp, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:       "vec-meta",
		Vector:   []float32{1.0, 0.0, 0.0},
		Metadata: meta,
	})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
}

// --- BatchInsert ---

func (s *GoVecServerSuite) TestBatchInsert_HappyPath() {
	stream, err := s.client.BatchInsert(context.Background())
	require.NoError(s.T(), err)

	for i, id := range []string{"b1", "b2", "b3"} {
		require.NoError(s.T(), stream.Send(&pb.InsertRequest{
			Id:     id,
			Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
		}))
	}

	resp, err := stream.CloseAndRecv()
	require.NoError(s.T(), err)
	assert.Equal(s.T(), int32(3), resp.InsertedCount)
	assert.Empty(s.T(), resp.Errors)
}

func (s *GoVecServerSuite) TestBatchInsert_InvalidSparseVector() {
	stream, err := s.client.BatchInsert(context.Background())
	require.NoError(s.T(), err)

	// Send one valid and one with mismatched sparse indices/values
	require.NoError(s.T(), stream.Send(&pb.InsertRequest{
		Id:     "good",
		Vector: []float32{1.0, 2.0, 3.0},
	}))
	require.NoError(s.T(), stream.Send(&pb.InsertRequest{
		Id:     "bad-sparse",
		Vector: []float32{1.0, 2.0, 3.0},
		Sparse: &pb.SparseVector{
			Indices: []uint32{0, 1},
			Values:  []float32{0.5}, // length mismatch
		},
	}))

	resp, err := stream.CloseAndRecv()
	require.NoError(s.T(), err)
	assert.Equal(s.T(), int32(1), resp.InsertedCount)
	assert.Len(s.T(), resp.Errors, 1)
	assert.Equal(s.T(), "bad-sparse", resp.Errors[0].Id)
}

// --- Search ---

func (s *GoVecServerSuite) TestSearch_HappyPath() {
	// Insert a vector first
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "search-vec",
		Vector: []float32{1.0, 0.0, 0.0},
	})
	require.NoError(s.T(), err)

	resp, err := s.client.Search(context.Background(), &pb.SearchRequest{
		QueryVector: []float32{1.0, 0.0, 0.0},
		K:           1,
	})
	require.NoError(s.T(), err)
	assert.Len(s.T(), resp.Results, 1)
	assert.Equal(s.T(), "search-vec", resp.Results[0].Id)
}

func (s *GoVecServerSuite) TestSearch_EmptyVector() {
	_, err := s.client.Search(context.Background(), &pb.SearchRequest{K: 1})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
}

func (s *GoVecServerSuite) TestSearch_NegativeK() {
	_, err := s.client.Search(context.Background(), &pb.SearchRequest{
		QueryVector: []float32{1.0, 0.0, 0.0},
		K:           -1,
	})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
}

// Search carried the same miscategorisation Insert did: a wrong-width query
// came back as Internal, blaming the server for a bad request.
func (s *GoVecServerSuite) TestSearch_DimensionMismatch() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "search-width-seed",
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.NoError(s.T(), err)

	_, err = s.client.Search(context.Background(), &pb.SearchRequest{
		QueryVector: []float32{1.0, 2.0},
		K:           1,
	})

	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
	assert.Contains(s.T(), status.Convert(err).Message(), "index requires 3")
}

// --- Delete ---

func (s *GoVecServerSuite) TestDelete_HappyPath() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "to-delete",
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.NoError(s.T(), err)

	resp, err := s.client.Delete(context.Background(), &pb.DeleteRequest{Id: "to-delete"})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
	assert.Equal(s.T(), "to-delete", resp.Id)
}

func (s *GoVecServerSuite) TestDelete_NotFound() {
	_, err := s.client.Delete(context.Background(), &pb.DeleteRequest{Id: "nonexistent"})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.NotFound, status.Code(err))
}

// --- GetByID ---

func (s *GoVecServerSuite) TestGetByID_HappyPath() {
	meta := map[string]*structpb.Value{
		"label": structpb.NewStringValue("stored"),
	}
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:       "fetch-me",
		Vector:   []float32{1.0, 2.0, 3.0},
		Metadata: meta,
		Sparse:   &pb.SparseVector{Indices: []uint32{0, 2}, Values: []float32{0.5, 0.25}},
	})
	require.NoError(s.T(), err)

	resp, err := s.client.GetByID(context.Background(), &pb.GetByIDRequest{Id: "fetch-me"})

	require.NoError(s.T(), err)
	assert.Equal(s.T(), "fetch-me", resp.Id)
	assert.Equal(s.T(), []float32{1.0, 2.0, 3.0}, resp.Vector)
	assert.Equal(s.T(), "stored", resp.Metadata["label"].GetStringValue())
	require.NotNil(s.T(), resp.Sparse, "a stored sparse vector must come back")
	assert.Equal(s.T(), []uint32{0, 2}, resp.Sparse.Indices)
}

// A record with no sparse vector must not carry an empty message back --
// SparseToProto returns nil for that case and this pins it over the wire.
func (s *GoVecServerSuite) TestGetByID_OmitsEmptySparse() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "dense-only",
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.NoError(s.T(), err)

	resp, err := s.client.GetByID(context.Background(), &pb.GetByIDRequest{Id: "dense-only"})

	require.NoError(s.T(), err)
	assert.Nil(s.T(), resp.Sparse)
}

func (s *GoVecServerSuite) TestGetByID_NotFound() {
	_, err := s.client.GetByID(context.Background(), &pb.GetByIDRequest{Id: "nonexistent"})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.NotFound, status.Code(err))
}

func (s *GoVecServerSuite) TestGetByID_MissingID() {
	_, err := s.client.GetByID(context.Background(), &pb.GetByIDRequest{})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.InvalidArgument, status.Code(err))
}

// --- Reset ---

func (s *GoVecServerSuite) TestReset_ClearsEveryVector() {
	for _, id := range []string{"r1", "r2"} {
		_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
			Id:     id,
			Vector: []float32{1.0, 2.0, 3.0},
		})
		require.NoError(s.T(), err)
	}
	before, err := s.client.Stats(context.Background(), &pb.StatsRequest{})
	require.NoError(s.T(), err)
	require.Equal(s.T(), int32(2), before.VectorCount)

	resp, err := s.client.Reset(context.Background(), &pb.ResetRequest{})

	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)

	after, err := s.client.Stats(context.Background(), &pb.StatsRequest{})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), int32(0), after.VectorCount)

	_, err = s.client.GetByID(context.Background(), &pb.GetByIDRequest{Id: "r1"})
	assert.Equal(s.T(), codes.NotFound, status.Code(err), "a cleared vector must not be fetchable")
}

// Reset must leave the index usable, not just empty -- clearing the ID mapper
// and then inserting again is where a half-finished Clear would show up.
func (s *GoVecServerSuite) TestReset_IndexStillUsableAfterwards() {
	_, err := s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "before-reset",
		Vector: []float32{1.0, 2.0, 3.0},
	})
	require.NoError(s.T(), err)

	_, err = s.client.Reset(context.Background(), &pb.ResetRequest{})
	require.NoError(s.T(), err)

	_, err = s.client.Insert(context.Background(), &pb.InsertRequest{
		Id:     "after-reset",
		Vector: []float32{3.0, 2.0, 1.0},
	})
	require.NoError(s.T(), err)

	found, err := s.client.Search(context.Background(), &pb.SearchRequest{
		QueryVector: []float32{3.0, 2.0, 1.0},
		K:           5,
	})
	require.NoError(s.T(), err)
	require.Len(s.T(), found.Results, 1)
	assert.Equal(s.T(), "after-reset", found.Results[0].Id)
}

// --- Stats ---

func (s *GoVecServerSuite) TestStats() {
	resp, err := s.client.Stats(context.Background(), &pb.StatsRequest{})
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), resp.VectorCount, int32(0))
}

// --- Info ---

func (s *GoVecServerSuite) TestInfo() {
	resp, err := s.client.Info(context.Background(), &pb.InfoRequest{})
	require.NoError(s.T(), err)
	// The test engine (created via testutil.NewTestIndex) bypasses the factory
	// and does not populate indexType/distanceMetric — assert the RPC succeeds.
	assert.GreaterOrEqual(s.T(), resp.VectorCount, int32(0))
}

// --- Flush ---

func (s *GoVecServerSuite) TestFlush() {
	resp, err := s.client.Flush(context.Background(), &pb.FlushRequest{})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
}

// --- Health ---

func (s *GoVecServerSuite) TestHealth() {
	resp, err := s.client.Health(context.Background(), &pb.HealthRequest{})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
}
