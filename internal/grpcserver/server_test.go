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
