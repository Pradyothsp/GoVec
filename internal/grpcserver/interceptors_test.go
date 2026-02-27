package grpcserver_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

const authTestAPIKey = "test-key"

// AuthInterceptorSuite tests auth interceptor behaviour in isolation.
type AuthInterceptorSuite struct {
	suite.Suite
	lis    *bufconn.Listener
	conn   *grpc.ClientConn
	client pb.GoVecServiceClient
}

func TestAuthInterceptorSuite(t *testing.T) {
	suite.Run(t, new(AuthInterceptorSuite))
}

func (s *AuthInterceptorSuite) SetupTest() {
	engine := testutil.NewTestIndex(s.T())
	s.lis = bufconn.Listen(1024 * 1024)

	srv := grpcserver.SetupGRPCServer(engine, authTestAPIKey, filepath.Join(s.T().TempDir(), "data.bin"), 4)
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

// TestNoMetadata — missing authorization header → Unauthenticated.
func (s *AuthInterceptorSuite) TestNoMetadata() {
	_, err := s.client.Health(context.Background(), &pb.HealthRequest{})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.Unauthenticated, status.Code(err))
}

// TestWrongToken — wrong token → Unauthenticated.
func (s *AuthInterceptorSuite) TestWrongToken() {
	ctx := metadata.AppendToOutgoingContext(context.Background(), "authorization", "Bearer wrong-key")
	_, err := s.client.Health(ctx, &pb.HealthRequest{})
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.Unauthenticated, status.Code(err))
}

// TestCorrectToken — correct token → OK.
func (s *AuthInterceptorSuite) TestCorrectToken() {
	ctx := metadata.AppendToOutgoingContext(context.Background(), "authorization", "Bearer "+authTestAPIKey)
	resp, err := s.client.Health(ctx, &pb.HealthRequest{})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
}

// TestNoAuthRequired — empty apiKey → any request passes without a header.
func (s *AuthInterceptorSuite) TestNoAuthRequired() {
	engine := testutil.NewTestIndex(s.T())
	lis := bufconn.Listen(1024 * 1024)
	srv := grpcserver.SetupGRPCServer(engine, "", filepath.Join(s.T().TempDir(), "data.bin"), 4)
	go srv.Serve(lis) //nolint:errcheck // test server
	s.T().Cleanup(srv.GracefulStop)

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(bufconnDialer(lis)),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(s.T(), err)
	defer conn.Close() //nolint:errcheck // test cleanup

	client := pb.NewGoVecServiceClient(conn)
	resp, err := client.Health(context.Background(), &pb.HealthRequest{})
	require.NoError(s.T(), err)
	assert.Equal(s.T(), "ok", resp.Status)
}

// TestStreamWithCorrectToken — client-streaming RPC passes with correct token.
func (s *AuthInterceptorSuite) TestStreamWithCorrectToken() {
	ctx := metadata.AppendToOutgoingContext(context.Background(), "authorization", "Bearer "+authTestAPIKey)
	stream, err := s.client.BatchInsert(ctx)
	require.NoError(s.T(), err)
	resp, err := stream.CloseAndRecv()
	require.NoError(s.T(), err)
	assert.Equal(s.T(), int32(0), resp.InsertedCount)
}

// TestStreamWithWrongToken — client-streaming RPC fails with wrong token.
func (s *AuthInterceptorSuite) TestStreamWithWrongToken() {
	ctx := metadata.AppendToOutgoingContext(context.Background(), "authorization", "Bearer bad-token")
	stream, err := s.client.BatchInsert(ctx)
	require.NoError(s.T(), err)
	_, err = stream.CloseAndRecv()
	require.Error(s.T(), err)
	assert.Equal(s.T(), codes.Unauthenticated, status.Code(err))
}
