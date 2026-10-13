package grpcserver_test

import (
	"context"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

// newClient serves engine over an in-memory gRPC connection, requiring apiKey
// when it isn't empty, and returns a client for it.
func newClient(t *testing.T, engine index.Engine, apiKey string) pb.GoVecServiceClient {
	t.Helper()
	lis := bufconn.Listen(1024 * 1024)
	srv := grpcserver.SetupGRPCServer(engine, apiKey, 4)
	go srv.Serve(lis) //nolint:errcheck // test server
	t.Cleanup(srv.GracefulStop)

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return pb.NewGoVecServiceClient(conn)
}

// TestGRPC_VectorLifecycle drives one index through every RPC, as
// TestREST_VectorLifecycle does over REST; the status strings the two answer
// with are compared in internal/test/integration/transport_parity_test.go.
func TestGRPC_VectorLifecycle(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := newClient(t, testutil.NewTestIndexAt(t, t.TempDir()+"/snapshot.bin"), "")

	// Act + Assert, one step at a time: each step depends on the ones before.
	health, err := client.Health(ctx, &pb.HealthRequest{})
	require.NoError(t, err)
	assert.Equal(t, "ok", health.Status)

	inserted, err := client.Insert(ctx, &pb.InsertRequest{
		Id:       "doc1",
		Vector:   []float32{1, 0, 0},
		Metadata: map[string]*structpb.Value{"label": structpb.NewStringValue("stored")},
		Sparse:   &pb.SparseVector{Indices: []uint32{0, 2}, Values: []float32{0.5, 0.25}},
	})
	require.NoError(t, err)
	assert.Equal(t, "inserted", inserted.Status)

	stream, err := client.BatchInsert(ctx)
	require.NoError(t, err)
	require.NoError(t, stream.Send(&pb.InsertRequest{Id: "doc2", Vector: []float32{0.9, 0.1, 0}}))
	require.NoError(t, stream.Send(&pb.InsertRequest{Id: "bad", Vector: []float32{0, 1, 0}, Sparse: &pb.SparseVector{Indices: []uint32{0, 1}, Values: []float32{0.5}}}))
	require.NoError(t, stream.Send(&pb.InsertRequest{Id: "doc3", Vector: []float32{0, 1, 0}}))
	batch, err := stream.CloseAndRecv()
	require.NoError(t, err)
	assert.Equal(t, int32(2), batch.InsertedCount)
	require.Len(t, batch.Errors, 1)
	assert.Equal(t, "bad", batch.Errors[0].Id, "a bad item is reported against its ID and doesn't sink the batch")

	got, err := client.GetByID(ctx, &pb.GetByIDRequest{Id: "doc1"})
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 0, 0}, got.Vector)
	assert.Equal(t, "stored", got.Metadata["label"].GetStringValue())
	require.NotNil(t, got.Sparse)
	assert.Equal(t, []uint32{0, 2}, got.Sparse.Indices)

	denseOnly, err := client.GetByID(ctx, &pb.GetByIDRequest{Id: "doc2"})
	require.NoError(t, err)
	assert.Nil(t, denseOnly.Sparse, "a record with no sparse part must not carry an empty one")

	found, err := client.Search(ctx, &pb.SearchRequest{QueryVector: []float32{1, 0, 0}, K: proto.Int32(2)})
	require.NoError(t, err)
	require.Len(t, found.Results, 2)
	assert.Equal(t, "doc1", found.Results[0].Id)
	assert.Equal(t, "doc2", found.Results[1].Id)

	none, err := client.Search(ctx, &pb.SearchRequest{QueryVector: []float32{1, 0, 0}, K: proto.Int32(0)})
	require.NoError(t, err)
	assert.Empty(t, none.Results, "k=0 asks for nothing")

	deleted, err := client.Delete(ctx, &pb.DeleteRequest{Id: "doc2"})
	require.NoError(t, err)
	assert.Equal(t, "deleted", deleted.Status)
	assert.Equal(t, "doc2", deleted.Id)

	stats, err := client.Stats(ctx, &pb.StatsRequest{})
	require.NoError(t, err)
	assert.Equal(t, int32(2), stats.VectorCount)

	info, err := client.Info(ctx, &pb.InfoRequest{})
	require.NoError(t, err)
	assert.Equal(t, int32(3), info.Dimensions)

	flushed, err := client.Flush(ctx, &pb.FlushRequest{})
	require.NoError(t, err)
	assert.Equal(t, "flushed", flushed.Status)

	reset, err := client.Reset(ctx, &pb.ResetRequest{})
	require.NoError(t, err)
	assert.Equal(t, "reset", reset.Status)
	_, err = client.GetByID(ctx, &pb.GetByIDRequest{Id: "doc1"})
	assert.Equal(t, codes.NotFound, status.Code(err), "a reset must remove every vector")

	_, err = client.Insert(ctx, &pb.InsertRequest{Id: "after", Vector: []float32{0, 1}})
	require.NoError(t, err, "a reset index takes a new width")
	found, err = client.Search(ctx, &pb.SearchRequest{QueryVector: []float32{0, 1}, K: proto.Int32(5)})
	require.NoError(t, err)
	require.Len(t, found.Results, 1)
	assert.Equal(t, "after", found.Results[0].Id)
}

// TestGRPC_BadRequests: a request the caller got wrong is InvalidArgument or
// NotFound, saying what is wrong -- never Internal, which would blame the
// server. Wrong-width vectors used to come back as Internal.
func TestGRPC_BadRequests(t *testing.T) {
	tests := []struct {
		name     string
		call     func(ctx context.Context, c pb.GoVecServiceClient) error
		wantCode codes.Code
		wantMsg  string
	}{
		{"insert: missing id", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Insert(ctx, &pb.InsertRequest{Vector: []float32{1, 2, 3}})
			return err
		}, codes.InvalidArgument, ""},
		{"insert: missing vector", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Insert(ctx, &pb.InsertRequest{Id: "x"})
			return err
		}, codes.InvalidArgument, ""},
		{"insert: wrong width", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Insert(ctx, &pb.InsertRequest{Id: "x", Vector: []float32{1, 2}})
			return err
		}, codes.InvalidArgument, "index requires 3"},
		{"search: empty vector", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Search(ctx, &pb.SearchRequest{K: proto.Int32(1)})
			return err
		}, codes.InvalidArgument, ""},
		{"search: wrong width", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Search(ctx, &pb.SearchRequest{QueryVector: []float32{1, 2}, K: proto.Int32(1)})
			return err
		}, codes.InvalidArgument, "index requires 3"},
		{"search: missing k", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Search(ctx, &pb.SearchRequest{QueryVector: []float32{1, 2, 3}})
			return err
		}, codes.InvalidArgument, ""},
		{"search: negative k", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Search(ctx, &pb.SearchRequest{QueryVector: []float32{1, 2, 3}, K: proto.Int32(-1)})
			return err
		}, codes.InvalidArgument, ""},
		{"get: missing id", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.GetByID(ctx, &pb.GetByIDRequest{})
			return err
		}, codes.InvalidArgument, ""},
		{"get: unknown id", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.GetByID(ctx, &pb.GetByIDRequest{Id: "missing"})
			return err
		}, codes.NotFound, ""},
		{"delete: unknown id", func(ctx context.Context, c pb.GoVecServiceClient) error {
			_, err := c.Delete(ctx, &pb.DeleteRequest{Id: "missing"})
			return err
		}, codes.NotFound, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: an index 3 dimensions wide.
			ctx := context.Background()
			engine := testutil.NewTestIndex(t)
			client := newClient(t, engine, "")
			_, err := client.Insert(ctx, &pb.InsertRequest{Id: "stored", Vector: []float32{1, 2, 3}})
			require.NoError(t, err)

			// Act
			err = tt.call(ctx, client)

			// Assert
			assert.Equal(t, tt.wantCode, status.Code(err), "%v", err)
			assert.Contains(t, status.Convert(err).Message(), tt.wantMsg)
			assert.Equal(t, 1, engine.Len(), "a rejected request must change nothing")
		})
	}
}
