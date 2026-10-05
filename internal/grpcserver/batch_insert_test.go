package grpcserver_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/core"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

// callCounter is a real engine that counts how a transport writes to it.
type callCounter struct {
	index.Engine
	inserts, batches int
}

func (c *callCounter) Insert(ctx context.Context, id string, vec []float32, sparse core.SparseVector, meta map[string]any) error {
	c.inserts++
	return c.Engine.Insert(ctx, id, vec, sparse, meta)
}

func (c *callCounter) BatchInsert(ctx context.Context, items []index.BatchInsertItem) ([]index.BatchInsertError, error) {
	c.batches++
	return c.Engine.BatchInsert(ctx, items)
}

// countingClient serves engine over gRPC and returns a client for it.
func countingClient(t *testing.T, engine *callCounter) pb.GoVecServiceClient {
	t.Helper()
	lis := bufconn.Listen(1024 * 1024)
	srv := grpcserver.SetupGRPCServer(engine, "", filepath.Join(t.TempDir(), "data.bin"), 4)
	go srv.Serve(lis) //nolint:errcheck // test server
	t.Cleanup(srv.GracefulStop)
	conn, err := grpc.NewClient("passthrough:///bufnet", grpc.WithContextDialer(bufconnDialer(lis)), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return pb.NewGoVecServiceClient(conn)
}

// TestBatchInsert_IsOneEngineBatch: a streamed batch goes to the engine as a
// batch, one WAL fsync, as REST's does. It used to insert item by item, an
// fsync per vector, so the same batch was far slower over gRPC.
func TestBatchInsert_IsOneEngineBatch(t *testing.T) {
	engine := &callCounter{Engine: testutil.NewTestIndex(t)}

	stream, err := countingClient(t, engine).BatchInsert(context.Background())
	require.NoError(t, err)
	for _, req := range []*pb.InsertRequest{
		{Id: "a", Vector: []float32{1, 0, 0}},
		{Id: "bad", Vector: []float32{0, 1, 0}, Sparse: &pb.SparseVector{Indices: []uint32{0, 1}, Values: []float32{0.5}}},
		{Id: "b", Vector: []float32{0, 0, 1}},
	} {
		require.NoError(t, stream.Send(req))
	}
	resp, err := stream.CloseAndRecv()
	require.NoError(t, err)

	assert.Equal(t, 1, engine.batches)
	assert.Zero(t, engine.inserts)
	assert.Equal(t, int32(2), resp.InsertedCount)
	require.Len(t, resp.Errors, 1)
	assert.Equal(t, "bad", resp.Errors[0].Id)
	assert.Equal(t, 2, engine.Len())
}

// TestBatchInsert_LongStream_GoesInChunks: a stream longer than one chunk is
// applied as several engine batches, and every item lands.
func TestBatchInsert_LongStream_GoesInChunks(t *testing.T) {
	const n = 1001 // one full chunk of 1000, then one more
	engine := &callCounter{Engine: testutil.NewTestIndex(t)}

	stream, err := countingClient(t, engine).BatchInsert(context.Background())
	require.NoError(t, err)
	for i := range n {
		require.NoError(t, stream.Send(&pb.InsertRequest{Id: fmt.Sprintf("v%d", i), Vector: []float32{1, float32(i), 0}}))
	}
	resp, err := stream.CloseAndRecv()
	require.NoError(t, err)

	assert.Equal(t, 2, engine.batches)
	assert.Equal(t, int32(n), resp.InsertedCount)
	assert.Empty(t, resp.Errors)
	assert.Equal(t, n, engine.Len())
}
