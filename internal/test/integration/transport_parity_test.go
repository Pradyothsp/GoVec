package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/release"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

// Choosing a protocol is meant to change the wire format and nothing else, so
// an operation must report the same status string whichever transport ran it.
// It did not: REST answered "inserted"/"deleted"/"flushed"/"reset" while gRPC
// answered "ok" to all four, and only health agreed. A caller checking
// `status == "ok"` therefore passed over gRPC and failed over REST.
//
// Both suites had their own status assertions and both were green -- neither
// could see the other. This test drives one engine through both transports and
// compares the answers, so the two cannot drift apart again without failing.

// parityServers starts a REST router and a gRPC server over separate engines
// and returns a caller for each.
func parityServers(t *testing.T) (rest *gin.Engine, grpcClient pb.GoVecServiceClient) {
	t.Helper()
	gin.SetMode(gin.TestMode)

	restEngine := testutil.NewTestIndex(t)
	rest = api.SetupRouter(restEngine, "", filepath.Join(t.TempDir(), "rest.bin"))

	grpcEngine := testutil.NewTestIndex(t)
	lis := bufconn.Listen(1024 * 1024)
	srv := grpcserver.SetupGRPCServer(grpcEngine, "", filepath.Join(t.TempDir(), "grpc.bin"), 4)
	go srv.Serve(lis) //nolint:errcheck // test server
	t.Cleanup(srv.GracefulStop)

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })

	return rest, pb.NewGoVecServiceClient(conn)
}

// restStatus performs a REST call and digs the status string out of the
// envelope, which is {"success":true,"data":{"status":...}}.
func restStatus(t *testing.T, router *gin.Engine, method, path string, body any) string {
	t.Helper()

	var reader *bytes.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		require.NoError(t, err)
		reader = bytes.NewReader(raw)
	} else {
		reader = bytes.NewReader(nil)
	}

	req := httptest.NewRequest(method, path, reader)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var envelope struct {
		Data struct {
			Status string `json:"status"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &envelope),
		"body was %s", rec.Body.String())
	require.NotEmpty(t, envelope.Data.Status,
		"%s %s returned no status: %s", method, path, rec.Body.String())

	return envelope.Data.Status
}

func TestStatusStrings_RESTAndGRPCAgree(t *testing.T) {
	router, grpcClient := parityServers(t)
	ctx := context.Background()
	vec := []float32{1, 2, 3}

	tests := []struct {
		name string
		rest func() string
		grpc func() string
	}{
		{
			name: "insert",
			rest: func() string {
				return restStatus(t, router, http.MethodPost, "/api/v1/vectors",
					map[string]any{"id": "parity", "vector": vec})
			},
			grpc: func() string {
				resp, err := grpcClient.Insert(ctx, &pb.InsertRequest{Id: "parity", Vector: vec})
				require.NoError(t, err)
				return resp.Status
			},
		},
		{
			name: "delete",
			rest: func() string {
				return restStatus(t, router, http.MethodDelete, "/api/v1/vectors/parity", nil)
			},
			grpc: func() string {
				resp, err := grpcClient.Delete(ctx, &pb.DeleteRequest{Id: "parity"})
				require.NoError(t, err)
				return resp.Status
			},
		},
		{
			name: "flush",
			rest: func() string {
				return restStatus(t, router, http.MethodPost, "/api/v1/admin/flush", nil)
			},
			grpc: func() string {
				resp, err := grpcClient.Flush(ctx, &pb.FlushRequest{})
				require.NoError(t, err)
				return resp.Status
			},
		},
		{
			name: "reset",
			rest: func() string {
				return restStatus(t, router, http.MethodPost, "/api/v1/admin/reset", nil)
			},
			grpc: func() string {
				resp, err := grpcClient.Reset(ctx, &pb.ResetRequest{})
				require.NoError(t, err)
				return resp.Status
			},
		},
		{
			name: "health",
			rest: func() string {
				return restStatus(t, router, http.MethodGet, "/health", nil)
			},
			grpc: func() string {
				resp, err := grpcClient.Health(ctx, &pb.HealthRequest{})
				require.NoError(t, err)
				return resp.Status
			},
		},
	}

	// Sequential rather than parallel: insert has to land before delete runs,
	// on both transports.
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			restStatus := tt.rest()
			grpcStatus := tt.grpc()

			assert.Equal(t, restStatus, grpcStatus,
				"%s reports %q over REST but %q over gRPC; the two transports must agree",
				tt.name, restStatus, grpcStatus)
		})
	}
}

// A client asking a server which release it is must get the same answer over either
// transport -- the field was added to both at once, and this keeps either from dropping it.
func TestInfoVersion_RESTAndGRPCAgree(t *testing.T) {
	original := release.Version
	release.Version = "v9.9.9-parity"
	t.Cleanup(func() { release.Version = original })

	router, grpcClient := parityServers(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/info", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var envelope struct {
		Data struct {
			Version string `json:"version"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &envelope), "body was %s", rec.Body.String())

	resp, err := grpcClient.Info(context.Background(), &pb.InfoRequest{})
	require.NoError(t, err)

	assert.Equal(t, "v9.9.9-parity", envelope.Data.Version, "REST /info version")
	assert.Equal(t, "v9.9.9-parity", resp.Version, "gRPC Info version")
}

// TestSparseVector_RESTAndGRPCAgree: the sparse-vector rule lives in the
// engine, so both transports accept an empty one as "no sparse part" and
// reject mismatched lengths as the caller's error. REST used to reject an
// empty one too, and the two answered a mismatch with different messages.
func TestSparseVector_RESTAndGRPCAgree(t *testing.T) {
	router, grpcClient := parityServers(t)
	ctx := context.Background()
	vec := []float32{1, 2, 3}

	restInsert := func(id string, indices []uint32, values []float32) *httptest.ResponseRecorder {
		raw, err := json.Marshal(map[string]any{"id": id, "vector": vec, "sparse_vector": map[string]any{"indices": indices, "values": values}})
		require.NoError(t, err)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/vectors", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	t.Run("empty is accepted", func(t *testing.T) {
		rec := restInsert("empty", []uint32{}, []float32{})
		assert.Equal(t, http.StatusCreated, rec.Code, rec.Body.String())

		_, err := grpcClient.Insert(ctx, &pb.InsertRequest{Id: "empty", Vector: vec, Sparse: &pb.SparseVector{}})
		assert.NoError(t, err)
	})

	t.Run("mismatched lengths are rejected", func(t *testing.T) {
		rec := restInsert("bad", []uint32{0, 1}, []float32{0.5})
		assert.Equal(t, http.StatusBadRequest, rec.Code)

		_, err := grpcClient.Insert(ctx, &pb.InsertRequest{Id: "bad", Vector: vec, Sparse: &pb.SparseVector{Indices: []uint32{0, 1}, Values: []float32{0.5}}})
		require.Error(t, err)
		assert.Equal(t, codes.InvalidArgument, status.Code(err))

		assert.Contains(t, rec.Body.String(), "indices and values must have the same length")
		assert.Contains(t, err.Error(), "indices and values must have the same length")
	})
}
