package grpcserver_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/test/testutil"
)

// The auth interceptors guard unary and streaming RPCs alike.
func TestAuthInterceptors(t *testing.T) {
	unary := func(ctx context.Context, c pb.GoVecServiceClient) error {
		_, err := c.Health(ctx, &pb.HealthRequest{})
		return err
	}
	streaming := func(ctx context.Context, c pb.GoVecServiceClient) error {
		stream, err := c.BatchInsert(ctx)
		if err != nil {
			return err
		}
		_, err = stream.CloseAndRecv()
		return err
	}
	tests := []struct {
		name     string
		apiKey   string
		header   string // authorization; "" sends none
		call     func(context.Context, pb.GoVecServiceClient) error
		wantCode codes.Code
	}{
		{"no key configured, no header", "", "", unary, codes.OK},
		{"unary: no header", "secret", "", unary, codes.Unauthenticated},
		{"unary: wrong token", "secret", "Bearer wrong", unary, codes.Unauthenticated},
		{"unary: correct token", "secret", "Bearer secret", unary, codes.OK},
		{"stream: wrong token", "secret", "Bearer wrong", streaming, codes.Unauthenticated},
		{"stream: correct token", "secret", "Bearer secret", streaming, codes.OK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			client := newClient(t, testutil.NewTestIndex(t), tt.apiKey)
			ctx := context.Background()
			if tt.header != "" {
				ctx = metadata.AppendToOutgoingContext(ctx, "authorization", tt.header)
			}

			// Act
			err := tt.call(ctx, client)

			// Assert
			assert.Equal(t, tt.wantCode, status.Code(err), "%v", err)
		})
	}
}
