package grpcserver

import (
	"context"
	"crypto/subtle"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// UnaryLoggingInterceptor logs the method, duration, and status code for unary RPCs.
func UnaryLoggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	code := status.Code(err)

	event := log.Info()
	switch {
	case code == codes.OK:
		event = log.Info()
	case code == codes.NotFound || code == codes.InvalidArgument || code == codes.AlreadyExists || code == codes.Unauthenticated:
		event = log.Warn()
	case err != nil:
		event = log.Error()
	}

	event.
		Str("method", info.FullMethod).
		Dur("duration", time.Since(start)).
		Str("code", code.String()).
		Msg("grpc unary")
	return resp, err
}

// UnaryAuthInterceptor returns a unary interceptor that enforces Bearer token auth.
// When apiKey is empty the interceptor is a no-op.
func UnaryAuthInterceptor(apiKey string) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		if apiKey == "" {
			return handler(ctx, req)
		}
		if err := checkAuth(ctx, apiKey); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

// StreamLoggingInterceptor logs the method, duration, and status code for streaming RPCs.
func StreamLoggingInterceptor(
	srv interface{},
	ss grpc.ServerStream,
	info *grpc.StreamServerInfo,
	handler grpc.StreamHandler,
) error {
	start := time.Now()
	err := handler(srv, ss)
	code := status.Code(err)

	event := log.Info()
	switch {
	case code == codes.OK:
		event = log.Info()
	case code == codes.NotFound || code == codes.InvalidArgument || code == codes.AlreadyExists || code == codes.Unauthenticated:
		event = log.Warn()
	case err != nil:
		event = log.Error()
	}

	event.
		Str("method", info.FullMethod).
		Dur("duration", time.Since(start)).
		Str("code", code.String()).
		Msg("grpc stream")
	return err
}

// StreamAuthInterceptor returns a stream interceptor that enforces Bearer token auth.
// When apiKey is empty the interceptor is a no-op.
func StreamAuthInterceptor(apiKey string) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		if apiKey == "" {
			return handler(srv, ss)
		}
		if err := checkAuth(ss.Context(), apiKey); err != nil {
			return err
		}
		return handler(srv, ss)
	}
}

// checkAuth reads the "authorization" metadata header and validates the Bearer token.
func checkAuth(ctx context.Context, apiKey string) error {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return status.Error(codes.Unauthenticated, "missing metadata")
	}
	vals := md.Get("authorization")
	if len(vals) == 0 {
		return status.Error(codes.Unauthenticated, "missing authorization header")
	}
	token := strings.TrimPrefix(vals[0], "Bearer ")
	if subtle.ConstantTimeCompare([]byte(token), []byte(apiKey)) != 1 {
		return status.Error(codes.Unauthenticated, "invalid token")
	}
	return nil
}
