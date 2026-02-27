package grpcserver

import (
	"context"
	"crypto/subtle"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/Pradyothsp/govec/internal/requestid"
)

// UnaryLoggingInterceptor logs the method, duration, and status code for unary RPCs.
// It extracts or generates a correlation ID from incoming metadata and attaches
// a scoped zerolog logger to the context so all downstream logs carry the ID.
func UnaryLoggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	id := extractOrGenerate(ctx)
	logger := log.With().Str("correlation_id", id).Logger()
	ctx = logger.WithContext(ctx)

	start := time.Now()
	resp, err := handler(ctx, req)
	code := status.Code(err)

	l := zerolog.Ctx(ctx)
	event := l.Info()
	switch {
	case code == codes.NotFound || code == codes.InvalidArgument || code == codes.AlreadyExists || code == codes.Unauthenticated:
		event = l.Warn()
	case err != nil:
		event = l.Error()
	}

	event.
		Str("method", info.FullMethod).
		Dur("duration", time.Since(start)).
		Str("code", code.String()).
		Msg("grpc unary")
	return resp, err
}

// extractOrGenerate reads the "x-correlation-id" metadata key from the incoming
// gRPC context, or generates a fresh UUID if absent or blank.
func extractOrGenerate(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if vals := md.Get("x-correlation-id"); len(vals) > 0 && vals[0] != "" {
			return vals[0]
		}
	}
	return requestid.Generate()
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

// wrappedStream overrides Context() so the enriched context (with correlation ID)
// propagates to handlers and any downstream code that calls ss.Context().
type wrappedStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (w *wrappedStream) Context() context.Context { return w.ctx }

// StreamLoggingInterceptor logs the method, duration, and status code for streaming RPCs.
// Like its unary counterpart it attaches a correlation-ID-scoped logger to the stream context.
func StreamLoggingInterceptor(
	srv interface{},
	ss grpc.ServerStream,
	info *grpc.StreamServerInfo,
	handler grpc.StreamHandler,
) error {
	id := extractOrGenerate(ss.Context())
	logger := log.With().Str("correlation_id", id).Logger()
	enriched := logger.WithContext(ss.Context())

	start := time.Now()
	err := handler(srv, &wrappedStream{ServerStream: ss, ctx: enriched})
	code := status.Code(err)

	l := zerolog.Ctx(enriched)
	event := l.Info()
	switch {
	case code == codes.NotFound || code == codes.InvalidArgument || code == codes.AlreadyExists || code == codes.Unauthenticated:
		event = l.Warn()
	case err != nil:
		event = l.Error()
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
