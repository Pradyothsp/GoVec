package grpcserver

import (
	"google.golang.org/grpc"

	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/index"
)

// SetupGRPCServer builds a ready-to-serve *grpc.Server with auth + logging interceptors.
// The caller is responsible for creating the net.Listener and calling Serve.
func SetupGRPCServer(engine index.Engine, apiKey, dataPath string, maxRecvMsgSizeMB int) *grpc.Server {
	srv := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxRecvMsgSizeMB*1024*1024),
		grpc.ChainUnaryInterceptor(
			UnaryLoggingInterceptor,
			UnaryAuthInterceptor(apiKey),
		),
		grpc.ChainStreamInterceptor(
			StreamLoggingInterceptor,
			StreamAuthInterceptor(apiKey),
		),
	)
	pb.RegisterGoVecServiceServer(srv, NewGoVecServer(engine, dataPath))
	return srv
}
