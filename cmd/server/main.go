package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	_ "net/http/pprof" //nolint:gosec // G108: pprof only starts when cfg.Server.PprofEnabled is true; addr is localhost-bound
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"

	"github.com/Pradyothsp/govec/build/swagger"
	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/index"
	"github.com/Pradyothsp/govec/internal/release"
)

// @title           GoVec API
// @version         1.0
// @description     A vector database REST API for storing and querying vector embeddings.
// @host            localhost:8000
// @BasePath        /
// @schemes         http
//
// @securityDefinitions.apikey BearerAuth
// @in                         header
// @name                       Authorization
// @description                Type 'Bearer ' followed by your API key.
func main() {
	setupLogging()

	// Load configuration
	cfg := loadConfiguration()
	setLogLevel(cfg.Server.LogLevel)

	// pprof debug server — config-driven, localhost-bound only
	debugSrv := startPprofServer(cfg)

	// Initialize WAL
	wal, err := index.NewWAL(cfg.Storage.WalPath)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to open WAL")
	}

	log.Info().Str("version", release.Version).Msg("starting server")

	// Create engine via factory
	engine, err := index.NewEngine(cfg.Engine, cfg.Storage, wal)
	if err != nil {
		if closeErr := wal.Close(); closeErr != nil {
			log.Error().Err(closeErr).Msg("failed to close WAL after engine creation failure")
		}
		log.Fatal().Err(err).Msg("failed to create engine")
	}

	// Engine and WAL initialized, set up cleanup
	defer func(wal *index.WAL) {
		err := wal.Close()
		if err != nil {
			log.Error().Err(err).Msg("failed to close WAL")
		}
	}(wal)

	// The generated spec carries the static @version annotation; report the running release.
	swagger.SwaggerInfo.Version = release.Version

	router := api.SetupRouter(engine, cfg.Server.APIKey, cfg.Storage.DataPath)

	// RECOVERY SEQUENCE -- before any server starts, so no request (REST or
	// gRPC) ever sees a partly loaded index.
	if err := runRecovery(cfg, engine); err != nil {
		if closeErr := wal.Close(); closeErr != nil {
			log.Error().Err(closeErr).Msg("failed to close WAL after recovery failure")
		}
		log.Fatal().Err(err).Msg("refusing to start") //nolint:gocritic // exitAfterDefer: WAL is explicitly closed above before fatal exit
	}

	grpcSrv, err := startGRPCServer(cfg, engine)
	if err != nil {
		if closeErr := wal.Close(); closeErr != nil {
			log.Error().Err(closeErr).Msg("failed to close WAL after gRPC startup failure")
		}
		log.Fatal().Err(err).Msg("failed to start gRPC server")
	}

	// Start Background Snapshotting (The "Auto-Save")
	stopAutoSave := startAutoSave(cfg, engine)

	// Create and start an HTTP server
	srv := startHTTPServer(cfg, router)

	// Wait for the termination signal
	waitForInterrupt()

	// Graceful shutdown
	shutdownServer(cfg, engine, srv, debugSrv, grpcSrv, stopAutoSave)
}

// setupLogging configures the global logger based on the environment.
func setupLogging() {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnixMs
	if fi, err := os.Stdout.Stat(); err == nil && (fi.Mode()&os.ModeCharDevice) != 0 {
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stdout})
	} else {
		log.Logger = zerolog.New(os.Stdout).With().Timestamp().Logger()
	}
}

// setLogLevel sets the global log level.
func setLogLevel(levelStr string) {
	level, err := zerolog.ParseLevel(levelStr)
	if err != nil {
		level = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(level)
	log.Info().Str("level", level.String()).Msg("log level set")
}

// startPprofServer starts the pprof debug server if enabled in configuration.
func startPprofServer(cfg *config.Config) *http.Server {
	if !cfg.Server.PprofEnabled {
		return nil
	}

	srv := &http.Server{
		Addr:              cfg.Server.PprofAddr,
		ReadHeaderTimeout: 5 * time.Second,
		Handler:           http.DefaultServeMux,
	}

	go func() {
		log.Info().Str("addr", cfg.Server.PprofAddr).Msg("pprof debug server listening")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error().Err(err).Msg("pprof server error")
		}
	}()

	return srv
}

// startGRPCServer starts the gRPC server if enabled in configuration.
func startGRPCServer(cfg *config.Config, engine index.Engine) (*grpc.Server, error) {
	if !cfg.GRPC.Enabled {
		return nil, nil
	}

	srv := grpcserver.SetupGRPCServer(engine, cfg.Server.APIKey, cfg.Storage.DataPath, cfg.GRPC.MaxRecvMsgSizeMB)
	grpcAddr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.GRPC.Port)

	lis, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		return nil, fmt.Errorf("failed to listen for gRPC on %s: %w", grpcAddr, err)
	}

	go func() {
		log.Info().Str("addr", grpcAddr).Msg("gRPC server listening")
		if err := srv.Serve(lis); err != nil {
			log.Error().Err(err).Msg("gRPC server error")
		}
	}()

	return srv, nil
}

// runRecovery handles the snapshot loading and WAL replay sequence.
//
// A snapshot that exists but can't be loaded is an error, not a warning: the
// WAL only holds writes since the last save, so replaying it alone would serve
// a fraction of the data as if it were all of it. A missing snapshot is a
// fresh start -- LoadFromFile returns nil for it.
//
// A WAL that can't be read to the end is an error for the same reason: every
// write after the failure point would be silently dropped. A missing WAL is a
// fresh start, and single malformed lines are skipped inside ReplayWAL.
func runRecovery(cfg *config.Config, engine index.Engine) error {
	// Step 1: Load the base snapshot from DataPath (GOB format)
	log.Info().Str("path", cfg.Storage.DataPath).Msg("loading snapshot from disk")
	if err := engine.LoadFromFile(context.Background(), cfg.Storage.DataPath); err != nil {
		return fmt.Errorf("load snapshot %s (restore a backup, or delete it to start empty): %w", cfg.Storage.DataPath, err)
	}
	log.Info().Int("vectors", engine.Len()).Msg("snapshot loaded")

	// Step 2: Replay the WAL from WalPath (JSON format) to recover uncommitted changes
	log.Info().Str("path", cfg.Storage.WalPath).Msg("replaying WAL")
	if err := engine.ReplayWAL(cfg.Storage.WalPath); err != nil {
		return fmt.Errorf("replay WAL %s (fix or move it aside to start from the snapshot alone): %w", cfg.Storage.WalPath, err)
	}
	log.Info().Int("vectors", engine.Len()).Msg("WAL replay complete")
	return nil
}

// startAutoSave starts the background snapshotting ticker. The returned stop
// function cancels it and waits for the goroutine to exit, so once stop
// returns no auto-save is running or can start -- shutdown relies on that to
// make its own save the last one.
func startAutoSave(cfg *config.Config, engine index.Engine) (stop func()) {
	if !cfg.Storage.AutoSaveEnabled {
		log.Info().Msg("auto-save disabled")
		return func() {}
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(cfg.Storage.AutoSaveInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				log.Info().Msg("auto-saving snapshot")
				if err := engine.SaveToFile(ctx, cfg.Storage.DataPath); err != nil {
					log.Error().Err(err).Msg("failed to save snapshot")
				} else {
					log.Info().Msg("snapshot saved")
				}
			}
		}
	}()

	log.Info().Dur("interval", cfg.Storage.AutoSaveInterval).Msg("auto-save enabled")
	return func() {
		cancel()
		<-done
	}
}

// startHTTPServer configures and starts the REST API server.
func startHTTPServer(cfg *config.Config, router http.Handler) *http.Server {
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout,
	}

	go func() {
		log.Info().Str("addr", cfg.Server.Address()).Msg("GoVec server running")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("failed to start server")
		}
	}()

	return srv
}

// waitForInterrupt blocks until a SIGINT or SIGTERM is received.
func waitForInterrupt() {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info().Msg("shutdown signal received")
}

// shutdownServer handles the graceful shutdown of all server components.
//
// Order matters: stop taking requests, stop auto-save, then save. Saving first
// would let requests accepted afterwards land only in the WAL, and a late
// auto-save could follow the final one.
func shutdownServer(cfg *config.Config, engine index.Engine, srv, debugSrv *http.Server, grpcSrv *grpc.Server, stopAutoSave func()) {
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error().Err(err).Msg("server forced to shutdown")
	}

	if debugSrv != nil {
		if err := debugSrv.Shutdown(ctx); err != nil {
			log.Error().Err(err).Msg("pprof server forced to shutdown")
		}
	}

	if grpcSrv != nil {
		grpcSrv.GracefulStop()
		log.Info().Msg("gRPC server stopped")
	}

	stopAutoSave()

	log.Info().Msg("saving data to disk")
	if err := engine.SaveToFile(context.Background(), cfg.Storage.DataPath); err != nil {
		log.Error().Err(err).Msg("error saving data on shutdown")
	} else {
		log.Info().Msg("data saved successfully")
	}

	log.Info().Msg("server stopped")
}

// loadConfiguration handles loading the application configuration from
// file, environment variables, and defaults. It terminates on error.
func loadConfiguration() *config.Config {
	log.Info().Msg("loading configuration")

	configPath := os.Getenv("GOVEC_CONFIG_PATH")
	if configPath == "" {
		configPath = "config.yaml"
	}

	loader := config.NewLoader(configPath)
	cfg, err := loader.Load()
	if err != nil {
		log.Fatal().Err(err).Msg("failed to load configuration")
	}

	log.Info().Int("port", cfg.Server.Port).Str("storage", cfg.Storage.DataPath).Msg("configuration loaded")
	return cfg
}
