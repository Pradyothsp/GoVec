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

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/config"
	"github.com/Pradyothsp/govec/internal/grpcserver"
	"github.com/Pradyothsp/govec/internal/index"
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

	log.Info().Msg("starting server")

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

	router := api.SetupRouter(engine, cfg.Server.APIKey, cfg.Storage.DataPath)
	grpcSrv, err := startGRPCServer(cfg, engine)
	if err != nil {
		if closeErr := wal.Close(); closeErr != nil {
			log.Error().Err(closeErr).Msg("failed to close WAL after gRPC startup failure")
		}
		log.Fatal().Err(err).Msg("failed to start gRPC server") //nolint:gocritic // exitAfterDefer: WAL is explicitly closed above before fatal exit
	}

	// RECOVERY SEQUENCE
	if err := runRecovery(cfg, engine); err != nil {
		if closeErr := wal.Close(); closeErr != nil {
			log.Error().Err(closeErr).Msg("failed to close WAL after recovery failure")
		}
		log.Fatal().Err(err).Msg("refusing to start") //nolint:gocritic // exitAfterDefer: WAL is explicitly closed above before fatal exit
	}

	// Start Background Snapshotting (The "Auto-Save")
	startAutoSave(cfg, engine)

	// Create and start an HTTP server
	srv := startHTTPServer(cfg, router)

	// Wait for the termination signal
	waitForInterrupt()

	// Graceful shutdown
	shutdownServer(cfg, engine, srv, debugSrv, grpcSrv)
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
		log.Warn().Err(err).Msg("WAL replay warning")
	} else {
		log.Info().Int("vectors", engine.Len()).Msg("WAL replay complete")
	}
	return nil
}

// startAutoSave starts the background snapshotting ticker.
func startAutoSave(cfg *config.Config, engine index.Engine) {
	if !cfg.Storage.AutoSaveEnabled {
		log.Info().Msg("auto-save disabled")
		return
	}

	go func() {
		ticker := time.NewTicker(cfg.Storage.AutoSaveInterval)
		defer ticker.Stop()
		for range ticker.C {
			log.Info().Msg("auto-saving snapshot")
			if err := engine.SaveToFile(context.Background(), cfg.Storage.DataPath); err != nil {
				log.Error().Err(err).Msg("failed to save snapshot")
			} else {
				log.Info().Msg("snapshot saved")
			}
		}
	}()

	log.Info().Dur("interval", cfg.Storage.AutoSaveInterval).Msg("auto-save enabled")
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
func shutdownServer(cfg *config.Config, engine index.Engine, srv, debugSrv *http.Server, grpcSrv *grpc.Server) {
	// Save data before shutdown
	log.Info().Msg("saving data to disk")
	if err := engine.SaveToFile(context.Background(), cfg.Storage.DataPath); err != nil {
		log.Error().Err(err).Msg("error saving data on shutdown")
	} else {
		log.Info().Msg("data saved successfully")
	}

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
