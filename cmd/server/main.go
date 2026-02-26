package main

import (
	"context"
	"net/http"
	_ "net/http/pprof" //nolint:gosec // G108: pprof only starts when cfg.Server.PprofEnabled is true; addr is localhost-bound
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/config"
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
	// Structured JSON logging for production
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnixMs
	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Logger()

	// Load configuration
	cfg := loadConfiguration()

	// Apply configured log level (parsed by zerolog; defaults to "info")
	level, err := zerolog.ParseLevel(cfg.Server.LogLevel)
	if err != nil {
		level = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(level)
	log.Info().Str("level", level.String()).Msg("log level set")

	// pprof debug server — config-driven, localhost-bound only
	var debugSrv *http.Server
	if cfg.Server.PprofEnabled {
		debugSrv = &http.Server{
			Addr:              cfg.Server.PprofAddr,
			ReadHeaderTimeout: 5 * time.Second, // prevent Slowloris on debug port
			Handler:           http.DefaultServeMux,
		}
		go func() {
			log.Info().Str("addr", cfg.Server.PprofAddr).Msg("pprof debug server listening")
			if err := debugSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed { //nolint:gosec // localhost-bound
				log.Error().Err(err).Msg("pprof server error")
			}
		}()
	}

	// Initialize WAL
	wal, err := index.NewWAL(cfg.Storage.WalPath)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to open WAL")
	}

	log.Info().Msg("starting server")

	// Create engine via factory
	engine, err := index.NewEngine(cfg.Engine, cfg.Storage, wal)
	if err != nil {
		_ = wal.Close() //nolint:errcheck // best-effort cleanup before fatal exit
		log.Fatal().Err(err).Msg("failed to create engine")
	}

	// Engine created successfully, set up cleanup
	defer wal.Close() //nolint:errcheck // best-effort cleanup on shutdown

	router := api.SetupRouter(engine, cfg.Server.APIKey, cfg.Storage.DataPath)

	// RECOVERY SEQUENCE
	// Step 1: Load the base snapshot from DataPath (GOB format)
	log.Info().Str("path", cfg.Storage.DataPath).Msg("loading snapshot from disk")
	if err := engine.LoadFromFile(cfg.Storage.DataPath); err != nil {
		if os.IsNotExist(err) {
			log.Info().Msg("no snapshot found, starting fresh")
		} else {
			log.Warn().Err(err).Msg("could not load snapshot")
		}
	} else {
		log.Info().Int("vectors", engine.Len()).Msg("snapshot loaded")
	}

	// Step 2: Replay the WAL from WalPath (JSON format) to recover uncommitted changes
	log.Info().Str("path", cfg.Storage.WalPath).Msg("replaying WAL")
	if err := engine.ReplayWAL(cfg.Storage.WalPath); err != nil {
		log.Warn().Err(err).Msg("WAL replay warning")
	} else {
		log.Info().Int("vectors", engine.Len()).Msg("WAL replay complete")
	}

	// Start Background Snapshotting (The "Auto-Save")
	if cfg.Storage.AutoSaveEnabled {
		go func() {
			ticker := time.NewTicker(cfg.Storage.AutoSaveInterval)
			defer ticker.Stop()
			for range ticker.C {
				log.Info().Msg("auto-saving snapshot")
				if err := engine.SaveToFile(cfg.Storage.DataPath); err != nil {
					log.Error().Err(err).Msg("failed to save snapshot")
				} else {
					log.Info().Msg("snapshot saved")
				}
			}
		}()
		log.Info().Dur("interval", cfg.Storage.AutoSaveInterval).Msg("auto-save enabled")
	} else {
		log.Info().Msg("auto-save disabled")
	}

	// Create HTTP server
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: cfg.Server.ShutdownTimeout,
	}

	// Run server in goroutine
	go func() {
		log.Info().Str("addr", cfg.Server.Address()).Msg("GoVec server running")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("failed to start server")
		}
	}()

	// Setup signal handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info().Msg("shutdown signal received")

	// Save data before shutdown
	log.Info().Msg("saving data to disk")
	if err := engine.SaveToFile(cfg.Storage.DataPath); err != nil {
		log.Error().Err(err).Msg("error saving data on shutdown")
	} else {
		log.Info().Msg("data saved successfully")
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error().Err(err).Msg("server forced to shutdown")
		return
	}

	if debugSrv != nil {
		if err := debugSrv.Shutdown(ctx); err != nil {
			log.Error().Err(err).Msg("pprof server forced to shutdown")
		}
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
