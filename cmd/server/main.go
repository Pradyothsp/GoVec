package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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
	// Load configuration
	cfg := loadConfiguration()

	// Initialize WAL
	wal, err := index.NewWAL(cfg.Storage.WalPath)
	if err != nil {
		log.Fatal("Failed to open WAL:", err)
	}

	// Initialize components
	log.Println("Starting server...")

	// Create engine via factory
	engine, err := index.NewEngine(cfg.Engine, cfg.Storage, wal)
	if err != nil {
		_ = wal.Close() //nolint:errcheck // best-effort cleanup before fatal exit
		log.Fatalf("Failed to create engine: %v", err)
	}

	// Engine created successfully, set up cleanup
	defer wal.Close() //nolint:errcheck // best-effort cleanup on shutdown

	router := api.SetupRouter(engine, cfg.Server.APIKey)

	// RECOVERY SEQUENCE
	// Step 1: Load the base snapshot from DataPath (GOB format)
	log.Println("📂 Loading snapshot from disk...")
	if err := engine.LoadFromFile(cfg.Storage.DataPath); err != nil {
		if os.IsNotExist(err) {
			log.Println("⚠️ No snapshot found, starting fresh.")
		} else {
			log.Printf("⚠️ Warning: Could not load snapshot: %v", err)
		}
	} else {
		log.Printf("✅ Loaded %d vectors from snapshot!", engine.Len())
	}

	// Step 2: Replay the WAL from WalPath (JSON format) to recover uncommitted changes
	log.Println("🔄 Replaying WAL...")
	if err := engine.ReplayWAL(cfg.Storage.WalPath); err != nil {
		log.Printf("⚠️ WAL Replay warning: %v", err)
	} else {
		log.Printf("✅ WAL replay complete. Total vectors: %d", engine.Len())
	}

	// Start Background Snapshotting (The "Auto-Save")
	if cfg.Storage.AutoSaveEnabled {
		go func() {
			ticker := time.NewTicker(cfg.Storage.AutoSaveInterval)
			defer ticker.Stop()
			for range ticker.C {
				log.Println("💾 Auto-saving snapshot...")
				if err := engine.SaveToFile(cfg.Storage.DataPath); err != nil {
					log.Printf("❌ Failed to save snapshot: %v", err)
				} else {
					log.Println("✅ Snapshot saved.")
				}
			}
		}()
		log.Printf("🔄 Auto-save enabled (interval: %v)", cfg.Storage.AutoSaveInterval)
	} else {
		log.Println("⏸️  Auto-save disabled")
	}

	// Create HTTP server
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: cfg.Server.ShutdownTimeout,
	}

	// Run server in goroutine
	go func() {
		log.Printf("GoVec server is running on %s\n", cfg.Server.Address())
		log.Println("Press Ctrl+C to shutdown gracefully")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Setup signal handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// Graceful shutdown
	log.Println("Received shutdown signal, shutting down gracefully...")

	// Save data before shutdown
	log.Println("💾 Saving data to disk...")
	if err := engine.SaveToFile(cfg.Storage.DataPath); err != nil {
		log.Printf("❌ Error saving data on shutdown: %v", err)
	} else {
		log.Println("✅ Data saved successfully")
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Println("Server forced to shutdown:", err)
		return
	}

	log.Println("All requests completed, server stopped")
}

// loadConfiguration handles loading the application configuration from
// file, environment variables, and defaults. It terminates on error.
func loadConfiguration() *config.Config {
	log.Println("Loading configuration...")

	configPath := os.Getenv("GOVEC_CONFIG_PATH")
	if configPath == "" {
		configPath = "config.yaml"
	}

	loader := config.NewLoader(configPath)
	cfg, err := loader.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	log.Printf("✅ Configuration loaded: port=%d, storage=%s", cfg.Server.Port, cfg.Storage.DataPath)
	return cfg
}
