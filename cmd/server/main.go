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
	"github.com/Pradyothsp/govec/internal/core"
)

// @title           GoVec API
// @version         1.0
// @description     A vector database REST API for storing and querying vector embeddings.
// @host            localhost:8000
// @BasePath        /
// @schemes         http
func main() {
	// 1. Load configuration
	cfg := loadConfiguration()

	// 2. Initialize components
	log.Println("Starting server...")
	index := core.NewVectorIndex()
	router := api.SetupRouter(index)

	// 3. Load existing data (Persistence)
	log.Println("📂 Loading data from disk...")
	if err := index.LoadFromFile(cfg.Storage.DataPath); err != nil {
		log.Printf("⚠️ Warning: Could not load index: %v", err)
	} else {
		log.Printf("✅ Loaded %d vectors from disk!", len(index.Store))
	}

	// 4. Start Background Snapshotting (The "Auto-Save")
	if cfg.Storage.AutoSaveEnabled {
		go func() {
			ticker := time.NewTicker(cfg.Storage.AutoSaveInterval)
			defer ticker.Stop()
			for range ticker.C {
				log.Println("💾 Auto-saving snapshot...")
				if err := index.SaveToFile(cfg.Storage.DataPath); err != nil {
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

	// 5. Create HTTP server
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: cfg.Server.ShutdownTimeout,
	}

	// 6. Run server in goroutine
	go func() {
		log.Printf("GoVec server is running on %s\n", cfg.Server.Address())
		log.Println("Press Ctrl+C to shutdown gracefully")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// 7. Setup signal handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// 8. Graceful shutdown
	log.Println("Received shutdown signal, shutting down gracefully...")

	// Save data before shutdown
	log.Println("💾 Saving data to disk...")
	if err := index.SaveToFile(cfg.Storage.DataPath); err != nil {
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
