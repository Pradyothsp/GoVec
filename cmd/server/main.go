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
	"github.com/Pradyothsp/govec/internal/core"
)

const StoragePath = "./govec_data.bin"

const (
	ServerPort      = ":8000"
	ShutdownTimeout = 10 * time.Second
)

func main() {
	// 1. Initialize components
	log.Println("Starting server...")
	index := core.NewVectorIndex()
	router := api.SetupRouter(index)

	// 2. Load existing data (Persistence)
	log.Println("📂 Loading data from disk...")
	if err := index.LoadFromFile(StoragePath); err != nil {
		log.Printf("⚠️ Warning: Could not load index: %v", err)
	} else {
		log.Printf("✅ Loaded %d vectors from disk!", len(index.Store))
	}

	// 3. Start Background Snapshotting (The "Auto-Save")
	go func() {
		ticker := time.NewTicker(60 * time.Second) // Save every 60s
		for range ticker.C {
			log.Println("💾 Auto-saving snapshot...")
			if err := index.SaveToFile(StoragePath); err != nil {
				log.Printf("❌ Failed to save snapshot: %v", err)
			} else {
				log.Println("✅ Snapshot saved.")
			}
		}
	}()

	// 3. Create HTTP server
	srv := &http.Server{
		Addr:    ServerPort,
		Handler: router,
	}

	// 4. Run server in goroutine
	go func() {
		log.Printf("GoVec server is running on %s\n", ServerPort)
		log.Println("Press Ctrl+C to shutdown gracefully")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// 5. Setup signal handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// 6. Graceful shutdown
	log.Println("Received shutdown signal, shutting down gracefully...")

	// Save data before shutdown
	log.Println("💾 Saving data to disk...")
	if err := index.SaveToFile(StoragePath); err != nil {
		log.Printf("❌ Error saving data on shutdown: %v", err)
	} else {
		log.Println("✅ Data saved successfully")
	}

	ctx, cancel := context.WithTimeout(context.Background(), ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("All requests completed, server stopped")
}
