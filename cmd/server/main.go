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

const (
	ServerPort      = ":8000"
	ShutdownTimeout = 10 * time.Second
)

func main() {
	// 1. Initialize components
	log.Println("Starting server...")
	index := core.NewVectorIndex()
	router := api.SetupRouter(index)

	// 2. Create HTTP server
	srv := &http.Server{
		Addr:    ServerPort,
		Handler: router,
	}

	// 3. Run server in goroutine
	go func() {
		log.Printf("GoVec server is running on %s\n", ServerPort)
		log.Println("Press Ctrl+C to shutdown gracefully")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// 4. Setup signal handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// 5. Graceful shutdown
	log.Println("Received shutdown signal, shutting down gracefully...")

	ctx, cancel := context.WithTimeout(context.Background(), ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("All requests completed, server stopped")
}
