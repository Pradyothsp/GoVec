package main

import (
	"log"

	"github.com/Pradyothsp/govec/internal/api"
	"github.com/Pradyothsp/govec/internal/core"
)

func main() {
	log.Println("Starting server...")

	// Initialize Vector Index
	index := core.NewVectorIndex()

	// Initialize Router
	router := api.SetupRouter(index)

	log.Println("GoVec is starting on :8000...")
	if err := router.Run(":8000"); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
