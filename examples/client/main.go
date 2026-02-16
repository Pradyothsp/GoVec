package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// This simulates what OpenAI or HuggingFace would return for these texts
var mockEmbeddings = map[string][]float32{
	"How to reset password": {0.95, 0.05, 0.02}, // Mostly "Security" dimension
	"Billing and payments":  {0.03, 0.98, 0.04}, // Mostly "Money" dimension
	"Office location map":   {0.05, 0.02, 0.91}, // Mostly "Location" dimension

	// The User Query: "I forgot my access code"
	// Notice it is mathematically closest to "Reset password"
	"user_query": {0.89, 0.08, 0.05},
}

type VectorPayload struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata"`
}

type SearchPayload struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

func main() {
	baseURL := "http://localhost:8000/api/v1"
	fmt.Println("🚀 Starting Help Desk Demo...")

	// 1. Ingest Knowledge Base (The "Training" Phase)
	kb := []struct {
		Text string
		Type string
	}{
		{"How to reset password", "security"},
		{"Billing and payments", "finance"},
		{"Office location map", "general"},
	}

	for i, item := range kb {
		fmt.Printf("📥 Indexing: %s...\n", item.Text)

		payload := VectorPayload{
			ID:       fmt.Sprintf("doc_%d", i),
			Vector:   mockEmbeddings[item.Text],
			Metadata: map[string]interface{}{"content": item.Text, "category": item.Type},
		}

		sendRequest("POST", baseURL+"/vectors", payload)
	}

	// 2. The User Asks a Question
	queryText := "I forgot my access code"
	fmt.Printf("\n❓ User asks: '%s'\n", queryText)
	fmt.Println("🔍 Searching GoVec for semantic matches...")

	queryPayload := SearchPayload{
		Vector: mockEmbeddings["user_query"],
		K:      1, // Just give me the best answer
	}

	sendRequest("POST", baseURL+"/query", queryPayload)
}

// Helper to send HTTP requests
func sendRequest(method, url string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	req, _ := http.NewRequest(method, url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	// Pretty print the JSON response
	var prettyJSON bytes.Buffer
	json.Indent(&prettyJSON, body, "", "  ")
	fmt.Println("👉 Response:", prettyJSON.String())
}
