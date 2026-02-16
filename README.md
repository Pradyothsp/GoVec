# GoVec

A high-performance vector database implementation written in Go, designed for storing and managing vector embeddings with associated metadata.

[![Go Version](https://img.shields.io/badge/Go-1.26.0-00ADD8?style=flat&logo=go)](https://go.dev/)
[![CI](https://github.com/Pradyothsp/govec/actions/workflows/go.yml/badge.svg)](https://github.com/Pradyothsp/govec/actions/workflows/go.yml)
[![Tests](https://img.shields.io/badge/tests-103%20passing-success)](/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](/)

## Features

- **Fast In-Memory Storage** - Optimized vector storage using Go maps
- **Thread-Safe Operations** - Concurrent access protection with `sync.RWMutex`
- **Flexible Metadata** - Store arbitrary JSON metadata alongside vectors
- **RESTful API** - Simple HTTP API built with Gin framework
- **Clean Architecture** - Dependency injection and separation of concerns
- **Comprehensive Tests** - 103 tests with 100% coverage in core modules
- **Modern Tooling** - Task runner and enhanced test output with gotestsum

## Quick Start

### Prerequisites

- Go 1.26.0 or higher
- [Task](https://taskfile.dev/) (optional, for convenient commands)
- [gotestsum](https://github.com/gotestyourself/gotestsum) (optional, for better test output)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd govec

# Install dependencies
go mod download

# Or use Task
task deps
```

### Running the Server

```bash
# Using Go
go run cmd/server/main.go

# Or using Task
task run
```

The server will start on `http://localhost:8000`

### Building

```bash
# Build binary
go build -o govec cmd/server/main.go

# Or using Task
task build

# Run the binary
./govec
```

## Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok"
}
```

### Insert a Vector

```bash
curl -X POST http://localhost:8000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "vec1",
    "vector": [0.1, 0.2, 0.3, 0.4],
    "metadata": {
      "label": "example",
      "category": "test",
      "score": 0.95
    }
  }'
```

Response:
```json
{
  "message": "Vector inserted successfully",
  "id": "vec1"
}
```

### Insert Vector Without Metadata

```bash
curl -X POST http://localhost:8000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "vec2",
    "vector": [0.5, 0.6, 0.7, 0.8]
  }'
```

### Supported Vector Dimensions

GoVec supports vectors of any dimension, from small 2D vectors to large embeddings:

- **Small**: 2D, 3D (spatial coordinates)
- **Medium**: 128D, 384D (sentence embeddings)
- **Large**: 512D, 768D, 1536D (language models like BERT, OpenAI)
- **Extra Large**: 4096D+ (advanced embeddings)

### Metadata Types

Metadata can include various JSON types:

- **Strings**: `"label": "text"`
- **Numbers**: `"score": 0.95`, `"count": 42`
- **Booleans**: `"active": true`
- **Arrays**: `"tags": ["tag1", "tag2"]`
- **Nested Objects**: `"info": {"key": "value"}`

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/vectors` | Insert or update a vector |

### POST `/api/v1/vectors`

**Request Body:**

```json
{
  "id": "string (required)",
  "vector": [float32] (required),
  "metadata": {
    "key": "value"
  } (optional)
}
```

**Response (200 OK):**

```json
{
  "message": "Vector inserted successfully",
  "id": "string"
}
```

**Error Responses:**

- `400 Bad Request` - Invalid JSON or missing required fields
- `415 Unsupported Media Type` - Missing `Content-Type: application/json` header

## Development

### Project Structure

```
govec/
├── cmd/server/          # Application entry point
├── internal/
│   ├── api/            # HTTP layer (Gin router)
│   ├── core/           # Core domain logic (VectorIndex)
│   └── test/           # Test utilities and integration tests
├── Taskfile.yaml       # Task runner configuration
└── go.mod              # Go module definition
```

### Running Tests

```bash
# Quick test run
task test

# Watch mode (auto re-run on changes)
task test:watch

# With coverage report
task test:coverage

# With race detector
task test:race

# Full test suite
task test:all

# Test specific package
task test:pkg PKG=internal/core

# Or using go test directly
go test ./...
go test -v ./...
go test -cover ./...
go test -race ./...
```

### Available Commands

Run `task --list` to see all available commands:

```
task test              # Run tests with pretty output
task test:verbose      # Verbose test output
task test:watch        # Watch mode (TDD)
task test:coverage     # Generate coverage report
task test:race         # Run with race detector
task build             # Build the server binary
task run               # Run the server
task clean             # Clean build artifacts
task deps              # Install dependencies
```

### Test Coverage

- **internal/core**: 100% (50 tests)
- **internal/api**: 100% (3 tests)
- **internal/api/handlers**: 100% (50 tests)
- **Total**: 103 tests, all passing

### Architecture

GoVec follows clean architecture principles with dependency injection:

```
main.go
  ├─> Creates VectorIndex (core)
  └─> Passes to SetupRouter (api)
       └─> Passes to NewVectorHandler (handlers)
            └─> Handlers call VectorIndex methods
```

**Key Components:**

- **VectorNode** (`internal/core/models.go`) - Data model for vectors with metadata
- **VectorIndex** (`internal/core/index.go`) - Thread-safe in-memory vector storage
- **VectorHandler** (`internal/api/handlers/vectors.go`) - HTTP request handlers
- **SetupRouter** (`internal/api/router.go`) - API routing configuration

## Roadmap

- [ ] Vector search and similarity queries (cosine similarity, euclidean distance)
- [ ] Persistence layer (disk storage, WAL)
- [ ] Advanced indexing algorithms (HNSW, IVF)
- [ ] Batch operations (bulk insert, batch search)
- [ ] Query filters based on metadata
- [ ] Configuration system (YAML/JSON config files)
- [ ] Go client library
- [ ] CLI tool for management
- [ ] Horizontal scaling support
- [ ] Metrics and observability

## Performance

Current implementation focuses on correctness and thread-safety:

- **Thread-safe**: All operations protected with `sync.RWMutex`
- **In-memory**: Fast O(1) insert and lookup
- **Zero-copy**: Stores vector pointers for memory efficiency
- **Concurrent**: Supports multiple simultaneous operations

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `task test:all`
2. No race conditions: `task test:race`
3. Maintain 100% coverage in core modules
4. Follow Go best practices and existing code style

## License

[Add your license here]

## Acknowledgments

Built with:
- [Gin](https://github.com/gin-gonic/gin) - HTTP web framework
- [Task](https://taskfile.dev/) - Task runner
- [gotestsum](https://github.com/gotestyourself/gotestsum) - Enhanced test output

---

**Status**: Early development (in-memory only, no search yet)

For detailed development guidelines, see [CLAUDE.md](CLAUDE.md)
