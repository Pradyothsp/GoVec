# GoVec

A high-performance vector database implementation written in Go, designed for storing and managing vector embeddings with associated metadata.

[![Go Version](https://img.shields.io/badge/Go-1.26.0-00ADD8?style=flat&logo=go)](https://go.dev/)
[![CI](https://github.com/Pradyothsp/govec/actions/workflows/go.yml/badge.svg)](https://github.com/Pradyothsp/govec/actions/workflows/go.yml)
[![Tests](https://img.shields.io/badge/tests-255%20passing-success)](/)
[![Coverage](https://img.shields.io/badge/coverage-98.1%25-brightgreen)](/)

## Features

- **Fast In-Memory Storage** - Optimized vector storage using Go maps
- **Thread-Safe Operations** - Concurrent access protection with `sync.RWMutex`
- **Flexible Metadata** - Store arbitrary JSON metadata alongside vectors
- **Vector Search** - Cosine similarity search with top-K nearest neighbors
- **Persistence Layer** - Automatic snapshots with GOB encoding and graceful shutdown
- **Configuration System** - YAML-based config with environment variable overrides
- **RESTful API** - Simple HTTP API built with Gin framework
- **Clean Architecture** - DDD principles with dependency injection
- **Comprehensive Tests** - 255 tests with 98.1% coverage
- **Modern Tooling** - Task runner and enhanced test output with gotestsum
- **CI/CD Pipeline** - Separate lint, test, vet, and security jobs via GitHub Actions
- **Docker Support** - Multi-stage Dockerfile for production deployments
- **Linting & Security** - golangci-lint, go vet, govulncheck, and pre-commit hooks

## Quick Start

### Prerequisites

- Go 1.26.0 or higher
- [Task](https://taskfile.dev/) (optional, for convenient commands)
- [gotestsum](https://github.com/gotestyourself/gotestsum) (optional, for better test output)
- [golangci-lint](https://golangci-lint.run/) (optional, for linting)
- [govulncheck](https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck) (optional, for vulnerability scanning)
- [pre-commit](https://pre-commit.com/) (optional, for commit hooks)

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

### Configuration

GoVec uses a YAML configuration file with environment variable overrides.

**Quick configuration:**

```bash
# Use default config (config.yaml)
task run

# Custom port
GOVEC_SERVER_PORT=9000 task run

# Disable auto-save
GOVEC_AUTO_SAVE_ENABLED=false task run

# Custom config file
GOVEC_CONFIG_PATH=./custom.yaml task run
```

**Available environment variables:**
- `GOVEC_SERVER_PORT` - Server port (default: 8000)
- `GOVEC_STORAGE_PATH` - Data file path (default: ./govec_data.bin)
- `GOVEC_AUTO_SAVE_ENABLED` - Enable/disable auto-save (default: true)
- `GOVEC_AUTO_SAVE_INTERVAL` - Auto-save interval (default: 60s)

See `config.example.yaml` for all available options.

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
| POST | `/api/v1/query` | Search for similar vectors |

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

### POST `/api/v1/query`

Search for vectors similar to a query vector using cosine similarity.

**Request Body:**

```json
{
  "vector": [float32] (required),
  "k": integer (optional, default: all results)
}
```

**Response (200 OK):**

```json
[
  {
    "ID": "vec1",
    "Score": 0.98,
    "Meta": {
      "label": "example"
    }
  },
  {
    "ID": "vec2",
    "Score": 0.87,
    "Meta": {}
  }
]
```

Results are sorted by similarity score (descending).

**Error Responses:**

- `400 Bad Request` - Invalid JSON or missing vector
- `500 Internal Server Error` - Empty query vector or dimension mismatch

## Development

### Project Structure

```
govec/
├── cmd/server/          # Application entry point
├── internal/
│   ├── api/            # HTTP layer (Gin router)
│   ├── config/         # Configuration system (DDD)
│   ├── core/           # Core domain logic (VectorIndex)
│   └── test/           # Test utilities and integration tests
├── .github/workflows/  # CI/CD pipeline (lint, test, vet, security)
├── .golangci.yml       # golangci-lint configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── Dockerfile          # Multi-stage production image
├── config.yaml         # Default configuration
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
# Testing
task test              # Run tests with pretty output
task test:verbose      # Verbose test output
task test:watch        # Watch mode (TDD)
task test:coverage     # Generate coverage report
task test:race         # Run with race detector
task test:all          # Full suite (race + coverage)
task test:pkg PKG=...  # Test a specific package

# Static analysis
task lint              # Run golangci-lint
task vet               # Run go vet
task fmt:check         # Check formatting (non-destructive)
task fmt:fix           # Apply gofmt formatting
task vuln              # Run govulncheck
task check             # Run all static checks (lint + vet + fmt + vuln)

# Build & run
task build             # Build the server binary
task run               # Run the server
task clean             # Clean build artifacts

# Dependencies & tools
task deps              # Install/tidy dependencies
task install-gotestsum # Install gotestsum
task install-tools     # Install all dev tools (gotestsum, golangci-lint, govulncheck)
```

### Test Coverage

- **internal/config**: 98.1% (49 tests)
- **internal/core**: 96.1% (76 tests)
- **internal/api**: 100% (13 tests)
- **internal/api/handlers**: 100% (50 tests)
- **internal/test/integration**: 67 tests
- **Total**: 255 tests, all passing

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

- [x] Vector search and similarity queries (cosine similarity)
- [x] Persistence layer (GOB encoding with atomic writes)
- [x] Configuration system (YAML config with env var overrides)
- [ ] Advanced indexing algorithms (HNSW, IVF)
- [ ] Batch operations (bulk insert, batch search)
- [ ] Query filters based on metadata
- [ ] Euclidean distance and other similarity metrics
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
4. Pass all static checks: `task check`
5. Follow Go best practices and existing code style

### Setting Up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Docker

```bash
# Build the image
docker build -t govec .

# Run the container
docker run -p 8000:8000 govec
```

## License

[Add your license here]

## Acknowledgments

Built with:
- [Gin](https://github.com/gin-gonic/gin) - HTTP web framework
- [Task](https://taskfile.dev/) - Task runner
- [gotestsum](https://github.com/gotestyourself/gotestsum) - Enhanced test output
- [golangci-lint](https://golangci-lint.run/) - Go linter aggregator
- [govulncheck](https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck) - Go vulnerability scanner
- [pre-commit](https://pre-commit.com/) - Git hook framework

---

**Status**: Active development - Production-ready features: vector storage, search, persistence, and configuration

For detailed development guidelines, see [CLAUDE.md](CLAUDE.md)
