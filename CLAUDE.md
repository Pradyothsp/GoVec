# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoVec is a vector database implementation written in Go. The project provides a server that can store and manage vector embeddings with associated metadata through a REST API.

## Getting Started

### Prerequisites

- **Go 1.26.0+** - [Download](https://go.dev/dl/)
- **Task** - [Installation guide](https://taskfile.dev/installation/)
- **gotestsum** - Installed via `task install-gotestsum`

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd govec

# 2. Install dependencies
task deps

# 3. Install gotestsum (if not already installed)
task install-gotestsum

# 4. Run tests to verify setup
task test

# 5. Build the server
task build

# 6. Run the server
task run
```

The server will start on `http://localhost:8000`

### Verify Installation

```bash
task --version       # Should show Task version
gotestsum --version  # Should show gotestsum version
go version           # Should show Go 1.26.0 or compatible
task test            # Should show 103 passing tests
```

## Development Tooling

This project uses modern Go development tools for an enhanced developer experience:

- **[Task](https://taskfile.dev/)** - Task runner for convenient command execution (replaces Makefiles)
- **[gotestsum](https://github.com/gotestyourself/gotestsum)** - Enhanced test output with colors, summaries, and multiple formats
- **Go 1.26.0** - Latest Go version with improved performance and features

### Configuration Files

- **Taskfile.yaml** - Task runner configuration with all project commands
- **.gotestsum.yml** - gotestsum configuration (format, watch settings, slow test threshold)

### Why These Tools?

**Task** provides:
- Simple, readable YAML syntax (vs complex Makefile syntax)
- Cross-platform compatibility
- Built-in parallel execution
- Better error messages
- Native Go integration

**gotestsum** provides:
- Human-optimized, colorful test output
- Test summaries with pass/fail counts
- Multiple output formats (dots, testname, verbose, etc.)
- Watch mode for TDD workflows
- CI/CD integration (JUnit XML export)
- No code changes required (works via `go test -json`)

## Development Commands

### Quick Reference

```bash
task --list          # Show all available commands
task test            # Run tests (most common)
task build           # Build the server
task run             # Run the server
task clean           # Clean build artifacts
```

### Running the Server

```bash
# Using Task (recommended)
task run

# Or directly with Go
go run cmd/server/main.go
```

The server starts on port 8000.

### Building

```bash
# Using Task (recommended)
task build

# Or directly with Go
go build -o govec cmd/server/main.go
```

This creates a `govec` binary in the project root.

### Testing

The project uses [gotestsum](https://github.com/gotestyourself/gotestsum) for improved test output formatting and [Task](https://taskfile.dev/) for convenient command execution.

#### Quick Commands (Recommended)

```bash
# Using Taskfile (recommended)
task test              # Run tests with pretty output
task test:verbose      # Verbose output
task test:watch        # Watch mode (auto re-run on file changes)
task test:coverage     # With coverage report
task test:race         # With race detector
task test:all          # Full test suite (race + coverage)
task test:pkg PKG=internal/core  # Test specific package

# List all available tasks
task --list
```

#### Using gotestsum Directly

```bash
gotestsum                          # Run all tests (default format)
gotestsum --format testname        # Show each test name as it runs
gotestsum --format standard-verbose # Full verbose output
gotestsum --format dots            # Progress dots (. = pass, x = fail)
gotestsum --watch                  # Watch mode
gotestsum -- -cover ./...          # With coverage
gotestsum -- -race ./...           # With race detector
```

#### Traditional Go Test Commands

```bash
go test ./...                      # Run all tests
go test ./internal/core            # Test specific package
go test ./internal/api/handlers    # Test specific package
go test -v ./...                   # Verbose output
go test -cover ./...               # With coverage
go test -race ./...                # With race detector
```

#### Output Formats

- `pkgname-and-test-fails` (default) - Package names + failures only
- `testname` - Shows each test name as it runs
- `standard-verbose` - Full verbose output
- `dots` - Progress dots
- `standard-quiet` - Only failures

#### CI Integration

For CI/CD pipelines, generate JUnit XML:

```bash
gotestsum --junitfile junit.xml --format standard-verbose
```

### Test Organization

The project has comprehensive test coverage organized by package:

**internal/core/models_test.go** (26 tests)
- VectorNode creation and initialization
- Field type validation
- Metadata type handling (strings, numbers, booleans, arrays, nested maps)
- Vector dimension support (2D to 4096D)

**internal/core/index_test.go** (24 tests)
- VectorIndex initialization
- Insert operations with/without metadata
- Concurrent insert operations
- Overwrite behavior
- Edge cases (empty ID, special characters, nil vectors, large dimensions)

**internal/core/persistence_test.go** (26 tests, NEW)
- Save/Load functionality with GOB encoding
- Data integrity for complex nested metadata
- Large vectors (1536D, 4096D) persistence
- Concurrent save operations with RWMutex protection
- File operations (overwrite, invalid paths, corrupted files)
- Atomic write verification with temp file cleanup
- Edge cases (empty index, special characters, long paths)

**internal/api/handlers/vectors_test.go** (50 tests)
- HTTP handler testing
- Request validation
- Response formatting
- Error handling
- Content-Type validation
- Various vector dimensions and metadata types

**internal/test/integration/api_test.go** (14 tests)
- End-to-end API testing
- Full request/response cycle
- Integration between handlers and core logic
- Concurrent HTTP requests
- Vector search functionality

**internal/test/integration/persistence_test.go** (9 tests, NEW)
- Server restart data preservation
- Graceful shutdown save functionality
- Crash recovery with auto-save
- Concurrent HTTP requests during save
- Corrupted file recovery
- Atomic write protection
- Large dataset persistence (1000 vectors)
- Auto-save simulation

**Total: 206 tests, 96.1% coverage in internal/core, 100% in internal/api**

### Dependencies

```bash
# Using Task (recommended)
task deps            # Download and tidy dependencies

# Or directly with Go
go get <package>     # Add a new dependency
go mod tidy          # Update dependencies
go mod vendor        # Vendor dependencies (if needed)
```

### All Available Task Commands

Run `task --list` to see all available commands:

**Testing:**
- `task test` - Run tests with pretty output (default format)
- `task test:verbose` - Run tests with full verbose output
- `task test:watch` - Watch mode (auto re-run tests on file changes)
- `task test:coverage` - Run tests with coverage report (generates HTML)
- `task test:race` - Run tests with race detector
- `task test:all` - Full test suite (race detector + coverage)
- `task test:pkg PKG=internal/core` - Test specific package
- `task test:dots` - Run tests with dots format
- `task test:testname` - Run tests showing each test name

**Building & Running:**
- `task build` - Build the GoVec server binary
- `task run` - Run the GoVec server
- `task clean` - Clean build artifacts and coverage files

**Dependencies:**
- `task deps` - Install and update project dependencies
- `task install-gotestsum` - Install gotestsum (if not present)

### Cleaning Up

```bash
# Using Task (recommended)
task clean           # Removes: govec binary, coverage.out, coverage.html

# Or manually
rm -f govec coverage.out coverage.html
```

## Project Structure

```
govec/
├── cmd/
│   └── server/
│       └── main.go              # Application entry point
├── internal/
│   ├── api/
│   │   ├── router.go            # HTTP router setup
│   │   └── handlers/
│   │       ├── vectors.go       # Vector API handlers
│   │       └── vectors_test.go  # Handler tests
│   ├── core/
│   │   ├── index.go             # VectorIndex implementation
│   │   ├── index_test.go        # Index tests
│   │   ├── models.go            # Core data models (VectorNode)
│   │   └── models_test.go       # Model tests
│   ├── test/
│   │   ├── integration/         # Integration tests
│   │   └── testutil/            # Test utilities
│   ├── config/                  # (Planned) Configuration
│   ├── index/                   # (Planned) Advanced indexing
│   └── storage/                 # (Planned) Persistence layer
├── pkg/
│   └── client/                  # (Planned) Client library
├── Taskfile.yaml                # Task runner configuration
├── .gotestsum.yml               # gotestsum configuration
├── CLAUDE.md                    # This file
├── go.mod                       # Go module definition
└── go.sum                       # Go dependencies lock file
```

## Architecture

### High-Level Structure

The codebase follows a clean architecture pattern with dependency injection:

1. **cmd/server/main.go** - Entry point that initializes the VectorIndex and passes it to the API router
2. **internal/core** - Core domain logic for vector storage
3. **internal/api** - HTTP layer using Gin framework
4. **internal/api/handlers** - HTTP handlers that depend on core services
5. **pkg/client** - (Empty) Planned client library location

### Dependency Flow

```
main.go
  → Creates VectorIndex (core)
  → Passes to SetupRouter (api)
    → Passes to NewVectorHandler (handlers)
      → Handlers call VectorIndex methods
```

The architecture uses **constructor injection**: dependencies are passed explicitly when creating handlers, rather than using global variables or service locators.

### Key Components

**internal/core/models.go**
- `VectorNode` - Represents a vector with ID, vector data ([]float32), and arbitrary metadata

**internal/core/index.go**
- `VectorIndex` - In-memory storage using `map[string]*VectorNode`
- `Insert(id, vec, meta)` - Stores vectors with metadata
- `Search(query, k)` - Finds k nearest neighbors using cosine similarity

**internal/core/persistence.go** (NEW)
- `SaveToFile(path)` - Serializes index to disk using GOB encoding with atomic writes
- `LoadFromFile(path)` - Loads index from disk, gracefully handles missing files
- Atomic write protection using temporary files (.tmp)

**internal/core/similarity.go**
- `CosineSimilarity(a, b)` - Computes cosine similarity between two vectors

**internal/api/router.go**
- `SetupRouter(index)` - Configures Gin routes and injects dependencies into handlers
- Health check at `/health`
- API routes under `/api/v1`

**internal/api/handlers/vectors.go**
- `VectorHandler` - Holds reference to VectorIndex
- `Insert` endpoint - POST `/api/v1/vectors` accepts JSON with id, vector, and optional metadata
- `Search` endpoint - POST `/api/v1/query` accepts JSON with vector and k parameter

**cmd/server/main.go**
- Auto-save ticker (saves every 60 seconds)
- Graceful shutdown with data persistence
- Loads existing data on startup

## Current State

**Implemented:**
- ✅ Basic vector insertion with metadata support
- ✅ In-memory vector storage using VectorIndex
- ✅ REST API with Gin framework
- ✅ **Persistence layer with GOB encoding** (NEW)
  - SaveToFile/LoadFromFile with atomic writes
  - Auto-save every 60 seconds
  - Graceful shutdown saves data to disk
- ✅ **Vector search/query functionality** (NEW)
  - Cosine similarity search
  - Top-K nearest neighbors
  - POST `/api/v1/query` endpoint
- ✅ Comprehensive test suite (206 tests, 96.1% coverage in core modules)
- ✅ Modern development tooling (Task runner, gotestsum)
- ✅ Health check endpoint
- ✅ Clean architecture with dependency injection

**In Progress / Planned:**
- ⏳ Advanced indexing algorithms (HNSW, IVF - internal/index planned)
- ⏳ Storage layer abstraction (internal/storage - planned)
- ⏳ Configuration system (internal/config - planned)
- ⏳ Client library (pkg/client - planned)
- ⏳ Vector deletion and update endpoints
- ⏳ Metadata filtering in search

**Test Coverage:**
- internal/core: 96.1% (120 tests)
- internal/api: 100% (6 tests)
- internal/api/handlers: 100% (50 tests)
- internal/test/integration: 23 tests
- Total: 206 tests passing

## Development Workflow

### Recommended TDD Workflow

```bash
# 1. Start watch mode in one terminal
task test:watch

# 2. Write tests and code in your editor
# Tests automatically re-run on file save

# 3. When done, run full test suite
task test:all

# 4. Review coverage report
open coverage.html
```

### Before Committing

```bash
# Run full test suite with race detector and coverage
task test:all

# Verify no issues
task test:race

# Build to ensure no compilation errors
task build

# Clean up artifacts
task clean
```

### Testing Best Practices

1. **Use `task test` for quick feedback** - Fast, focused on failures
2. **Use `task test:watch` for TDD** - Automatically re-run tests on changes
3. **Use `task test:verbose` when debugging** - See all test output
4. **Use `task test:coverage` to track coverage** - Maintain 100% coverage in core modules
5. **Use `task test:race` before PRs** - Catch concurrency issues early
6. **Use `task test:pkg` for focused testing** - Faster iteration on specific packages

### Debugging Tips

```bash
# Run specific package tests with verbose output
task test:pkg PKG=internal/core
task test:verbose

# Run with race detector to catch concurrency issues
task test:race

# Check coverage for specific package
gotestsum -- -cover -coverprofile=coverage.out ./internal/core
go tool cover -html=coverage.out
```

## Troubleshooting

### Task not found

```bash
# macOS
brew install go-task

# Other platforms
go install github.com/go-task/task/v3/cmd/task@latest

# Verify installation
task --version
```

### gotestsum not found

```bash
# Install via Task
task install-gotestsum

# Or install directly
go install gotest.tools/gotestsum@latest

# Verify installation
gotestsum --version
```

### Tests failing

```bash
# Run with verbose output to see details
task test:verbose

# Run specific failing package
task test:pkg PKG=internal/core

# Clean and re-run
task clean
task test
```

### Port 8000 already in use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in code (update main.go)
```

## Go Version

The project uses Go 1.26.0. Ensure your local Go version is compatible.

```bash
# Check your Go version
go version

# Should output: go version go1.26.0 or higher
```
