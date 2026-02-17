# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoVec is a vector database implementation written in Go. The project provides a server that can store and manage vector embeddings with associated metadata through a REST API.

## Getting Started

### Prerequisites

- **Go 1.26.0+** - [Download](https://go.dev/dl/)
- **Task** - [Installation guide](https://taskfile.dev/installation/)
- **gotestsum** - Installed via `task install-gotestsum`
- **golangci-lint** - Installed via `task install-tools`
- **govulncheck** - Installed via `task install-tools`
- **pre-commit** (optional) - `pip install pre-commit && pre-commit install`

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
task --version           # Should show Task version
gotestsum --version      # Should show gotestsum version
go version               # Should show Go 1.26.0 or compatible
golangci-lint --version  # Should show golangci-lint version
govulncheck -version     # Should show govulncheck version
task test                # Should show 255 passing tests
```

## Development Tooling

This project uses modern Go development tools for an enhanced developer experience:

- **[Task](https://taskfile.dev/)** - Task runner for convenient command execution (replaces Makefiles)
- **[gotestsum](https://github.com/gotestyourself/gotestsum)** - Enhanced test output with colors, summaries, and multiple formats
- **[golangci-lint](https://golangci-lint.run/)** - Aggregated Go linter (errcheck, govet, staticcheck, gosec, gocritic, and more)
- **[govulncheck](https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck)** - Go vulnerability scanner
- **[pre-commit](https://pre-commit.com/)** - Git hook framework for file hygiene and Go tooling
- **Go 1.26.0** - Latest Go version with improved performance and features

### Configuration Files

- **Taskfile.yaml** - Task runner configuration with all project commands
- **.gotestsum.yml** - gotestsum configuration (format, watch settings, slow test threshold)
- **.golangci.yml** - golangci-lint linter selection and settings
- **.pre-commit-config.yaml** - Pre-commit hooks (trailing whitespace, gofmt, go vet, golangci-lint, govulncheck)
- **Dockerfile** - Multi-stage build for production container images

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

**golangci-lint** provides:
- Runs many linters in a single pass (fast)
- Catches correctness issues (errcheck, staticcheck), security (gosec), and style (revive, gocritic)
- Configured via `.golangci.yml` with sensible defaults and test-file exclusions

**govulncheck** provides:
- Scans Go modules against the Go vulnerability database
- Reports only reachable vulnerabilities (not noisy)
- Integrated into CI and pre-commit hooks

## Development Commands

### Quick Reference

```bash
task --list          # Show all available commands
task test            # Run tests (most common)
task check           # Run all static checks (lint + vet + fmt + vuln)
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

**internal/test/integration/persistence_test.go** (9 tests)
- Server restart data preservation
- Graceful shutdown save functionality
- Crash recovery with auto-save
- Concurrent HTTP requests during save
- Corrupted file recovery
- Atomic write protection
- Large dataset persistence (1000 vectors)
- Auto-save simulation

**internal/config/config_test.go** (2 tests)
- Default configuration values
- ServerConfig.Address() helper method

**internal/config/validation_test.go** (3 tests)
- Complete configuration validation
- Server config validation (port ranges, timeout constraints)
- Storage config validation (path, auto-save settings)

**internal/config/loader_test.go** (8 tests)
- YAML file loading (valid, invalid, missing files)
- Environment variable overrides
- Configuration priority (env > file > defaults)
- Boolean and duration parsing
- Partial YAML configuration
- Invalid configuration handling

**Total: 255 tests, 98.1% coverage in internal/config, 96.1% in internal/core, 100% in internal/api**

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

**Static Analysis:**
- `task lint` - Run golangci-lint
- `task vet` - Run go vet
- `task fmt:check` - Check formatting without modifying files (exits non-zero if unformatted)
- `task fmt:fix` - Apply gofmt formatting to all Go source files
- `task vuln` - Run govulncheck for known vulnerabilities
- `task check` - Run all static checks (vet + fmt + lint + vuln)

**Building & Running:**
- `task build` - Build the GoVec server binary
- `task run` - Run the GoVec server
- `task clean` - Clean build artifacts and coverage files

**Dependencies & Tools:**
- `task deps` - Install and update project dependencies
- `task install-gotestsum` - Install gotestsum (if not present)
- `task install-tools` - Install all dev tools (gotestsum, golangci-lint, govulncheck)

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
│   │   ├── router_test.go       # Router tests
│   │   └── handlers/
│   │       ├── vectors.go       # Vector API handlers
│   │       └── vectors_test.go  # Handler tests
│   ├── config/                  # Configuration system
│   │   ├── config.go            # Config structs (value objects)
│   │   ├── loader.go            # Config loading service
│   │   ├── validation.go        # Domain validation rules
│   │   ├── config_test.go       # Config struct tests
│   │   ├── loader_test.go       # Loader tests
│   │   └── validation_test.go   # Validation tests
│   ├── core/
│   │   ├── index.go             # VectorIndex implementation
│   │   ├── index_test.go        # Index tests
│   │   ├── models.go            # Core data models (VectorNode)
│   │   ├── models_test.go       # Model tests
│   │   ├── persistence.go       # Save/Load functionality
│   │   ├── persistence_test.go  # Persistence tests
│   │   └── similarity.go        # Cosine similarity
│   ├── test/
│   │   ├── integration/         # Integration tests
│   │   └── testutil/            # Test utilities
│   ├── index/                   # (Planned) Advanced indexing
│   └── storage/                 # (Planned) Persistence layer
├── pkg/
│   └── client/                  # (Planned) Client library
├── .github/
│   └── workflows/
│       └── go.yml               # CI: lint, test, vet, security jobs
├── config.yaml                  # Default configuration
├── config.example.yaml          # Example configuration
├── Dockerfile                   # Multi-stage production image
├── Taskfile.yaml                # Task runner configuration
├── .golangci.yml                # golangci-lint linter configuration
├── .gotestsum.yml               # gotestsum configuration
├── .pre-commit-config.yaml      # Pre-commit hooks configuration
├── CLAUDE.md                    # This file
├── go.mod                       # Go module definition
└── go.sum                       # Go dependencies lock file
```

## Architecture

### High-Level Structure

The codebase follows a clean architecture pattern with dependency injection:

1. **cmd/server/main.go** - Entry point that loads configuration and initializes components
2. **internal/config** - Configuration system with DDD value objects
3. **internal/core** - Core domain logic for vector storage
4. **internal/api** - HTTP layer using Gin framework
5. **internal/api/handlers** - HTTP handlers that depend on core services
6. **pkg/client** - (Empty) Planned client library location

### Dependency Flow

```
main.go
  → Loads Config (config)
  → Creates VectorIndex (core)
  → Passes VectorIndex to SetupRouter (api)
    → Passes VectorIndex to NewVectorHandler (handlers)
      → Handlers call VectorIndex methods
```

The architecture uses **constructor injection**: dependencies are passed explicitly when creating handlers and routers, rather than using global variables or service locators. Configuration is loaded once at startup and used to configure server and storage settings.

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

**internal/config/config.go**
- `Config` - Root configuration aggregate (DDD)
- `ServerConfig`, `StorageConfig` - Value objects
- Helper methods: `Address()`

**internal/config/loader.go** (NEW)
- `Loader` - Configuration loading service
- `Load()` - Loads from file + env vars with priority ordering
- Gracefully handles missing files (uses defaults)

**internal/config/validation.go** (NEW)
- Domain validation rules for all config sections
- Fail-fast validation at startup
- Clear error messages for invalid configuration

**internal/api/router.go**
- `SetupRouter(index)` - Configures Gin routes and injects dependencies into handlers
- Health check at `/health`
- API routes under `/api/v1`

**internal/api/handlers/vectors.go**
- `VectorHandler` - Holds reference to VectorIndex
- `Insert` endpoint - POST `/api/v1/vectors` accepts JSON with id, vector, and optional metadata
- `Search` endpoint - POST `/api/v1/query` accepts JSON with vector and k parameter

**cmd/server/main.go**
- Configuration loading at startup
- Auto-save ticker (configurable interval)
- Graceful shutdown with data persistence
- Loads existing data on startup

## Configuration

GoVec uses a YAML-based configuration system with environment variable overrides, following Domain-Driven Design (DDD) principles.

### Configuration File

Create `config.yaml` in the project root (see `config.example.yaml` for reference).

**Default configuration (config.yaml):**
```yaml
server:
  host: ""           # Bind to all interfaces
  port: 8000         # HTTP server port
  shutdown_timeout: 10s

storage:
  data_path: "./govec_data.bin"
  auto_save_enabled: true
  auto_save_interval: 60s
```

### Environment Variables

Override any config value using environment variables with the `GOVEC_` prefix:

**Server:**
- `GOVEC_SERVER_HOST` - Server host (default: "")
- `GOVEC_SERVER_PORT` - Server port (default: 8000)
- `GOVEC_SHUTDOWN_TIMEOUT` - Graceful shutdown timeout (default: 10s)

**Storage:**
- `GOVEC_STORAGE_PATH` - Data file path (default: "./govec_data.bin")
- `GOVEC_AUTO_SAVE_ENABLED` - Enable/disable auto-save (default: true)
- `GOVEC_AUTO_SAVE_INTERVAL` - Auto-save interval (default: 60s)

**Special:**
- `GOVEC_CONFIG_PATH` - Custom config file path (default: "config.yaml")

### Configuration Priority

Configuration is loaded with the following priority (highest to lowest):
1. Environment variables (highest)
2. Config file (config.yaml)
3. Built-in defaults (lowest)

### Examples

**Use custom port:**
```bash
GOVEC_SERVER_PORT=9000 task run
```

**Disable auto-save:**
```bash
GOVEC_AUTO_SAVE_ENABLED=false task run
```

**Use custom config file:**
```bash
GOVEC_CONFIG_PATH=./production.yaml task run
```

**Multiple overrides:**
```bash
GOVEC_SERVER_PORT=9000 GOVEC_AUTO_SAVE_INTERVAL=120s task run
```

### Validation

Configuration is validated at startup with fail-fast behavior:
- Server port: 1-65535
- Shutdown timeout: ≥ 0
- Storage path: cannot be empty
- Auto-save interval: ≥ 1s when enabled

Invalid configuration will cause the server to exit with a clear error message.

## Current State

**Implemented:**
- ✅ Basic vector insertion with metadata support
- ✅ In-memory vector storage using VectorIndex
- ✅ REST API with Gin framework
- ✅ **Persistence layer with GOB encoding**
  - SaveToFile/LoadFromFile with atomic writes
  - Configurable auto-save interval
  - Graceful shutdown saves data to disk
- ✅ **Vector search/query functionality**
  - Cosine similarity search
  - Top-K nearest neighbors
  - POST `/api/v1/query` endpoint
- ✅ **Configuration system (internal/config)**
  - YAML-based configuration with environment variable overrides
  - DDD architecture with value objects and validation
  - Fail-fast validation at startup
  - 98.1% test coverage
- ✅ Comprehensive test suite (255 tests, 98.1% coverage in config, 96.1% in core)
- ✅ Modern development tooling (Task runner, gotestsum)
- ✅ Health check endpoint
- ✅ Clean architecture with dependency injection
- ✅ **CI/CD pipeline (GitHub Actions)**
  - Separate jobs: lint, test (race + coverage), vet, security (govulncheck)
  - Codecov upload for coverage tracking
- ✅ **Docker support** - Multi-stage Dockerfile with non-root runtime user
- ✅ **Linting & security tooling**
  - golangci-lint with errcheck, staticcheck, gosec, gocritic, revive
  - govulncheck vulnerability scanning
  - pre-commit hooks for file hygiene and Go tooling

**In Progress / Planned:**
- ⏳ Advanced indexing algorithms (HNSW, IVF - internal/index planned)
- ⏳ Storage layer abstraction (internal/storage - planned)
- ⏳ Client library (pkg/client - planned)
- ⏳ Vector deletion and update endpoints
- ⏳ Metadata filtering in search

**Test Coverage:**
- internal/config: 98.1% (49 tests)
- internal/core: 96.1% (76 tests)
- internal/api: 100% (13 tests)
- internal/api/handlers: 100% (50 tests)
- internal/test/integration: 67 tests
- Total: 255 tests passing

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
# Run all static checks (lint, vet, fmt, vuln)
task check

# Run full test suite with race detector and coverage
task test:all

# Build to ensure no compilation errors
task build

# Clean up artifacts
task clean
```

If pre-commit is installed, hooks run automatically on `git commit` (gofmt, go vet, golangci-lint, govulncheck).

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
