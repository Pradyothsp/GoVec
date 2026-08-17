# AGENTS.md

This file provides guidance to agents (Claude, Gemini) and developers working on GoVec.

## Project Overview

GoVec is a high-performance vector database implemented in Go. It provides a REST API for storing and managing vector embeddings with associated metadata, supporting both float32 and int8 scalar quantization.

## Scope & Non-Goals

GoVec started life as a planning doc for "go-vector-lite," with an explicit positioning: a **compact, high-performance vector similarity search engine** — inspired by Pinecone, Weaviate, and Qdrant, but deliberately smaller, easier to run, and aimed at ML apps, prototyping, and local semantic search, not at matching production-scale platforms feature-for-feature. The original goal was portfolio-grade, production-quality Go code demonstrating good engineering at small scale, not a Pinecone competitor.

That framing sets real boundaries, not just aspirational ones:
- **Single-node, in-memory-first.** Sharding and replication were scoped as "future scaling" from the start — never core.
- **Small feature surface by design.** REST (+ optional gRPC), snapshot + WAL persistence, optional metadata filtering — not a query language, multi-tenancy, or a plugin ecosystem.
- **"Lite" is the point, not a placeholder.** When comparing against Chroma/Pinecone-class systems (see the `govec-bench` sibling repo), read the results against this framing — closing every gap to those systems was never the goal, and some gaps (e.g. `SELECT-NEIGHBORS-HEURISTIC`, dense ID-indexed storage) are deliberately deferred rather than missed.

## Getting Started

### Prerequisites
- **Go 1.26.0+**, **Task**, **gotestsum**, **golangci-lint**, **govulncheck**.
- Install tools: `task install-tools`.

### Quick Start
```bash
task deps           # Install dependencies
task test           # Run tests
task build          # Build the server
task run            # Start server on http://localhost:8000
```

## Development Tooling

- **Task**: Task runner (see `Taskfile.yaml`).
- **gotestsum**: Enhanced test output and watch mode.
- **golangci-lint**: Aggregated Go linter.
- **govulncheck**: Vulnerability scanner.
- **pre-commit**: Git hooks for hygiene and tooling.

## Development Commands

Run `task --list` to see all available commands.

| Command | Description |
|---------|-------------|
| `task test` | Run tests with pretty output |
| `task test:watch` | Auto re-run tests on file changes |
| `task test:all` | Run tests with race detector and coverage |
| `task check` | Run all static checks (lint, vet, fmt, vuln) |
| `task build` | Build the server binary |
| `task run` | Run the server |
| `task clean` | Remove build artifacts |

## Project Structure

```text
govec/
├── cmd/server/main.go       # Application entry point
├── internal/
│   ├── api/                 # HTTP layer (Gin)
│   │   └── handlers/        # API endpoint logic
│   ├── config/              # Configuration system (DDD)
│   ├── core/                # Core domain logic & VectorIndex
│   ├── hnsw/                # HNSW implementation
│   ├── index/               # Engine factory & WAL
│   └── test/                # Integration and E2E tests
├── build/swagger/           # Generated API documentation
└── config.yaml              # Default configuration
```

## Architecture

### High-Level Structure
GoVec uses a clean architecture with dependency injection and a factory pattern for engine creation.

**Dependency Flow:**
`main.go` → `config` → `index.NewEngine` → `api.SetupRouter` → `handlers.NewVectorHandler`

### Generic Engine & Quantization
The engine uses Go generics (`VectorIndex[T]`) to support different vector types:
- **float32**: High precision, 4 bytes/dim.
- **int8 (Scalar)**: 4x memory reduction, <0.001 precision loss.

**Key Components:**
- `internal/index/engine.go`: Common `Engine` interface.
- `internal/index/factory.go`: Creates `VectorIndex[T]` (brute-force) or `HNSWIndex[T]` (ANN) based on configuration.
- `internal/index/wal.go`: Write-ahead log for durability and recovery.
- `internal/core/persistence.go`: Atomic snapshot saving/loading using GOB.

## Configuration

GoVec uses YAML with environment variable overrides (prefix `GOVEC_`).

### Default `config.yaml`
```yaml
server:
  port: 8000
storage:
  data_path: "./govec_data.bin"
  wal_path: "./govec.wal"
  auto_save_interval: 60s
engine:
  quantization: "none"      # "none" (float32) or "scalar" (int8)
  distance_metric: "cosine" # "cosine" or "euclidean"
```

### Key Environment Variables
- `GOVEC_SERVER_PORT`
- `GOVEC_QUANTIZATION`
- `GOVEC_AUTO_SAVE_ENABLED`

## Current State & Roadmap

### Implemented
- ✅ REST API (Insert, Search, Health)
- ✅ Vector deletion (`DELETE /api/v1/vectors/:id`) and upsert-on-insert
- ✅ Reset/clear all vectors (`POST /api/v1/admin/reset`) -- see `docs/architecture/ID_MAPPING.md` amendment
- ✅ Generic Vector Engine (float32 & int8)
- ✅ Approximate nearest neighbor search (HNSW, alongside brute-force linear scan) -- see `internal/hnsw`, `internal/index/factory.go`, and `docs/architecture/HNSW_EF_CONSTRUCTION.md` for the `hnsw_ef_construction`/`hnsw_ef_search` split
- ✅ Cosine and Euclidean distance metrics (`engine.distance_metric`), both index types -- see `docs/architecture/DISTANCE_METRICS.md`
- ✅ Write-Ahead Log (WAL) & Crash Recovery
- ✅ Atomic Persistence (GOB encoding)
- ✅ CI/CD with GitHub Actions (lint, test, vuln)
- ✅ Docker support (multi-stage)
- ✅ HNSW heuristic neighbor selection (`SELECT-NEIGHBORS-HEURISTIC`) -- closed the recall@1 gap (75.0%→96.0%, matching Chroma exactly); also fixed two other real bugs found along the way (base layer needs 2×M capacity, not M; query-time search was exploring at breadth `k` instead of `max(k, ef_search)`). Real cost: insert and high-k query latency regressed, and RAM usage roughly doubled -- not yet recovered, see `docs/architecture/HNSW_NEIGHBOR_SELECTION.md`

### Planned
- ⏳ Recover insert/query/memory cost from the heuristic neighbor selection work above -- `EfConstruction`/`EfSearch` are now the dominant cost drivers (not `M`), unexplored so far -- see `docs/architecture/HNSW_NEIGHBOR_SELECTION.md`
- ⏳ Product & Binary quantization
- ⏳ Metadata filtering in search
- ❌ HNSW dense ID-indexed storage -- evaluated with real numbers (~11MB combined at 100k vectors, using the per-entry map-overhead constant measured in the neighbor-slice fix) and declined; not worth the implementation cost -- see `docs/architecture/HNSW_MEMORY_LAYOUT.md`
- ⏳ Concurrent-safe HNSW graph insertion (parallelize `BatchInsert` across goroutines) -- measured ceiling vs Chroma is only ~1.04x (see `govec-bench/STATUS.md` §19), low priority -- see `docs/architecture/HNSW_CONCURRENT_INSERT.md`

## Testing & Workflow

### Testing Standards
- **Coverage**: Aim for 100% in `internal/core` and `internal/config`.
- **Validation**: Use `task test:all` before any PR.
- **Integration**: New features must include integration tests in `internal/test/integration`.

### Before Committing
1. Run `task check` to ensure linting and formatting pass.
2. Run `task test` to verify logic.
3. Build the project using `task build`.
