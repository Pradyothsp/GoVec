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
- `internal/index/persistence.go`: Atomic snapshot saving/loading using GOB.

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

### Environment Variables
Every field in `Config` has a `GOVEC_*` override — see `internal/config/loader.go` for the
full list and README.md for it grouped by section. The name is `GOVEC_` plus the YAML key
(`storage.wal_path` → `GOVEC_WAL_PATH`), with the section included only where the key alone
would be ambiguous (`GOVEC_SERVER_PORT` vs `GOVEC_GRPC_PORT`).

**Adding a config field means adding its override too.** `TestLoadFromEnv_EveryFieldHasAnOverride`
fails if you don't. The readers live in `internal/config/env.go`, so it is one line.

## Current State & Roadmap

### Implemented
- ✅ REST API (insert, search, delete, health) and optional gRPC server
- ✅ Vector deletion (`DELETE /api/v1/vectors/:id`), upsert-on-insert, and reset (`POST /api/v1/admin/reset`)
- ✅ Generic vector engine (`VectorIndex[T]`) over float32 and int8
- ✅ Scalar (int8) quantization -- ~4x memory reduction
- ✅ Approximate nearest neighbor search (HNSW) alongside brute-force linear scan
- ✅ HNSW heuristic neighbor selection (`SELECT-NEIGHBORS-HEURISTIC`), with a distinct
  `hnsw_ef_construction` separate from `hnsw_ef_search`
- ✅ Concurrent-safe HNSW batch insert via round-based barrier parallelism
  (`hnsw_batch_parallelism` / `hnsw_batch_parallel_threshold`): each round runs every item's
  graph *search* concurrently, hits a `sync.WaitGroup` barrier, then replays every item's
  graph *mutation* serially -- so no read ever overlaps a write, by construction
- ✅ Cosine and Euclidean distance metrics, both index types
- ✅ Metadata filtering in search (`engine.enable_metadata_index`) -- in-memory inverted index;
  selective filters are pushed into graph traversal, non-selective ones post-filtered with over-fetch
- ✅ Write-ahead log (WAL) and crash recovery
- ✅ Atomic snapshot persistence (GOB encoding)
- ✅ CI/CD with GitHub Actions (lint, test, vuln), multi-stage Dockerfile

### Notes for deployers
- `GOMEMLIMIT` (Go 1.19+, no code change) substantially cuts measured RAM under sustained load
  at no latency cost. Deliberately not defaulted -- the right value is proportional to your
  dataset size, not a universal constant.

### Planned
- ⏳ Product and Binary quantization. Not similarly-sized asks: PQ needs training/codebook
  infrastructure that doesn't exist anywhere in this codebase; BQ is stateless, the same shape
  as the scalar quantization that already ships.
- ⏳ Full per-node-locking concurrent HNSW graph mutation (beyond the batch-scoped parallel
  insert that ships today). Bigger payoff ceiling, bigger risk: hnswlib needed a dedicated
  bug-fix PR for this exact design, and the failure mode (deadlock) isn't caught by `-race`,
  only by hitting the wrong interleaving under load.
- ⏳ Inverted-index-backed sparse scoring for hybrid search. Both engines keep a sparse
  inverted index (`InvertedIndex`, posting lists maintained on insert/delete and rebuilt on
  snapshot load), but **nothing reads the postings yet**: `computeSparseScores` scans every
  node's sparse vector instead. The index is deliberately kept as the foundation for this
  work -- don't delete it as dead code. Two problems it should solve:
  - HNSW hybrid search computes sparse scores for all `n` documents, then uses only the
    graph's ~`k` candidates, so every hybrid query is O(n) despite the O(log n) graph.
  - HNSW hybrid search only reranks the dense candidates. A strong keyword match that isn't
    close in meaning is never retrieved; full hybrid systems fetch candidates from both the
    graph and the inverted index and merge them.
- ❌ HNSW dense ID-indexed storage -- evaluated with real numbers (~11MB combined at 100k
  vectors) and declined; not worth the implementation cost.

## Testing & Workflow

### Testing Standards
- **Coverage**: Aim for 100% in `internal/core` and `internal/config`.
- **Validation**: Use `task test:all` before any PR.
- **Integration**: New features must include integration tests in `internal/test/integration`.

### Before Committing
1. Run `task check` to ensure linting and formatting pass.
2. Run `task test` to verify logic.
3. Build the project using `task build`.
