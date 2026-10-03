# AGENTS.md

Guidance for AI agents (Claude, Gemini) and contributors working in this repo. This file covers
how the code is organized, the rules it follows, and the traps to avoid. Feature status and
plans are in [docs/roadmap.md](docs/roadmap.md). User-facing API docs are in `README.md`.

## What GoVec is (and isn't)

A compact vector similarity search engine in Go: a REST API plus optional gRPC, float32 or int8
storage, brute-force or HNSW indexes, and WAL plus snapshot persistence. It is meant to be small,
production-quality code, not a Pinecone competitor. Use these limits when choosing a design:

- **Single-node, in-memory-first.** Don't add sharding, replication or multi-tenancy hooks.
- **Small surface by design.** REST/gRPC, persistence, metadata filtering. No query language or
  plugin system.
- **"Lite" is the point.** When `govec-bench` (sibling repo) shows a gap to Chroma/Pinecone,
  check `docs/roadmap.md` first. Some gaps are deliberately deferred or declined.

## Commands

| Command | Use |
|---|---|
| `task install-tools` | One-time setup: gotestsum, golangci-lint, govulncheck, swag, buf |
| `task test` | Fast test run |
| `task test:pkg PKG=internal/index` | Tests for one package |
| `task test:all` | Race detector and coverage. Run before calling work done |
| `task check` | vet, fmt, golangci-lint, govulncheck, buf lint |
| `task fmt:fix` | Auto-fix formatting |
| `task gen` | Regenerate Swagger (`build/swagger/`) and gRPC stubs (`gen/`) |
| `task build` / `task run` | Build the binary / run the server on :9697 |

`task run` writes `govec.wal` and `govec_data.bin` into the repo root, and `task clean` removes
them. Don't commit them.

## Code map

```text
cmd/server/main.go          wiring: config → index.NewEngine → REST router + optional gRPC server
internal/
  config/                   Config structs, YAML loader, GOVEC_* env overrides (env.go), validation
  core/                     pure domain: models, similarity math, quantization, filters, metadata index
  index/                    Engine interface + both engines, WAL, snapshots, input guards
    engine.go               the Engine interface every transport talks to
    factory.go              picks VectorIndex[T] (brute force) or HNSWIndex[T] from config
    index.go / hnsw_index.go  the two engines
    wal.go / persistence.go   durability: write-ahead log, atomic GOB snapshots
    dimension_guard.go      input sentinels + IsInvalidVectorError
    encode_guard.go         quantization preconditions (prepareForEncode)
  hnsw/                     vendored fork of an upstream HNSW library, made generic over T
  storage/                  mmap-backed store (unix, plus a stub for other platforms)
  api/                      REST layer (Gin): router, handlers/, middleware/, response/
  grpcserver/               gRPC layer; convert/ maps protobuf ↔ core types
  requestid/                correlation-ID propagation shared by both transports
  test/integration/         cross-package and end-to-end tests; testutil/ has shared helpers
proto/                      protobuf source of truth for the gRPC API
gen/                        generated gRPC code (do not edit)
build/swagger/              generated OpenAPI docs (do not edit)
docs/                       design notes, known issues, roadmap
```

Dependencies point inward: transports (`api`, `grpcserver`) → `index` → `core`. `core` imports
no other package from this repo, and `index` never imports a transport. Keep it that way.

## Conventions

**Transports are thin and must agree.** REST and gRPC are two wire formats over one `Engine`. An
operation must return the same result and status string over either one.
`internal/test/integration/transport_parity_test.go` enforces this. If you add or change an
operation, change both transports and extend the parity test.

**Classify errors where they originate.** The `index` layer defines sentinel errors and
classifiers such as `IsInvalidVectorError`. Transports call the classifier to choose HTTP 400 or
gRPC `InvalidArgument` over a 500. Don't make a transport `errors.Is` against index internals.
Add the sentinel to the classifier instead.

**REST responses use the envelope.** Always write through `response.OK` / `response.Fail`
(`{success, data, error}`). Never call `c.JSON` directly in a handler. Handlers carry swag
annotations (`@Summary`, `@Router`, …). Run `task gen:swagger` after changing them.

**Generated code is regenerated, never hand-edited.** Change `proto/` and run `task gen:proto`.
Change handler annotations and run `task gen:swagger`.

**Config fields come in sets.** A new field needs a default in `config.go`, a check in
`validation.go`, and a `GOVEC_*` override in `env.go`. Name the variable `GOVEC_` plus the YAML
key, adding the section only when the key alone is ambiguous (`GOVEC_SERVER_PORT` vs
`GOVEC_GRPC_PORT`). `TestLoadFromEnv_EveryFieldHasAnOverride` walks the struct by reflection and
fails if any field has no override. Enum-like values use the typed constants
(`config.QuantizationScalar`, `config.DistanceMetricCosine`) and never string literals.

**Logging uses zerolog** (`github.com/rs/zerolog/log`). Don't use `fmt.Print*` or the stdlib
`log` package.

**Comments explain why.** Existing code documents the reason for non-obvious decisions, often
naming the bug they prevent (see `encode_guard.go`, `dimension_guard.go`). Match that. Don't
narrate what the code does line by line.

## Testing

- testify everywhere: `require` for preconditions that make the rest meaningless, `assert` for
  the checks themselves. Integration suites use `testify/suite`.
- Use `t.TempDir()` for any WAL or snapshot path. Never write into the repo.
- Unit tests sit next to the code. Behaviour that crosses packages (persistence + recovery,
  transports + engine) goes in `internal/test/integration/`.
- Coverage target: ~100% for `internal/core` and `internal/config`.
- Known, unfixed bugs get a regression test committed with `t.Skip("<link to doc>")`, so the
  fix has a ready-made test to un-skip (example: `hnsw_filtered_search_test.go`).
- Before adding a test, check that the behaviour isn't already covered. The suite already has
  real redundancy, so don't add to it.

## Traps

- **Don't delete `InvertedIndex` as dead code.** Its postings are maintained but not read yet.
  It is the foundation for planned hybrid-search work (`docs/roadmap.md`). Snapshots don't store
  it; every load path rebuilds it.
- **HNSW graph mutation is single-writer.** Batch insert runs each round's graph *searches*
  concurrently, waits at a `sync.WaitGroup` barrier, then applies the *mutations* serially, so
  reads never overlap writes. Don't add concurrent mutation without reading the roadmap entry on
  per-node locking. Its deadlock failure mode is invisible to `-race`.
- **Scalar quantization assumes `[-1, 1]`.** `prepareForEncode` normalizes under cosine and
  rejects out-of-range input under Euclidean. Route any new insert path through it, or vectors
  get silently clamped and recall collapses.
- **Filtered HNSW search can return fewer than `k`.** This is known and parked. See
  `docs/hnsw-filtered-search-shortfall.md` before you touch filtering or "fix" a short result
  in a test.
- **`internal/hnsw` is a fork.** Keep changes there minimal and generic. GoVec-specific logic
  belongs in `internal/index`.

## Before you finish

1. `task fmt:fix && task check`: lint, vet, vuln and proto lint must be clean.
2. `task test:all`: tests pass under `-race`.
3. `task build`.
4. Commit messages follow the `commit-message-conventions` skill (`.agents/skills/`).
