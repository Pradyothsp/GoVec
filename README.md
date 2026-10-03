# GoVec

A compact, fast vector search engine written in Go: HNSW and brute-force indexes, int8
quantization, hybrid dense + sparse search, and crash-safe persistence, served over REST and
gRPC from a single static binary.

[![CI](https://github.com/Pradyothsp/govec/actions/workflows/go.yml/badge.svg)](https://github.com/Pradyothsp/govec/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Pradyothsp/govec.svg)](https://pkg.go.dev/github.com/Pradyothsp/govec)
[![Go Report Card](https://goreportcard.com/badge/github.com/Pradyothsp/govec)](https://goreportcard.com/report/github.com/Pradyothsp/govec)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

GoVec is inspired by Qdrant, Weaviate and Pinecone, but deliberately smaller. It targets
prototyping, local semantic search, and ML apps that need vector search on a single node,
without running a distributed platform. It starts in under 10 ms, and at the same recall as
Chroma and Qdrant it answers queries faster than either (see [Performance](#performance)).

## Ecosystem

| Repository | Description |
|---|---|
| **govec** (this repo) | The server: REST + gRPC vector search engine |
| [govec-python](https://github.com/Pradyothsp/govec-python) | Python SDK over REST or gRPC (`pip install govec`) |
| [govec-bench](https://github.com/Pradyothsp/govec-bench) | Benchmark harness comparing GoVec against Chroma and Qdrant |

## Features

- **Two index types.** HNSW for approximate nearest-neighbour search, using the paper's
  heuristic neighbour selection and separate `ef_construction` / `ef_search` knobs. Exact
  brute-force scan for small collections.
- **Scalar quantization.** Optional int8 storage stores vectors in a quarter of the space;
  measured at -61% disk and -26% RAM at 100k vectors. Vectors are range-checked before encoding, so
  quantization can't silently corrupt them.
- **Hybrid search.** Combine a dense embedding with a sparse keyword vector in one query.
- **Metadata filtering.** Exact-match filters on arbitrary JSON metadata, with an optional
  inverted index that pushes selective filters into the graph traversal.
- **Durability.** A write-ahead log plus atomic snapshots. The server replays the WAL on
  startup and refuses to start on a log it can't read, rather than silently dropping writes.
- **Cosine and Euclidean** distance on both index types.
- **REST and gRPC** with identical semantics, including streaming batch insert over gRPC. A
  parity test keeps the two from drifting.
- **Operable.** One binary, YAML config with an environment override for every field, optional
  bearer-token auth, structured logging with request IDs, and pprof on a separate port.

## Quick start

### Run the server

With Docker:

```bash
docker run -p 9697:9697 -v govec-data:/data ghcr.io/pradyothsp/govec:latest
```

Or from source (Go 1.26+):

```bash
git clone https://github.com/Pradyothsp/govec.git
cd govec
go run ./cmd/server
```

The server listens on `http://localhost:9697`.

### Insert and search

```bash
# Insert a vector with metadata
curl -X POST http://localhost:9697/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "doc-1", "vector": [0.1, 0.9, 0.2], "metadata": {"lang": "en", "year": 2024}}'

# Insert several at once
curl -X POST http://localhost:9697/api/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{"vectors": [
        {"id": "doc-2", "vector": [0.8, 0.1, 0.3], "metadata": {"lang": "de"}},
        {"id": "doc-3", "vector": [0.2, 0.8, 0.1], "metadata": {"lang": "en"}}
      ]}'

# Find the 2 nearest neighbours
curl -X POST http://localhost:9697/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.85, 0.15], "k": 2}'
```

```json
{
  "success": true,
  "data": [
    {"id": "doc-1", "score": 0.999, "meta": {"lang": "en", "year": 2024}},
    {"id": "doc-3", "score": 0.991, "meta": {"lang": "en"}}
  ]
}
```

Add a metadata filter (every key must match exactly):

```bash
curl -X POST http://localhost:9697/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.85, 0.15], "k": 5, "filter": {"lang": "de"}}'
```

For hybrid search, add a `sparse_vector` (`{"indices": [...], "values": [...]}`) to both the
insert and the query. Scores are blended as `0.7 × dense + 0.3 × sparse`.

### From Python

```bash
pip install govec   # or: uv add govec
```

See [govec-python](https://github.com/Pradyothsp/govec-python) for the client API. A minimal
Go client over plain `net/http` is in [`examples/client`](examples/client/main.go).

## API

Every REST response uses the same envelope:
`{"success": bool, "data": ..., "error": "..."}`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/vectors` | Insert or update (upsert) one vector |
| `POST` | `/api/v1/vectors/batch` | Insert many vectors; reports per-item errors |
| `GET` | `/api/v1/vectors/:id` | Fetch a vector and its metadata |
| `DELETE` | `/api/v1/vectors/:id` | Delete a vector |
| `POST` | `/api/v1/vectors/search` | k-NN search with optional `filter` and `sparse_vector` |
| `GET` | `/api/v1/stats` | Vector count |
| `GET` | `/api/v1/info` | Server version and engine configuration (index type, quantization, metric, dimensions) |
| `POST` | `/api/v1/admin/flush` | Write a snapshot now |
| `POST` | `/api/v1/admin/reset` | Delete all vectors |

- **Swagger UI:** interactive docs at `http://localhost:9697/swagger/index.html`.
- **gRPC:** set `grpc.enabled: true` to serve on port 9698. The service definition is in
  [`proto/govec/v1`](proto/govec/v1). It offers the same operations, plus client-streaming
  `BatchInsert`.
- **Auth:** set `server.api_key`, and every `/api/v1/*` request must send
  `Authorization: Bearer <key>`. `/health` stays open for probes.
- **Errors:** an invalid vector (empty, or the wrong dimension for the index) returns HTTP 400 /
  gRPC `InvalidArgument`.

## Configuration

GoVec reads `config.yaml` from the working directory, or the file named by `GOVEC_CONFIG_PATH`.
In the Docker image, mount your own file at `/app/config.yaml`. All data lives in the `/data`
volume, and gRPC needs `-p 9698:9698` once enabled.
Every field also has an environment variable, and the environment wins. The variable name is
`GOVEC_` plus the YAML key (`storage.wal_path` → `GOVEC_WAL_PATH`). The section is included
only where the key alone would be ambiguous (`GOVEC_SERVER_PORT` vs `GOVEC_GRPC_PORT`).

The settings you're most likely to change:

| Setting | Env var | Default | |
|---|---|---|---|
| `engine.index_type` | `GOVEC_INDEX_TYPE` | `brute` | `hnsw` for collections beyond ~10k vectors |
| `engine.quantization` | `GOVEC_QUANTIZATION` | `none` | `scalar` for int8 storage |
| `engine.distance_metric` | `GOVEC_DISTANCE_METRIC` | `cosine` | or `euclidean` |
| `engine.enable_metadata_index` | `GOVEC_ENABLE_METADATA_INDEX` | `false` | faster filtered search, more RAM |
| `storage.data_path` | `GOVEC_DATA_PATH` | `./govec_data.bin` | snapshot file |
| `storage.wal_path` | `GOVEC_WAL_PATH` | `./govec.wal` | write-ahead log |
| `server.api_key` | `GOVEC_API_KEY` | empty (auth off) | bearer token for `/api/v1/*` |
| `grpc.enabled` | `GOVEC_GRPC_ENABLED` | `false` | serve gRPC on `grpc.port` (9698) |

[`config.example.yaml`](config.example.yaml) documents every option, including the HNSW tuning
parameters (`hnsw_m`, `hnsw_ef_search`, `hnsw_ef_construction`).

**Memory tip:** set `GOMEMLIMIT` to 2–3x your expected live data, kept under the container's
memory limit. Under sustained load it cuts resident memory substantially (-46% in benchmarks)
with no latency cost. It isn't set by default because the right value depends on dataset size.

## How it works

```text
            REST (Gin)        gRPC
                 \             /
                  Engine interface
                 /              \
        VectorIndex[T]      HNSWIndex[T]       T = float32 | int8
          (brute)              (ANN)
                 \              /
          WAL  +  atomic GOB snapshots  (+ optional mmap vector store)
```

- **One engine, two transports.** REST and gRPC are thin adapters over a single `Engine`
  interface. Both share input validation and error classification, so they can't disagree.
- **Generic storage.** Both indexes are written once over `T = float32 | int8`. Quantization is
  a config switch, not a separate code path. Under cosine, vectors are unit-normalized before
  int8 encoding. Under Euclidean, out-of-range vectors are rejected, because normalizing them
  would change the distance.
- **HNSW.** A fork of [coder/hnsw](https://github.com/coder/hnsw), made generic and extended
  with `SELECT-NEIGHBORS-HEURISTIC`, a `2×M` base layer, and greedy descent through upper
  layers on insert. Batch inserts run each round's graph searches in parallel, then apply the
  mutations serially, so no read ever overlaps a write.
- **Durability.** Every write is appended to the WAL (batch inserts share a single fsync).
  Snapshots are written to a temp file and atomically renamed. On startup, GoVec loads the
  snapshot and replays the WAL past it. It refuses to start if either can't be read in full.

## Performance

Measured with [govec-bench](https://github.com/Pradyothsp/govec-bench) (October 2026), all four
systems in the same session, each in Docker capped at 2 CPU / 2 GB. The dataset is SIFT10K
(128-dim), with HNSW `M=16, ef_construction=200, ef_search=50`. Memory was measured at 100k
vectors. Latencies are medians across 3–4 repeated runs.

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | **1.17 ms** | 1.27 ms | 3.00 ms | 2.46 ms |
| Single insert (mean) | 2.25 ms | **2.14 ms** | 8.04 ms | 2.26 ms |
| Batch insert, per vector | 0.38 ms | 0.28 ms | **0.24 ms** | 0.96 ms |
| Recall@10 | **99.2%** | 93.9% | **99.2%** | **99.2%** |
| Cold start (p50) | 7.6 ms | **7.5 ms** | 60.4 ms | 12.9 ms |
| RAM, 100k vectors | 232 MB | 173 MB | **107 MB** | 111 MB |
| Disk, 100k vectors | 118 MB | **47 MB** | 87 MB | 231 MB |

At identical recall, GoVec had the fastest queries in every run: about 2.5x faster than Chroma
and 2x faster than Qdrant at the median. It inserts single vectors about 3.5x faster than
Chroma, on par with Qdrant, and starts about 8x faster than Chroma. Chroma leads on
batch-insert throughput and RAM. GoVec's HNSW stores neighbour lists as pointers rather than
flat ID arrays, which costs memory. Full methodology and raw results are in the govec-bench
repo.

## Status and limitations

GoVec is **pre-1.0 (v0.x)**. The REST and gRPC APIs are stable in practice but may change
before 1.0. It is built for prototyping, local semantic search, and embedding in ML
applications on a single node.

Limitations to know about:

- **Single node, in-memory first.** No sharding or replication. The dataset must fit in RAM
  unless you enable the mmap vector store.
- **Filtered HNSW search can return fewer than `k` results** when the filter is selective,
  especially with `enable_metadata_index: false`. The brute-force index is always exact.
- **Hybrid search scans all sparse vectors.** Sparse scoring is O(n) per query, so it suits
  small and medium collections.
- **Filters are exact-match only.** Keys are ANDed together, with no range or `OR` operators.

Planned work and deliberate non-goals are in [docs/roadmap.md](docs/roadmap.md).

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, the test and lint workflow, and project conventions.

## License

GoVec is licensed under the [Apache License 2.0](LICENSE).

The HNSW implementation in `internal/hnsw` is derived from
[coder/hnsw](https://github.com/coder/hnsw) (CC0-1.0).
