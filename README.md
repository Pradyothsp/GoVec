# GoVec

A compact, fast vector search engine written in Go: HNSW and brute-force indexes, int8
quantization, hybrid dense + sparse search, and crash-safe persistence, served over REST and
gRPC from a single static binary.

[![CI](https://github.com/Pradyothsp/govec/actions/workflows/go.yml/badge.svg)](https://github.com/Pradyothsp/govec/actions/workflows/go.yml)
[![Release](https://img.shields.io/github/v/release/Pradyothsp/govec?sort=semver)](https://github.com/Pradyothsp/govec/releases)
[![Go version](https://img.shields.io/github/go-mod/go-version/Pradyothsp/govec)](go.mod)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

GoVec is inspired by Qdrant, Weaviate and Pinecone, but deliberately smaller. It targets
prototyping, local semantic search, and ML apps that need vector search on a single node,
without running a distributed platform. It finds the same neighbours as Chroma and Qdrant at the
same settings and has the fastest inserts of the three; its queries are slower at 100k vectors,
and in memory it uses more RAM (see [Performance](#performance)).

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
  measured at -61% disk and -31% RAM at 100k vectors. Vectors are range-checked before encoding, so
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
insert and the query. Scores are blended as `0.7 × dense + 0.3 × sparse`. Hybrid scoring needs
`engine.enable_hybrid_search: true`, which the Docker image's config sets; without it a sparse
vector is stored but doesn't affect scores.

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

**Recall tip:** both `ef` defaults are 100, as in Chroma and Qdrant. On large datasets, if recall
matters more than insert speed, raise `hnsw_ef_construction` (to 200, say: at 100k text
embeddings that looked worth about half a point to a point of recall@10, at slower inserts) or
`hnsw_ef_search`, which costs query latency instead and can be changed without rebuilding.

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

Measured with [govec-bench](https://github.com/Pradyothsp/govec-bench) on 100k 1536-dimensional
OpenAI text embeddings (DBpedia), one database at a time in Docker with 2 CPUs and 2 GB each.
Every database builds HNSW with `M=16, ef_construction=100` and searches with ef 100, cosine;
every number for a database comes from one index, and recall is graded against exact cosine
neighbours.

Medians of three runs (October 7, 2026, GoVec 0.2.1) on a MacBook with Docker Desktop.

| | GoVec | GoVec (int8) | GoVec (mmap)† | Chroma | Qdrant |
|---|---:|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 6.23 ms | 6.45 ms | 7.93 ms | 5.58 ms | **4.61 ms** |
| Query latency over gRPC, k=10 (mean)‡ | 4.10 ms | | | | |
| Single insert (mean) | **5.06 ms** | 5.16 ms | 5.24 ms | 12.42 ms | 6.04 ms |
| Batch insert, per vector | **1.96 ms** | 2.04 ms | 2.12 ms | 2.21 ms | 7.62 ms |
| Recall@10 | 97.7% | 88.4% | 98.1% | 98.1% | **98.6%** |
| Recall@50 | 96.2% | 89.1% | 96.4% | 96.4% | **97.8%** |
| Restart, empty (median) | **9.0 ms** | 10.3 ms | 10.5 ms | 67.2 ms | 12.0 ms |
| Restart with the data loaded (median) | 930 ms | 553 ms | 474 ms | 209 ms | **81 ms** |
| RAM | 1,329 MB | 466 MB | 817 MB | 708 MB | 271 MB* |
| Disk (allocated) | 632 MB | **171 MB** | 632 MB | 659 MB | 774 MB |

\* Qdrant keeps its vectors on disk by default and lets the OS cache only part of them, so its
RAM isn't comparable with GoVec's and Chroma's, which hold every vector in memory.

† GoVec with `storage.enable_mmap: true` (and `engine.dimensions` set): vectors in memory-mapped
files instead of the Go heap; RAM is what's in use, including those files' cached pages. Its disk
is the same as in memory: each vector is stored once, in the mapped files instead of the snapshot.

‡ The same GoVec index, queried over gRPC instead of REST. Every other figure, Chroma's and
Qdrant's included, is over HTTP; Qdrant also has a gRPC API, not measured here. Compare this row
with GoVec's REST figure, not with the other databases.

All three find the same neighbours at the same ef, within the point or so that recall moves
between index builds. GoVec has the fastest single and batch inserts. Over REST its queries are
the slowest at this size, 1.1x Chroma and 1.4x Qdrant: at 10k vectors it was level with Qdrant,
and its latency grows faster with the graph, since distance computation doesn't use SIMD yet and
neighbour lists are pointers rather than flat ID arrays. Over gRPC the same queries take a third
less time. It restarts empty in about 9 ms but takes about 1 s to answer with 100k vectors loaded,
because it reads its whole snapshot back first. In memory it holds about 1.9x Chroma's RAM.
Memory-mapped (the mmap column), memory in use falls 38%, restarts with data take half as long
and recall is the same, but queries are about 27% slower. int8 recall on text embeddings is low
for now. SIFT
results, the 10k numbers, an ef sweep, the method and how to reproduce are in the govec-bench
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
