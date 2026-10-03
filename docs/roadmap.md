# Roadmap & Status

What ships today, what's planned, and what was deliberately declined. Read it against the
scope in `AGENTS.md`: GoVec is a single-node, "lite" engine, so it was never meant to close
every gap to Pinecone/Chroma-class systems.

## Implemented

- REST API (insert, search, delete, health) and optional gRPC server
- Vector deletion (`DELETE /api/v1/vectors/:id`), upsert-on-insert, and reset (`POST /api/v1/admin/reset`)
- Generic vector engine (`VectorIndex[T]`) over float32 and int8
- Scalar (int8) quantization, about 4x less memory
- Approximate nearest neighbor search (HNSW) alongside brute-force linear scan
- HNSW heuristic neighbor selection (`SELECT-NEIGHBORS-HEURISTIC`), with `hnsw_ef_construction`
  kept separate from `hnsw_ef_search`
- Concurrent-safe HNSW batch insert via round-based barrier parallelism
  (`hnsw_batch_parallelism` / `hnsw_batch_parallel_threshold`)
- Cosine and Euclidean distance metrics, both index types
- Metadata filtering in search (`engine.enable_metadata_index`): an in-memory inverted index.
  Selective filters are pushed into graph traversal and non-selective ones are post-filtered.
  **Known gap:** filtered HNSW search can return fewer than `k` results; see
  [hnsw-filtered-search-shortfall.md](hnsw-filtered-search-shortfall.md)
- Write-ahead log (WAL) and crash recovery
- Atomic snapshot persistence (GOB encoding)
- CI with GitHub Actions (lint, test, vuln) and a multi-stage Dockerfile

## Planned

- **Product and Binary quantization.** These are different sizes of work. PQ needs
  training/codebook infrastructure that doesn't exist anywhere in the codebase. BQ is
  stateless, the same shape as the scalar quantization that already ships.
- **Full per-node-locking concurrent HNSW graph mutation** (beyond today's batch-scoped
  parallel insert). The payoff ceiling is higher and so is the risk: hnswlib needed a dedicated
  bug-fix PR for this exact design, and its failure mode (deadlock) isn't caught by `-race`,
  only by hitting the wrong interleaving under load.
- **Inverted-index-backed sparse scoring for hybrid search.** Both engines maintain a sparse
  `InvertedIndex`, but nothing reads the postings yet: `computeSparseScores` scans every
  node's sparse vector. Two problems this should solve:
  - HNSW hybrid search computes sparse scores for all `n` documents and then uses only the
    graph's ~`k` candidates, so every hybrid query is O(n) despite the O(log n) graph.
  - HNSW hybrid search only reranks dense candidates. A strong keyword match that isn't close
    in meaning is never retrieved. Full hybrid systems fetch candidates from both the graph and
    the inverted index and merge them.

## Declined

- **HNSW dense ID-indexed storage.** Evaluated with real numbers (~11MB combined at 100k
  vectors) and declined as not worth the implementation cost.

## Deployment notes

- `GOMEMLIMIT` (Go 1.19+, no code change) substantially cuts measured RAM under sustained load
  at no latency cost. It is deliberately not defaulted because the right value depends on
  dataset size, not on a universal constant.
