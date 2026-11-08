# How HNSW deletes work, and why not mark-and-skip

**Status:** fixed by scanning the layer on delete (option A below); mark-and-skip (option C) parked
**Date:** 2026-10-22
**Affects:** `index_type: hnsw`. The brute-force engine has no graph.
**Regression tests:** `TestGraph_Delete_LeavesNoLinkToTheDeletedNode` (`internal/hnsw`),
`TestSaveAndLoad_AfterDeletes` (`internal/index`)

## Summary

Deleting from an HNSW index used to leave links to the deleted node behind. The next snapshot
saved them, and loading it failed with `failed to resolve neighbor`: after a batch insert and a
few deletes, the server could not restart. Through the engine, 300 batch inserts, 30 deletes and a
save produced an unloadable snapshot in 20 out of 20 runs.

Deletes now find and remove every link to the deleted node. That costs a scan of each layer the
node was in: about 5–8 ms per delete at 100k vectors.

## Mechanism

HNSW links are one-way. Node X can list Y as a neighbour without Y listing X: neighbour lists are
capped, and each node keeps the best ones for itself. The base layer allows `2*M` links per node
and the others `M`, and the parallel batch insert plans links against a snapshot of the graph, so
one-way links are common.

The old delete removed links only from the nodes the deleted node itself listed
(`layerNode.isolate`). Any node that listed the deleted one, without being listed back, kept a
pointer to it:

- **Snapshots:** the export wrote that neighbour's key; the import could not resolve it and failed.
- **Search:** a traversal could still walk through the deleted node. `HNSWIndex.Search` drops keys
  that are no longer stored, so it was never returned, but it could take a result slot.
- **Repair:** `replenish` gathers candidates from neighbours' neighbours, so a dangling link could
  hand the deleted node back to a live one.

Nothing records who links to a node, so finding those links means looking at every node.

## The options

| | How | Cost | Catch |
|---|---|---|---|
| **A. Scan on delete** (chosen) | Scan each layer the node was in, drop every link to it, repair the nodes that lost one | One pass over the layer per delete: ~nodes × `M` comparisons | Delete time grows with the index; local repair can weaken the graph over many deletes |
| B. Incoming-link lists | Each node also records who links to it | A second list per node; every link change updates both sides | More memory and more code where links can drift; makes the planned flat layout harder |
| C. Mark and skip | Flag the node deleted, keep it as a bridge, never return it, reuse its slot later | A flag flip | Memory and disk aren't freed; search keeps visiting dead nodes; needs a compaction job |

`layer.detach` (`internal/hnsw/delete.go`) does A. It repairs the affected nodes in key order, not
map order, so the same deletes always produce the same graph.

## Why mature systems mark instead

hnswlib (behind Chroma) marks deleted nodes and can reuse their slots for new inserts. Lucene,
Qdrant, Weaviate and Milvus also delete by marking, and rebuild or clean up the graph later in a
separate step (segment merges, background optimisation, tombstone clean-up). The reasons:

1. **True deletion is expensive at their scale.** Finding who links to a node and choosing
   replacement neighbours for each of them means scans and searches. At hundreds of millions of
   vectors that can't run on every delete. A mark is a single write.
2. **Removing nodes damages the graph.** A deleted node is often a bridge between regions. Cut it
   out and patch locally, and some nodes can become unreachable, which lowers recall without any
   error. A marked node still works as a bridge for searches.
3. **Their memory is fixed slots.** hnswlib preallocates one flat array; a slot can't be freed from
   the middle, but it can be reused.
4. **Concurrency.** Flipping a flag is safe while searches run; rewiring many lists isn't.

Marking is half a feature on its own: deleted nodes keep their memory, searches keep paying for
them, and recall and speed drift as they pile up. Each of those systems pairs it with compaction.

## Why A for GoVec, for now

GoVec is a single node aimed at roughly 100k–1M vectors, and the bug was a failure to restart. A
fixes it with a small change and frees memory immediately. C is a design change that is only
complete with compaction.

Measured on 100,000 clustered 128-d vectors, cosine, `M` 16, `ef_construction` 100,
`ef_search` 100, seed 1, deleting in random order, Apple M2:

| Deleted | Time per delete (mean / p99) | Recall@10 after deletes | Fresh build of survivors |
|---|---|---|---|
| 0% | | 0.9850 | |
| 10% | 8.2 ms / 12.9 ms | 0.9870 | 0.9887 |
| 30% | 7.4 ms / 9.7 ms | 0.9890 | 0.9957 |
| 50% | 5.3 ms / 7.8 ms | 0.9890 | 0.9977 |

Recall holds up, but falls behind a fresh build as deletes accumulate: 0.2 points at 10% deleted,
0.9 at 50%. That is reason 2 above, at small scale. A delete holds the engine's write lock for
those 5–8 ms, so searches wait behind it. The engine also fsyncs the WAL for each delete, which
this measurement leaves out.

## When to revisit

- **Deletes are frequent** (say, more than ~100 a second sustained), or the index grows well past
  1M: the scan becomes the bottleneck. Consider C with compaction, or B.
- **Recall after heavy churn matters:** the gap to a fresh build grows with the share deleted. A
  rebuild after a large share of deletes, or compaction under C, would close it.
- **A flat layout** (vectors and links in arrays indexed by ID, planned for performance) changes
  how nodes are stored; C's slot reuse fits a flat array naturally, so decide there.
