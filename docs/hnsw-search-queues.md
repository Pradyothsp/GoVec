# HNSW search queues: why they're typed, not `container/heap`

**Status:** done
**Date:** 2026-10-22
**Code:** `internal/hnsw/queue.go`, used by the layer search in `internal/hnsw/search.go`
**Tests:** `TestQueues_PopInContainerHeapOrder`, `TestQueues_PeekLeavesTheTop`

## Summary

A layer search keeps two priority queues: the candidates still to explore, nearest first, and the
best results so far, worst first so the one to evict is always on top. They used to be a generic
wrapper around Go's `container/heap`, which boxed every candidate into a new heap object on each
push and pop. Two small queues that hold only search candidates replaced it. Search got 23–26%
faster and went from ~2,570 allocations to 40, with identical results.

## What a search does with the queues

For each node it expands, the search checks the node's neighbours. Each new neighbour is:

1. pushed onto the **candidates** queue, if it could still improve the results;
2. compared with the **worst result so far** (the top of the results queue), and swapped in if it
   is closer.

The search stops when the nearest remaining candidate is farther than the worst result. At 10k
vectors a query does thousands of these pushes, pops and peeks.

## What was wrong

`container/heap` works with any type by passing items as `any`. Turning a concrete value (a node
pointer and a distance) into an `any` copies it into a new heap allocation: every `Push` boxed a
candidate, and every `Pop` returned one in a box. The boxes became garbage at once.

Found while measuring the visited set (`internal/hnsw/visited.go`), which was expected to cut
allocations and didn't. A memory profile showed over 95% of a search's allocations in the heap's
`Push` and `Pop`, and the CPU profile ~8% in `runtime.convT`, the boxing itself, before counting
the garbage collector's work.

## What replaced it

`nearestFirst` (min-heap) and `farthestFirst` (max-heap) are binary heaps on a
`[]searchCandidate` that compare `dist` directly: no interface calls, no `any`, no allocation per
item. Their slices are sized to the search breadth (`ef`) up front. The generic
`internal/hnsw/heap` package had no other user, and four of its methods were already unused, so it
was deleted.

**Ties.** When two candidates have the same distance, the heap's internal steps decide which comes
out first, and that decides which node the search explores next. So the sift-up and sift-down steps
are `container/heap`'s, unchanged. `TestQueues_PopInContainerHeapOrder` pushes and pops 2,000
candidates whose distances mostly tie and checks both queues return them in exactly
`container/heap`'s order; changing one `<` to `<=` fails it.

## Measured

`BenchmarkSearch/hnsw` and `BenchmarkBuild/hnsw` (`internal/test/integration`), 10k × 256-d
clustered vectors, cosine, seed 1 so both versions build the same graph, six alternating rounds on
an Apple M2:

| | Before | After | Change |
|---|---:|---:|---:|
| Search, float32 | 281 µs | 207 µs | −26% |
| Search, int8 | 321 µs | 248 µs | −23% |
| Build, float32 | 53.7 µs/vector | 46.6 | −13% |
| Build, int8 | 61.6 µs/vector | 54.4 | −12% |
| Allocations per search | 2,569 | 40 | −98% |
| Bytes per search | 79 KiB | 39 KiB | −50% |
| Allocations per 10k build | 8.89M | 475k | −95% |
| Recall@10 | 0.9995 / 0.8445 | same | |

The speed-up is larger than the boxing's own 8% of CPU, and not because of garbage collection:
measured with `runtime/metrics`, GC was 5.5% of search CPU before and 3.3% after on this data, and
0.5% and 0.2% on DBpedia text embeddings, where the index's large live heap makes collections rare.
The rest is the work around each allocation: the indirect `Less`/`Swap` calls inside every sift
step, and the allocator's own bookkeeping. Building gains too, because each insert runs the same
search to find its neighbours.

On DBpedia (1536-d) the visited set and these queues together cut search 15% at 10k vectors and
12% at 100k: the same per-node saving, but distance work is a far larger share at 1536 dimensions.

With the visited set before it, search on this benchmark went from 379 µs to 207 µs (−45%) and
from ~2,680 allocations to 40.
