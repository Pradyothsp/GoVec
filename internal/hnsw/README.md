# hnsw

GoVec's Hierarchical Navigable Small World graph: approximate nearest-neighbour
search over vectors stored as `[]float32` or `[]int8`. You can read up on how
HNSW works [here](https://www.pinecone.io/learn/series/faiss/hnsw/), or in the
[original paper](https://arxiv.org/pdf/1603.09320).

## Origin

This package started as a copy of [coder/hnsw](https://github.com/coder/hnsw)
(CC0-1.0, public domain). GoVec has since changed it enough that it is its own
code, not a fork kept in step with upstream: generic vector types,
`SELECT-NEIGHBORS-HEURISTIC`, corrected search termination, filtered search,
parallel batch insert, per-node distance caching, squared-distance ranking and
topology-only persistence. Upstream's whole-graph `Export`/`Import`,
`SavedGraph` and distance-function registry have been removed.

## Boundary

`hnsw` is a graph algorithm over keys and vectors, with the metric passed in as
a `core.Metric`. It knows nothing of GoVec's records, WAL, config, metadata or
transports; that belongs in `internal/index`. Keep it that way, so the graph can
be tested and benchmarked on its own.

| Operation | Complexity            | Description                                  |
| --------- | --------------------- | -------------------------------------------- |
| Insert    | $O(log(n))$           | Insert a vector into the graph               |
| Delete    | $O(M^2 \cdot log(n))$ | Delete a vector from the graph               |
| Search    | $O(log(n))$           | Search for the nearest neighbors of a vector |
| Lookup    | $O(1)$                | Retrieve a vector by ID                      |

> [!NOTE]
> Complexities are approximate where $n$ is the number of vectors in the graph
> and $M$ is the maximum number of neighbors each node can have.

## Usage

```go
import (
    "github.com/Pradyothsp/govec/internal/core"
    "github.com/Pradyothsp/govec/internal/hnsw"
)

metric, err := core.ResolveMetric[[]float32](core.MetricCosine)
if err != nil {
    panic(err)
}

g := hnsw.NewGraph[uint32, []float32]()
g.Metric = metric

g.Add(
    hnsw.MakeNode[uint32, []float32](1, []float32{1, 1, 1}),
    hnsw.MakeNode[uint32, []float32](2, []float32{1, -1, 0.999}),
    hnsw.MakeNode[uint32, []float32](3, []float32{1, 0, -0.5}),
)

neighbors := g.Search([]float32{0.5, 0.5, 0.5}, 1)
```

For `[]int8` vectors, resolve `core.ResolveMetric[[]int8]` and quantize with
`core.QuantizeVector`. The graph uses the metric's `Distance`, plus its optional
`Precompute`/`CachedDistance` and `SquaredDistance`/`FromSquaredDistance` pairs
when they are set.

## Persistence

`Graph.ExportTopology` writes the graph's structure (`M`, `Ml` and every
layer's edges) and nothing else. `Graph.ImportTopology` reads it back, asking
the caller for each node's vector. GoVec's snapshots store each vector once, in
a node record or the mmap store, so the graph never stores a second copy.

The metric and `EfSearch` are not saved. Set them on the graph before importing:
they describe how the graph is searched, which is the caller's configuration,
not part of the saved structure.

## Performance

By and large the greatest effect you can have on the performance of the graph
is reducing the dimensionality of your data. At 1536 dimensions (OpenAI default),
70% of the query process under default parameters is spent in the distance function.

If you're struggling with slowness / latency, consider:
* Reducing dimensionality
* Increasing $M$

And, if you're struggling with excess memory usage, consider:
* Reducing $M$ a.k.a `Graph.M` (the maximum number of neighbors each node can have)
* Reducing $m_L$ a.a. `Graph.Ml` (the level generation parameter)

### Performance Comparison: Float32 vs. Int8 (1536D Vectors)

| Metric                   | Float32 (No Quantization) | Int8 (Scalar Quantization) | Improvement (Int8 vs. Float32) |
|--------------------------|---------------------------|----------------------------|--------------------------------|
| **Memory per 1536D Vector** | 6,144 bytes               | 1,536 bytes                | **4x Reduction**               |
| **Search Speed**         | 100%                      | ~95%                       | Minor performance hit          |
| **Recall@10**            | 100%                      | ~99.5%                     | Negligible precision loss      |

## Memory Overhead

The memory overhead of a graph looks like:

$$
\displaylines{
mem_{graph} = n \cdot \log(n) \cdot \text{size(id)} \cdot M \\
mem_{vector\_data} = n \cdot d \cdot \text{size(V)} \\
mem_{total} = mem_{graph} + mem_{vector\_data}
}
$$

where:
* $n$ is the number of vectors in the graph
* $\text{size(id)}$ is the average size of the key in bytes (e.g., 4 bytes for `uint32`)
* $M$ is the maximum number of neighbors each node can have
* $d$ is the dimensionality of the vectors
* $\text{size(V)}$ is the size of each vector dimension (4 bytes for `float32`, 1 byte for `int8`)
* $mem_{graph}$ is the memory used by the graph structure across all layers
* $mem_{vector\_data}$ is the memory used by the vectors themselves

You can infer that:
* Connectivity ($M$) is very expensive if keys are large.
* If $d \cdot \text{size(V)}$ is far larger than $M \cdot \text{size(id)}$, you should expect linear memory usage spent on representing vector data.
* If $d \cdot \text{size(V)}$ is far smaller than $M \cdot \text{size(id)}$, you should expect $n \cdot \log(n)$ memory usage spent on representing graph structure.

In the example of a graph with 256 dimensions, and $M = 16$, with 4 byte keys (`uint32`):

*   **For `[]float32` vectors:**
    *   `mem_vector_data` per vector: $256 \cdot 4 = 1024$ bytes
    *   `mem_graph` per vector: $16 \cdot 4 = 64$ bytes (for neighbors)
    *   Total per vector: $1024 + 64 = 1088$ bytes (approx.)

*   **For `[]int8` vectors (4x memory reduction):**
    *   `mem_vector_data` per vector: $256 \cdot 1 = 256$ bytes
    *   `mem_graph` per vector: $16 \cdot 4 = 64$ bytes (for neighbors)
    *   Total per vector: $256 + 64 = 320$ bytes (approx.)
