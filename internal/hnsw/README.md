# HNSW - Generic Vector Types (GoVec Fork)

[![GoDoc](https://pkg.go.dev/github.com/Pradyothsp/govec/internal/hnsw)](https://pkg.go.dev/github.com/Pradyothsp/govec/internal/hnsw)
![Go workflow status](https://github.com/Pradyothsp/govec/actions/workflows/go.yaml/badge.svg)

Hierarchical Navigable Small World graphs for approximate nearest neighbor search in Go.

**This is a vendored fork of [coder/hnsw](https://github.com/coder/hnsw)** with added support for generic vector types, enabling efficient memory usage with different quantization levels.

## Key Differences from Original

- ✅ **Generic vector type parameter `V`**: Supports `[]float32`, `[]int8`, and custom vector types via the `VectorType` interface.
- ✅ **Int8 Scalar Quantization**: Achieves a 4x memory reduction with minimal precision loss.
- ✅ **Same API, More Flexibility**: The core HNSW algorithm remains consistent, offering expanded utility through generics.
- ✅ **Go 1.26.0+**: Optimized for the latest Go features.

## Attribution

- **Original Project:** [coder/hnsw](https://github.com/coder/hnsw)
- **Original License:** CC0-1.0 (Public Domain)
- **Vendored for:** GoVec vector database
- **Modifications:** Generic type support, int8 distance functions, Go 1.26.0+ compatibility.

Package `hnsw` implements Hierarchical Navigable Small World graphs in Go. You
can read up about how they work [here](https://www.pinecone.io/learn/series/faiss/hnsw/). In essence,
they allow for fast approximate nearest neighbor searches with high-dimensional
vector data.

This package can be thought of as an in-memory alternative to your favorite
vector database (e.g. Pinecone, Weaviate). It implements just the essential
operations:

| Operation | Complexity            | Description                                  |
| --------- | --------------------- | -------------------------------------------- |
| Insert    | $O(log(n))$           | Insert a vector into the graph               |
| Delete    | $O(M^2 \cdot log(n))$ | Delete a vector from the graph               |
| Search    | $O(log(n))$           | Search for the nearest neighbors of a vector |
| Lookup    | $O(1)$                | Retrieve a vector by ID                      |

> [!NOTE]
> Complexities are approximate where $n$ is the number of vectors in the graph
> and $M$ is the maximum number of neighbors each node can have. This [paper](https://arxiv.org/pdf/1603.09320) is a good resource for understanding the effect of
> the various construction parameters.

## Usage

```go
import "github.com/Pradyothsp/govec/internal/hnsw"

// --- Float32 Vectors (Standard Precision) ---
gFloat := hnsw.NewGraph[uint32, []float32]()
gFloat.Distance = hnsw.CosineDistanceFloat32 // Or hnsw.EuclideanDistanceFloat32

gFloat.Add(
    hnsw.MakeNode[uint32, []float32](1, []float32{1, 1, 1}),
    hnsw.MakeNode[uint32, []float32](2, []float32{1, -1, 0.999}),
    hnsw.MakeNode[uint32, []float32](3, []float32{1, 0, -0.5}),
)

neighborsFloat := gFloat.Search(
    []float32{0.5, 0.5, 0.5},
    1,
)
fmt.Printf("Best match (float32): %v\n", neighborsFloat[0].Value)


// --- Int8 Vectors (4x Memory Savings) ---
// Requires converting float32 vectors to int8 using scalar quantization.
// For example: vInt8 := core.QuantizeVector(vFloat)
gInt8 := hnsw.NewGraph[uint32, []int8]()
gInt8.Distance = hnsw.CosineDistanceInt8 // Or hnsw.EuclideanDistanceInt8

gInt8.Add(
    hnsw.MakeNode[uint32, []int8](1, []int8{127, 127, 127}),
    hnsw.MakeNode[uint32, []int8](2, []int8{127, -127, 120}),
    hnsw.MakeNode[uint32, []int8](3, []int8{127, 0, -60}),
)

neighborsInt8 := gInt8.Search(
    []int8{60, 60, 60},
    1,
)
fmt.Printf("Best match (int8): %v\n", neighborsInt8[0].Value)




## Persistence

While all graph operations are in-memory, `hnsw` provides facilities for loading/saving from persistent storage. The binary encoding supports both `[]float32` and `[]int8` vector types seamlessly.

For an `io.Reader`/`io.Writer` interface, use `Graph.Export` and `Graph.Import`.

If you're using a single file as the backend, hnsw provides a convenient `SavedGraph` type instead:

```go
path := "some.graph"
// For float32
g1Float, err := hnsw.LoadSavedGraph[uint32, []float32](path)
if err != nil {
    panic(err)
}
// Insert some vectors
for i := 0; i < 128; i++ {
    g1Float.Add(hnsw.MakeNode[uint32, []float32](uint32(i), []float32{float32(i), float32(i)/2}))
}
g1Float.Distance = hnsw.CosineDistanceFloat32
err = g1Float.Save()
if err != nil {
    panic(err)
}

// For int8
g1Int8, err := hnsw.LoadSavedGraph[uint32, []int8](path)
if err != nil {
    panic(err)
}
// Insert some vectors
for i := 0; i < 128; i++ {
    g1Int8.Add(hnsw.MakeNode[uint32, []int8](uint32(i), []int8{int8(i), int8(i)/2}))
}
g1Int8.Distance = hnsw.CosineDistanceInt8
err = g1Int8.Save()
if err != nil {
    panic(err)
}

// Later... Load the float32 graph
g2Float, err := hnsw.LoadSavedGraph[uint32, []float32](path)
if err != nil {
    panic(err)
}
// Load the int8 graph
g2Int8, err := hnsw.LoadSavedGraph[uint32, []int8](path)
if err != nil {
    panic(err)
}
```

See more:
* [Export](https://pkg.go.dev/github.com/Pradyothsp/govec/internal/hnsw#Graph.Export)
* [Import](https://pkg.go.dev/github.com/Pradyothsp/govec/internal/hnsw#Graph.Import)
* [SavedGraph](https://pkg.go.dev/github.com/Pradyothsp/govec/internal/hnsw#SavedGraph)

We use a fast binary encoding for the graph, so you can expect to save/load
nearly at disk speed. On my M3 Macbook I get these benchmark results:

```
goos: darwin
goarch: arm64
pkg: github.com/Pradyothsp/govec/internal/hnsw
BenchmarkGraph_Import-16            4029            259927 ns/op         796.85 MB/s      496022 B/op       3212 allocs/op
BenchmarkGraph_Export-16            7042            168028 ns/op        1232.49 MB/s      239886 B/op       2388 allocs/op
PASS
ok      github.com/Pradyothsp/govec/internal/hnsw   2.624s
```

when saving/loading a graph of 100 vectors with 256 dimensions.

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
