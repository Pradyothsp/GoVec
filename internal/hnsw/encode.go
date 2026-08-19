package hnsw

import (
	"bufio"
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/google/renameio"
)

// errorEncoder is a helper type to encode multiple values

var byteOrder = binary.LittleEndian

// binaryRead reads data from an io.Reader into the provided data interface, handling various types.
// It supports reading integers, strings, slices of float32, and slices of int8.
// The size (in bytes) of the data read is returned along with any error.
func binaryRead(r io.Reader, data interface{}) (int, error) {
	switch v := data.(type) {
	case *int:
		br, ok := r.(io.ByteReader)
		if !ok {
			return 0, fmt.Errorf("reader does not implement io.ByteReader")
		}

		i, err := binary.ReadVarint(br)
		if err != nil {
			return 0, err
		}

		*v = int(i)
		// TODO: this will usually overshoot size, consider returning actual varint size.
		return binary.MaxVarintLen64, nil

	case *string:
		var ln int
		_, err := binaryRead(r, &ln) // Read string length first.
		if err != nil {
			return 0, err
		}

		s := make([]byte, ln)
		bytesRead, err := binaryRead(r, &s) // Read string bytes.
		*v = string(s)
		return bytesRead + binary.MaxVarintLen64, err // Include length of the varint.

	case *[]float32:
		var ln int
		_, err := binaryRead(r, &ln) // Read slice length.
		if err != nil {
			return 0, err
		}

		*v = make([]float32, ln)
		return binary.Size(*v) + binary.MaxVarintLen64, binary.Read(r, byteOrder, *v) // Read slice elements.

	case *[]int8:
		var ln int
		_, err := binaryRead(r, &ln) // Read slice length.
		if err != nil {
			return 0, err
		}

		*v = make([]int8, ln)
		return binary.Size(*v) + binary.MaxVarintLen64, binary.Read(r, byteOrder, *v) // Read slice elements.

	case io.ReaderFrom:
		n, err := v.ReadFrom(r)
		return int(n), err

	default:
		// For other types, use standard binary.Read.
		return binary.Size(data), binary.Read(r, byteOrder, data)
	}
}

// binaryWrite writes data to an io.Writer from the provided data interface, handling various types.
// It supports writing integers, strings, slices of float32, and slices of int8.
// The size (in bytes) of the data written is returned along with any error.
func binaryWrite(w io.Writer, data any) (int, error) {
	switch v := data.(type) {
	case int:
		var buf [binary.MaxVarintLen64]byte
		n := binary.PutVarint(buf[:], int64(v))
		bytesWritten, err := w.Write(buf[:n])
		return bytesWritten, err
	case io.WriterTo:
		n, err := v.WriteTo(w)
		return int(n), err
	case string:
		n1, err := binaryWrite(w, len(v)) // Write string length first.
		if err != nil {
			return n1, err
		}
		n2, err := io.WriteString(w, v) // Write string bytes.
		if err != nil {
			return n1 + n2, err
		}

		return n1 + n2, nil
	case []float32:
		n1, err := binaryWrite(w, len(v)) // Write slice length.
		if err != nil {
			return n1, err
		}
		return n1 + binary.Size(v), binary.Write(w, byteOrder, v) // Write slice elements.
	case []int8:
		n1, err := binaryWrite(w, len(v)) // Write slice length.
		if err != nil {
			return n1, err
		}
		return n1 + binary.Size(v), binary.Write(w, byteOrder, v) // Write slice elements.

	default:
		// For other types, use standard binary.Write.
		sz := binary.Size(data)
		err := binary.Write(w, byteOrder, data)
		if err != nil {
			return 0, fmt.Errorf("encoding %T: %w", data, err)
		}
		return sz, err
	}
}

// multiBinaryWrite writes multiple data elements sequentially to an io.Writer.
// It uses binaryWrite for each element and aggregates the total bytes written.
func multiBinaryWrite(w io.Writer, data ...any) (int, error) {
	var written int
	for _, d := range data {
		n, err := binaryWrite(w, d)
		written += n
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// multiBinaryRead reads multiple data elements sequentially from an io.Reader.
// It uses binaryRead for each element and aggregates the total bytes read.
// An error includes the type and index of the element that failed to be read.
func multiBinaryRead(r io.Reader, data ...any) (int, error) {
	var read int
	for i, d := range data {
		n, err := binaryRead(r, d)
		read += n
		if err != nil {
			return read, fmt.Errorf("reading %T at index %v: %w", d, i, err)
		}
	}
	return read, nil
}

const (
	encodingVersion         = 1 // binary format for Export/Import (includes node vectors)
	encodingVersionTopology = 2 // binary format for ExportTopology/ImportTopology (no vectors)
)

// Increment this version if the serialization format changes to prevent
// loading of incompatible older graph files.

// Export writes the entire graph's structure and node data to the provided io.Writer
// in a binary format. This includes graph parameters (M, Ml, EfSearch), the
// registered name of the distance function used, and the nodes across all layers.
// It determines the appropriate distance function name based on the graph's
// generic vector type (V) and uses a custom binary encoding scheme.
//
// The generic type V of the Graph must be compatible with the encoding/decoding
// operations (i.e., []float32 or []int8).
func (h *Graph[K, V]) Export(w io.Writer) error {
	// Determine distance function name based on vector type V.
	// This name is saved so the correct function can be restored during Import.
	var distFuncName string
	var ok bool

	// Use type assertion on the underlying function type to find its registered name.
	switch fn := any(h.Distance).(type) {
	case DistanceFunc[[]float32]:
		distFuncName, ok = distanceFuncToNameFloat32(fn)
	case DistanceFunc[[]int8]:
		distFuncName, ok = distanceFuncToNameInt8(fn)
	default:
		return fmt.Errorf("unsupported vector type in distance function for Export: %T", h.Distance)
	}

	if !ok {
		return fmt.Errorf("distance function %v must be registered with RegisterDistanceFuncFloat32 or RegisterDistanceFuncInt8 for persistence", h.Distance)
	}

	// Write graph metadata and parameters.
	_, err := multiBinaryWrite(
		w,
		encodingVersion, // Protocol version for compatibility checks.
		h.M,
		h.Ml,
		h.EfSearch,
		distFuncName, // Store the name of the distance function.
	)
	if err != nil {
		return fmt.Errorf("encode graph parameters: %w", err)
	}

	// Write the number of layers.
	_, err = binaryWrite(w, len(h.layers))
	if err != nil {
		return fmt.Errorf("encode number of layers: %w", err)
	}

	// Iterate through each layer and encode its nodes and their neighbors.
	for _, layer := range h.layers {
		_, err = binaryWrite(w, len(layer.nodes)) // Number of nodes in current layer.
		if err != nil {
			return fmt.Errorf("encode number of nodes in layer: %w", err)
		}
		for _, node := range layer.nodes {
			// Encode node key, vector value, and number of neighbors.
			_, err = multiBinaryWrite(w, node.Key, node.Value, len(node.neighbors))
			if err != nil {
				return fmt.Errorf("encode node data for key %v: %w", node.Key, err)
			}

			// Encode keys of all neighbors.
			for _, neighbor := range node.neighbors {
				_, err = binaryWrite(w, neighbor.Key)
				if err != nil {
					return fmt.Errorf("encode neighbor key %v for node %v: %w", neighbor.Key, node.Key, err)
				}
			}
		}
	}

	return nil
}

// Import reads a graph from the provided io.Reader, reconstructing its structure
// and node data. It expects the binary format to match that produced by Export.
//
// The generic type V of the Graph must be compatible with the encoded data
// (i.e., []float32 or []int8).
// The imported graph's parameters (M, Ml, EfSearch) will be set from the
// stored data, potentially overriding the current graph's values.
func (h *Graph[K, V]) Import(r io.Reader) error {
	var (
		version      int    // Protocol version.
		distFuncName string // Name of the distance function.
	)
	// Read graph metadata and parameters.
	_, err := multiBinaryRead(r, &version, &h.M, &h.Ml, &h.EfSearch, &distFuncName)
	if err != nil {
		return fmt.Errorf("decode graph parameters: %w", err)
	}

	// Check encoding version for compatibility.
	if version != encodingVersion {
		return fmt.Errorf("incompatible encoding version: expected %d, got %d", encodingVersion, version)
	}

	// Restore the distance function based on its name and the graph's vector type V.
	switch any(h.Distance).(type) {
	case DistanceFunc[[]float32]:
		distFunc, found := distanceFuncsFloat32[distFuncName]
		if !found {
			return fmt.Errorf("unknown distance function %q for float32 vectors during import", distFuncName)
		}
		// Type assertion is safe here because we're in the float32 case and the function is known.
		if converted, assertOk := any(distFunc).(DistanceFunc[V]); assertOk {
			h.Distance = converted
		} else {
			return fmt.Errorf("type assertion failed for float32 distance function %q: received incompatible type", distFuncName)
		}
	case DistanceFunc[[]int8]:
		distFunc, found := distanceFuncsInt8[distFuncName]
		if !found {
			return fmt.Errorf("unknown distance function %q for int8 vectors during import", distFuncName)
		}
		// Type assertion is safe here because we're in the int8 case and the function is known.
		if converted, assertOk := any(distFunc).(DistanceFunc[V]); assertOk {
			h.Distance = converted
		} else {
			return fmt.Errorf("type assertion failed for int8 distance function %q: received incompatible type", distFuncName)
		}
	default:
		return fmt.Errorf("unsupported vector type for distance function %q during import: %T", distFuncName, h.Distance)
	}

	// Initialize RNG if it's not set.
	if h.Rng == nil {
		h.Rng = defaultRand()
	}

	// Read the number of layers.
	var nLayers int
	_, err = binaryRead(r, &nLayers)
	if err != nil {
		return fmt.Errorf("decode number of layers: %w", err)
	}

	h.layers = make([]*layer[K, V], nLayers)
	// Read each layer's nodes.
	for i := 0; i < nLayers; i++ {
		var nNodes int
		_, err = binaryRead(r, &nNodes) // Number of nodes in current layer.
		if err != nil {
			return fmt.Errorf("decode number of nodes in layer %d: %w", i, err)
		}

		nodes := make(map[K]*layerNode[K, V], nNodes)
		neighborKeysByNode := make(map[K][]K, nNodes)
		// Read each node's data.
		for j := 0; j < nNodes; j++ {
			var key K
			var vec V
			var nNeighbors int
			_, err = multiBinaryRead(r, &key, &vec, &nNeighbors)
			if err != nil {
				return fmt.Errorf("decode node %d in layer %d: %w", j, i, err)
			}

			// Read neighbor keys (pointers resolved in a second pass below,
			// once every node in the layer has been read).
			neighborKeys := make([]K, nNeighbors)
			for k := 0; k < nNeighbors; k++ {
				var neighbor K
				_, err = binaryRead(r, &neighbor)
				if err != nil {
					return fmt.Errorf("decode neighbor %d for node %v in layer %d: %w", k, key, i, err)
				}
				neighborKeys[k] = neighbor
			}

			nodes[key] = &layerNode[K, V]{
				Node: Node[K, V]{
					Key:   key,
					Value: vec,
				},
				// Recomputed on import, same as on insert -- a persisted
				// snapshot doesn't carry the cache on disk (it's cheap to
				// rederive and doing so keeps the on-disk format unchanged).
				precomputed: h.precompute(vec),
			}
			neighborKeysByNode[key] = neighborKeys
		}

		// After all nodes in the layer are read, resolve neighbor keys to pointers.
		for key, neighborKeys := range neighborKeysByNode {
			node := nodes[key]
			node.neighbors = make([]*layerNode[K, V], 0, len(neighborKeys))
			for _, neighborKey := range neighborKeys {
				neighborNode, found := nodes[neighborKey]
				if !found {
					// This indicates data corruption or an invalid neighbor key.
					return fmt.Errorf("failed to resolve neighbor %v for node %v in layer %d", neighborKey, node.Key, i)
				}
				node.neighbors = append(node.neighbors, neighborNode)
			}
		}
		h.layers[i] = &layer[K, V]{nodes: nodes}
	}

	return nil
}

// ExportTopology writes the graph structure (parameters + edges) to w without
// embedding node vectors.  This is used with the mmap storage backend where
// vectors live in mmap-backed files rather than in the GOB snapshot.
//
// Format (version 2):
//
//	version | M | Ml | EfSearch | distFuncName | nLayers |
//	  for each layer: nNodes | for each node: key | nNeighbors | neighbor keys...
func (h *Graph[K, V]) ExportTopology(w io.Writer) error {
	var distFuncName string
	var ok bool

	switch fn := any(h.Distance).(type) {
	case DistanceFunc[[]float32]:
		distFuncName, ok = distanceFuncToNameFloat32(fn)
	case DistanceFunc[[]int8]:
		distFuncName, ok = distanceFuncToNameInt8(fn)
	default:
		return fmt.Errorf("unsupported vector type in distance function for ExportTopology: %T", h.Distance)
	}
	if !ok {
		return fmt.Errorf("distance function must be registered for persistence")
	}

	_, err := multiBinaryWrite(w, encodingVersionTopology, h.M, h.Ml, h.EfSearch, distFuncName)
	if err != nil {
		return fmt.Errorf("encode topology parameters: %w", err)
	}

	_, err = binaryWrite(w, len(h.layers))
	if err != nil {
		return fmt.Errorf("encode topology layer count: %w", err)
	}

	for _, layer := range h.layers {
		_, err = binaryWrite(w, len(layer.nodes))
		if err != nil {
			return fmt.Errorf("encode topology node count: %w", err)
		}
		for _, node := range layer.nodes {
			_, err = multiBinaryWrite(w, node.Key, len(node.neighbors))
			if err != nil {
				return fmt.Errorf("encode topology node key %v: %w", node.Key, err)
			}
			for _, neighbor := range node.neighbors {
				_, err = binaryWrite(w, neighbor.Key)
				if err != nil {
					return fmt.Errorf("encode topology neighbor %v of node %v: %w", neighbor.Key, node.Key, err)
				}
			}
		}
	}
	return nil
}

// ImportTopology reads a topology-only graph export (version 2) from r, calling
// lookupVec for each node key to retrieve its mmap-backed vector.
// lookupVec must return (vector, true) for every key that appears in the export;
// a missing key causes a descriptive error.
func (h *Graph[K, V]) ImportTopology(r io.Reader, lookupVec func(K) (V, bool)) error {
	var (
		version      int
		distFuncName string
	)
	_, err := multiBinaryRead(r, &version, &h.M, &h.Ml, &h.EfSearch, &distFuncName)
	if err != nil {
		return fmt.Errorf("decode topology parameters: %w", err)
	}
	if version != encodingVersionTopology {
		return fmt.Errorf("incompatible topology version: expected %d, got %d", encodingVersionTopology, version)
	}

	// Restore distance function.
	switch any(h.Distance).(type) {
	case DistanceFunc[[]float32]:
		distFunc, found := distanceFuncsFloat32[distFuncName]
		if !found {
			return fmt.Errorf("unknown distance function %q for float32 during topology import", distFuncName)
		}
		if converted, ok := any(distFunc).(DistanceFunc[V]); ok {
			h.Distance = converted
		} else {
			return fmt.Errorf("type assertion failed for float32 distance function %q", distFuncName)
		}
	case DistanceFunc[[]int8]:
		distFunc, found := distanceFuncsInt8[distFuncName]
		if !found {
			return fmt.Errorf("unknown distance function %q for int8 during topology import", distFuncName)
		}
		if converted, ok := any(distFunc).(DistanceFunc[V]); ok {
			h.Distance = converted
		} else {
			return fmt.Errorf("type assertion failed for int8 distance function %q", distFuncName)
		}
	default:
		return fmt.Errorf("unsupported vector type for distance function %q during topology import", distFuncName)
	}

	if h.Rng == nil {
		h.Rng = defaultRand()
	}

	var nLayers int
	_, err = binaryRead(r, &nLayers)
	if err != nil {
		return fmt.Errorf("decode topology layer count: %w", err)
	}

	h.layers = make([]*layer[K, V], nLayers)
	for i := 0; i < nLayers; i++ {
		var nNodes int
		_, err = binaryRead(r, &nNodes)
		if err != nil {
			return fmt.Errorf("decode topology node count in layer %d: %w", i, err)
		}

		nodes := make(map[K]*layerNode[K, V], nNodes)
		neighborKeysByNode := make(map[K][]K, nNodes)
		for j := 0; j < nNodes; j++ {
			var key K
			var nNeighbors int
			_, err = multiBinaryRead(r, &key, &nNeighbors)
			if err != nil {
				return fmt.Errorf("decode topology node %d in layer %d: %w", j, i, err)
			}

			vec, ok := lookupVec(key)
			if !ok {
				return fmt.Errorf("topology import: no vector found for key %v", key)
			}

			neighborKeys := make([]K, nNeighbors)
			for k := 0; k < nNeighbors; k++ {
				var neighbor K
				_, err = binaryRead(r, &neighbor)
				if err != nil {
					return fmt.Errorf("decode topology neighbor %d of node %v in layer %d: %w", k, key, i, err)
				}
				neighborKeys[k] = neighbor
			}

			nodes[key] = &layerNode[K, V]{
				Node: Node[K, V]{Key: key, Value: vec},
				// See the equivalent Import() site's comment: recomputed on
				// import rather than persisted.
				precomputed: h.precompute(vec),
			}
			neighborKeysByNode[key] = neighborKeys
		}

		// Resolve neighbor keys to pointers.
		for key, neighborKeys := range neighborKeysByNode {
			node := nodes[key]
			node.neighbors = make([]*layerNode[K, V], 0, len(neighborKeys))
			for _, nk := range neighborKeys {
				nbNode, found := nodes[nk]
				if !found {
					return fmt.Errorf("topology import: failed to resolve neighbor %v for node %v in layer %d", nk, node.Key, i)
				}
				node.neighbors = append(node.neighbors, nbNode)
			}
		}
		h.layers[i] = &layer[K, V]{nodes: nodes}
	}
	return nil
}

// SavedGraph is a convenience wrapper around a Graph that simplifies persistence
// to and from a specified file path. It handles file operations and atomic
// saving, making it more user-friendly than directly calling Graph.Export and Graph.Import.
type SavedGraph[K cmp.Ordered, V VectorType] struct {
	*Graph[K, V]        // Embeds the HNSW graph itself.
	Path         string // The file path where the graph data is stored.
}

// LoadSavedGraph opens a graph from a file at the given path.
// If the file exists and contains data, the graph is imported from it.
// If the file does not exist or is empty, a new, empty graph with default
// parameters is returned.
//
// This function does not keep the file descriptor open; the file is read
// (or created) and then closed, allowing for flexible usage.
func LoadSavedGraph[K cmp.Ordered, V VectorType](path string) (*SavedGraph[K, V], error) {
	//nolint:gosec // G304: Path from the function parameter is acceptable as it's an intended file operation.
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("failed to open/create graph file at %q: %w", path, err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			// Log the error but don't return it, as a primary error might already be set.
			_ = err
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat graph file %q: %w", path, err)
	}

	g := NewGraph[K, V]() // Always start with a new graph instance.
	if info.Size() > 0 {  // If the file has content, try to import.
		err = g.Import(bufio.NewReader(f))
		if err != nil {
			return nil, fmt.Errorf("failed to import graph from %q: %w", path, err)
		}
	}

	return &SavedGraph[K, V]{Graph: g, Path: path}, nil
}

// Save writes the current state of the encapsulated Graph to its configured Path.
// It uses renameio.TempFile for atomic writes, ensuring data integrity by
// writing to a temporary file first and then atomically replacing the original.
func (g *SavedGraph[K, V]) Save() error {
	tmp, err := renameio.TempFile("", g.Path)
	if err != nil {
		return fmt.Errorf("failed to create temporary file for saving graph: %w", err)
	}
	defer func() {
		//nolint:errcheck // Best effort cleanup if atomic replacement fails or is not reached.
		_ = tmp.Cleanup()
	}()

	wr := bufio.NewWriter(tmp)
	err = g.Export(wr) // Export the graph data to the temporary file.
	if err != nil {
		return fmt.Errorf("failed to export graph to temporary file: %w", err)
	}

	err = wr.Flush() // Ensure all buffered data is written to the temporary file.
	if err != nil {
		return fmt.Errorf("failed to flush temporary file writer: %w", err)
	}

	err = tmp.CloseAtomicallyReplace() // Atomically replace the original file with the temporary one.
	if err != nil {
		return fmt.Errorf("failed to atomically replace graph file %q: %w", g.Path, err)
	}

	return nil
}
