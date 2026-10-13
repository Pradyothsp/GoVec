package hnsw

import (
	"encoding/binary"
	"fmt"
	"io"
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

// encodingVersionTopology is the ExportTopology/ImportTopology format. Bump it
// whenever that format changes, so an older graph is rejected, not misread.
const encodingVersionTopology = 3

// ExportTopology writes the graph's structure -- M, Ml and every layer's
// edges -- to w, without node vectors or search settings. GoVec's snapshots
// store each vector once, outside the graph (inline in a node record, or in
// the mmap store), and supply them back through ImportTopology's lookup.
//
// The distance functions and EfSearch are the caller's to set, not the
// file's. They used to be saved here too, and an import overwrote the graph's
// Distance and EfSearch with them -- but not Precompute, CachedDistance or
// SquaredDistance, so a loaded graph could rank on one metric and cache for
// another, and a changed ef_search never reached existing data. M and Ml stay:
// they describe how the saved edges were built.
//
// Format (version 3):
//
//	version | M | Ml | nLayers |
//	  for each layer: nNodes | for each node: key | nNeighbors | neighbor keys...
func (h *Graph[K, V]) ExportTopology(w io.Writer) error {
	_, err := multiBinaryWrite(w, encodingVersionTopology, h.M, h.Ml)
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

// ImportTopology reads what ExportTopology wrote from r, replacing the
// graph's layers and its M and Ml, and calling lookupVec for each node key to
// retrieve its vector. lookupVec must return (vector, true) for every key in
// the export; a missing key is an error. Set the graph's distance functions
// before calling it: each node's Precompute cache is rebuilt as it loads.
func (h *Graph[K, V]) ImportTopology(r io.Reader, lookupVec func(K) (V, bool)) error {
	var version int
	_, err := multiBinaryRead(r, &version, &h.M, &h.Ml)
	if err != nil {
		return fmt.Errorf("decode topology parameters: %w", err)
	}
	if version != encodingVersionTopology {
		return fmt.Errorf("incompatible topology version: expected %d, got %d", encodingVersionTopology, version)
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
				// Recomputed on import, same as on insert: the snapshot doesn't
				// carry the cache, which is cheap to rederive.
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
