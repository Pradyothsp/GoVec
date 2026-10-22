package hnsw

import (
	"cmp"
	"slices"
)

// Delete removes a node from the graph specified by its key.
// It attempts to preserve the graph's clustering properties by reconnecting
// affected neighbors, ensuring graph integrity after removal.
// Returns true if the node was found and deleted, false otherwise.
func (h *Graph[K, V]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false // Cannot delete from an empty graph.
	}

	deleteLayer := map[int]struct{}{} // Keep track of layers that become empty.
	var deleted bool
	dist := h.rankDistFuncFor() // built once, reused across every layer this key appears in; Delete never returns a distance value
	// Iterate through all layers, from highest to lowest, to remove the node.
	for i, layer := range h.layers {
		if _, ok := layer.nodes[key]; !ok {
			continue // Node not found in this layer, continue to next.
		}

		layer.remove(key)

		if len(layer.nodes) == 0 {
			deleteLayer[i] = struct{}{} // Mark layer as empty.
		}

		mCurMax := h.M

		if i == 0 {
			mCurMax = 2 * h.M // base layer allows up to 2*M connections -- see Add's comment
		}

		layer.detach(key, mCurMax, dist)
		deleted = true
	}

	// If any layers became empty, reconstruct the layers slice to remove them.
	if len(deleteLayer) > 0 {
		newLayers := make([]*layer[K, V], 0, len(h.layers)-len(deleteLayer))

		for i, layer := range h.layers {
			if _, ok := deleteLayer[i]; ok {
				continue // Skip empty layers.
			}

			newLayers = append(newLayers, layer)
		}

		h.layers = newLayers
	}

	return deleted
}

// detach removes every link to key left in the layer and repairs each node
// that lost one. Links are one-way: a node can point at key without key
// pointing back, so removing only key's own neighbours' links left those
// dangling. A snapshot then saved a link to a node that no longer existed,
// and loading it failed: after deletes, the server couldn't restart. Finding
// them takes a scan of the layer, since nothing records who links to a node.
func (l *layer[K, V]) detach(key K, m int, dist distFunc[V]) {
	var lost []*layerNode[K, V]

	for _, n := range l.nodes {
		if n.indexOfNeighbor(key) >= 0 {
			n.removeNeighbor(key)
			lost = append(lost, n)
		}
	}

	// Repair in key order: map order is random, and each repair can change
	// the next one's candidates, so the same deletes must repair the same way.
	slices.SortFunc(lost, func(a, b *layerNode[K, V]) int {
		return cmp.Compare(a.Key, b.Key)
	})

	for _, n := range lost {
		n.replenish(m, dist)
	}
}
