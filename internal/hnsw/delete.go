package hnsw

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
		node, ok := layer.nodes[key]
		if !ok {
			continue // Node not found in this layer, continue to next.
		}

		delete(layer.nodes, key) // Remove the node from the current layer.

		if len(layer.nodes) == 0 {
			deleteLayer[i] = struct{}{} // Mark layer as empty.
		}

		mCurMax := h.M

		if i == 0 {
			mCurMax = 2 * h.M // base layer allows up to 2*M connections -- see Add's comment
		}

		node.isolate(mCurMax, dist) // Disconnect the node and replenish its former neighbors.
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
