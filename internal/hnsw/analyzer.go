package hnsw

import "cmp"

// Analyzer is a struct that holds a graph and provides
// methods for analyzing it. It offers no compatibility guarantee
// as the methods of measuring the graph's health with change
// with the implementation.
type Analyzer[K cmp.Ordered, V VectorType] struct {
	Graph *Graph[K, V]
}

// Height returns the maximum layer height in the HNSW graph.
func (a *Analyzer[K, V]) Height() int {
	return len(a.Graph.layers)
}

// Connectivity returns the average number of edges in the
// graph for each non-empty layer.
func (a *Analyzer[K, V]) Connectivity() []float64 {
	var layerConnectivity []float64
	for _, layer := range a.Graph.layers {
		if len(layer.nodes) == 0 {
			continue
		}

		var sum float64
		for _, node := range layer.nodes {
			sum += float64(len(node.neighbors))
		}

		layerConnectivity = append(layerConnectivity, sum/float64(len(layer.nodes)))
	}

	return layerConnectivity
}

// Topography returns the number of nodes in each layer of the graph.
func (a *Analyzer[K, V]) Topography() []int {
	height := a.Height()
	topography := make([]int, 0, height)
	for _, layer := range a.Graph.layers {
		topography = append(topography, len(layer.nodes))
	}
	return topography
}
