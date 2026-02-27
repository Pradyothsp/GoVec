package main

import (
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/Pradyothsp/govec/internal/hnsw"
)

func main() {
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	g := hnsw.NewGraph[int, []float32]()
	g.Distance = hnsw.CosineDistanceFloat32
	g.Add(
		hnsw.MakeNode(1, []float32{1, 1, 1}),
		hnsw.MakeNode(2, []float32{1, -1, 0.999}),
		hnsw.MakeNode(3, []float32{1, 0, -0.5}),
	)

	neighbors := g.Search(
		[]float32{0.5, 0.5, 0.5},
		1,
	)
	log.Info().Int("id", neighbors[0].Key).Msg("best friend")
}
