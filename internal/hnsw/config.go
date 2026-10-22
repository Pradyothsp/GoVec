package hnsw

// Config is how a graph is built and searched. A zero M, Ml, EfSearch or
// EfConstruction takes this package's default when NewGraph builds the graph.
// The two batch settings give 0 their own meaning, documented on each.
type Config struct {
	// M is the maximum number of neighbors to keep for each node. 0 means
	// DefaultM; see its doc comment for the reasoning behind its value.
	M int

	// Ml is the level generation factor.
	// E.g., for Ml = 0.25, each layer is 1/4 the size of the previous layer.
	// 0 means DefaultMl.
	Ml float64

	// EfSearch is the number of nodes to consider in the search phase. 0 means
	// DefaultEfSearch; see its doc comment for the reasoning behind its value.
	// Higher values improve search accuracy (particularly at k > EfSearch) at
	// the expense of latency on every query, including cheap low-k ones.
	EfSearch int

	// EfConstruction is the number of nodes to consider when selecting neighbors
	// for a new node during graph construction (Add); 0 means
	// DefaultEfConstruction. It is independent of EfSearch: a low
	// EfConstruction produces a poorly-connected graph that no amount of
	// EfSearch or M can fully compensate for at query time. See
	// DefaultEfConstruction's doc comment for the reasoning behind its value;
	// higher values build a higher-quality graph at the cost of slower inserts.
	EfConstruction int

	// BatchParallelism caps how many items' graph searches run concurrently
	// within one Add() call's round-based parallelism (see addRound). 0
	// (the default) resolves to runtime.GOMAXPROCS(0) -- see
	// resolveParallelism. Never touched by addSerial/applyLive.
	BatchParallelism int

	// BatchParallelThreshold is the minimum node count an Add() call needs
	// before round-based parallelism kicks in at all; below it, every node
	// runs through addSerial instead (goroutine/WaitGroup overhead isn't
	// worth it for a handful of items -- and a single-node Add(), the most
	// common call shape, must always resolve here regardless of this value).
	// 0 (the default) resolves to DefaultBatchParallelThreshold -- see
	// resolveParallelism.
	BatchParallelThreshold int
}

// withDefaults fills each zero field that has a package default.
func (c Config) withDefaults() Config {
	if c.M <= 0 {
		c.M = DefaultM
	}

	if c.Ml <= 0 {
		c.Ml = DefaultMl
	}

	if c.EfSearch <= 0 {
		c.EfSearch = DefaultEfSearch
	}

	if c.EfConstruction <= 0 {
		c.EfConstruction = DefaultEfConstruction
	}

	return c
}

// Default HNSW parameters, used by NewGraph() and mirrored by
// internal/index/factory.go's server-config fallback defaults so the
// standalone-library and server-config paths agree without hand-copying
// literals in two places. internal/config/config.go's DefaultConfig() keeps
// its own literal instead of importing this package -- config is meant to
// stay independent of any specific index implementation -- but its values
// are expected to match these; keep them in sync by hand if either changes.
const (
	// DefaultM is a good default for OpenAI-style embeddings.
	DefaultM = 16
	// DefaultEfSearch is 100, as in Chroma and Qdrant. An ef sweep on 1536-dim
	// text embeddings (DBpedia, M=16, EfConstruction=100) gave recall@10 0.973
	// and recall@50 0.931 at ef 50, against 0.995 and 0.981 at ef 100; SIFT's
	// 128-dim vectors had hidden that gap (0.997 at ef 50). Larger k needs more:
	// recall@50 reached 0.997 only at ef 200.
	DefaultEfSearch = 100
	// DefaultEfConstruction is 100, as in Chroma and Qdrant. Against 200 (hnswlib's
	// default) it cut batch insert time by 21% on 1536-dim text embeddings and 35%
	// on SIFT, at 10k vectors, with the same recall from EfSearch 100 up; 200 only
	// gained recall at EfSearch 25-50. At 100k vectors 200 looked about half a point
	// to a point better at recall@10, within build-to-build noise, so 100 stays the
	// default and large datasets that need the recall can raise it.
	DefaultEfConstruction = 100
	// DefaultMl makes each layer about a quarter the size of the one below.
	DefaultMl = 0.25
	// DefaultBatchParallelThreshold: below this many nodes in one Add() call,
	// round-based parallelism (see addRound) is skipped entirely in favor of
	// addSerial -- goroutine/WaitGroup overhead isn't worth paying for a
	// handful of items. Picked from a real wall-clock sweep on an 8-core
	// arm64 dev machine (M=16/EfConstruction=200/EfSearch=50, cosine,
	// 128-dim): n=10 parallel was ~19% *slower* than serial (goroutine
	// overhead dominates at that size), n=20 was already ~28% *faster* and
	// the margin only widens from there (n=50: ~55% faster) -- so 20 sits
	// just above the measured crossover, not in the middle of it. See
	// HNSW_BULK_INSERT_PLAN.md §8 for the full sweep.
	DefaultBatchParallelThreshold = 20
)
