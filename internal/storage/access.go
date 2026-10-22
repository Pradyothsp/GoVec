package storage

// AccessPattern tells the OS how a store's vectors will be read, so it can
// decide how much to read ahead around each one.
type AccessPattern int

const (
	// AccessNormal leaves read-ahead to the kernel's default. It suits brute
	// force: every query reads every vector, though in map order rather than
	// slot order, so a sequential hint (read ahead hard, drop pages behind)
	// would fit worse.
	AccessNormal AccessPattern = iota

	// AccessRandom says each read is one vector at an unpredictable slot,
	// which is how an HNSW search hops through the graph. Read-ahead would pull
	// in neighbouring vectors nobody asked for and, under memory pressure,
	// evict ones that are hot.
	AccessRandom
)
