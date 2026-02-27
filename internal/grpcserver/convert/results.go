package convert

import (
	pb "github.com/Pradyothsp/govec/gen/govec/v1"
	"github.com/Pradyothsp/govec/internal/index"
)

// SearchResultsToProto converts a slice of index.SearchResult to protobuf SearchResult messages.
// Metadata conversion errors are silently dropped (keys with unsupported types are omitted).
func SearchResultsToProto(results []index.SearchResult) []*pb.SearchResult {
	out := make([]*pb.SearchResult, 0, len(results))
	for _, r := range results {
		pbMeta, _ := MetaToProto(r.Meta) //nolint:errcheck // unsupported metadata types are silently omitted from search results
		out = append(out, &pb.SearchResult{
			Id:    r.ID,
			Score: r.Score,
			Meta:  pbMeta,
		})
	}
	return out
}
