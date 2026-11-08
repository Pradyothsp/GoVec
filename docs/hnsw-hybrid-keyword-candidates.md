# Keyword candidates for HNSW hybrid search

**Status:** planned, not started
**Date:** 2026-10-13
**Affects:** `index_type: hnsw` with `enable_hybrid_search: true`, for queries that carry a sparse vector
**Does not affect:** the brute-force engine, dense-only queries, or HNSW with hybrid search off

## Summary

A hybrid query is scored on two things: how close it is in meaning (the dense
score, 70%) and how many keywords it shares (the sparse score, 30%). The
brute-force engine scores every document on both, so the best blend wins.

HNSW does it in two steps. First it walks the graph and collects the documents
nearest **in meaning**. Then it adds keyword scores to those documents and
re-sorts them. A document that isn't near in meaning never makes the first
list, so no keyword match, however strong, can bring it into the results.
Keywords can only re-order what meaning already found.

The fix is to collect candidates from both places: the nearest in meaning from
the graph, and the strongest keyword matches from the inverted index. Then
score them all the same way. Full hybrid search engines work this way.

## Example

A query for "iPhone 15 battery replacement" with `k = 5`:

| Document | Meaning | Keywords | Brute force | HNSW today |
|---|---|---|---|---|
| A: "iPhone 15 battery replacement", tersely worded | a little off | near-perfect | can rank first | **never seen** |
| B–F: general phone-battery articles | close | weak | ranked | ranked |

HNSW's graph returns B–F as the 5 nearest in meaning, re-ranks only those, and
A is never considered.

## How search works today

All references are to the current code on `refactor/unified-metric`.

| Where | What it does |
|---|---|
| `internal/index/hnsw_index.go:179` | `graph.SearchWithDistance(encodedQuery, fetchCount, …)`: the only source of candidates; `fetchCount` is `k`, or `2k` for a non-selective filter |
| `internal/index/hnsw_index.go:181-187` | each graph candidate is filtered, then scored with `score(id, metric.Score(distance))` |
| `internal/index/records.go:435-444` | `scorer`: computes `inverted.Scores(sparseQuery)` once, then blends `0.7·dense + 0.3·max(sparse, 0)` per candidate |
| `internal/core/invertedindex.go` | `Scores(query)`: the sparse score of every document that shares a token with the query |

`Scores` already returns every document that has a keyword match. Today it is
only used to look up the graph's candidates in, and the rest of the map is
thrown away.

## Proposed change

Only `HNSWIndex.Search` changes. With hybrid search on and a non-empty sparse
query:

1. **Meaning candidates.** Search the graph exactly as today.
2. **Keyword candidates.** From the same `Scores` map the scorer uses, take the
   documents whose sparse score is above 0 and that pass the filter, and keep
   the top `N` by sparse score.
3. **Merge.** A document in both lists keeps its graph result. For each
   keyword-only document, compute its dense score directly:
   `metric.Similarity(encodedQuery, node.Vector)`, the same calculation brute
   force uses.
4. **Blend and rank** with the existing scorer and `rankTop(results, k)`.

### Why the scores stay comparable

Graph candidates are scored with `metric.Score(distance)`; keyword-only
candidates with `metric.Similarity`. `TestMetrics_FunctionsAgree`
(`internal/metric/metric_test.go`) already guarantees these agree to within
1e-5 for every metric and both vector types. So a keyword-only document and a
graph document are ranked on the same scale. This holds under scalar
quantization too: the query is encoded once, and `Similarity` is the int8 one.

### The cap `N`

Without a cap, a query containing a common token could pull in thousands of
keyword candidates, each needing a full dense similarity, which would undo what
HNSW saves. The cap bounds that work to `N` extra similarity computations per
query.

Proposed config field, following the "config fields come in sets" rule
(default in `config.go`, check in `validation.go`, `GOVEC_*` override in
`loader.go` using the helpers in `env.go`):

```yaml
engine:
  hybrid_keyword_candidates: 100   # extra candidates HNSW hybrid search takes from keyword matches; 0 = off
```

- **`0` turns the feature off** and restores today's behaviour, which makes
  before/after comparisons in govec-bench a config switch.
- **Validation:** must be `>= 0`.
- **Brute force ignores it.** It already scores every document.

### Selecting the top `N`

The scores map can be large for a common token, so selection keeps a min-heap
of size `N` while iterating it: `O(m log N)` for `m` keyword matches, rather
than sorting all `m`. Filtering happens before the heap, so a selective filter
still yields up to `N` useful candidates.

### Cost per hybrid query

| Step | Cost | New? |
|---|---|---|
| `inverted.Scores` | proportional to the postings of the query's tokens | no: already paid today |
| top-`N` selection | `O(m log N)` | yes |
| dense score for keyword-only candidates | at most `N` similarity computations (`O(N·d)`) | yes |
| blend, sort, truncate | over at most `fetchCount + N` results | slightly more |

At 1536 dimensions, a float32 similarity takes well under a microsecond (see
`DotFloat32`'s comment), so `N = 100` adds in the order of tens of
microseconds. This needs confirming with a benchmark.

## Edge cases

| Case | Behaviour |
|---|---|
| Hybrid search off, or empty sparse query | unchanged: graph candidates only |
| `hybrid_keyword_candidates: 0` | unchanged: graph candidates only |
| Sparse score ≤ 0 | not a keyword candidate; the scorer already treats it as 0 |
| Filter, metadata index on | keyword candidates must be in the allowlist |
| Filter, metadata index off | keyword candidates must pass `core.MatchFilter` |
| Document deleted | the inverted index drops it on delete; the `nodes` lookup guards anyway |
| Empty graph | returns early, as today |
| `k = 0` | skipped before any of this, as today |

### Relation to the filtered-search shortfall

[hnsw-filtered-search-shortfall.md](hnsw-filtered-search-shortfall.md) is a
separate problem: filtered HNSW search can return fewer than `k`. Keyword
candidates can top a short hybrid result up, but only for hybrid queries. They
do not fix that issue, and the plan doesn't try to.

## Implementation steps

1. **`records`:** split the scorer so the sparse scores are computed once and
   shared between the blend and candidate selection. For example, have
   `scorer` also return the scores map.
2. **Top-`N` selection:** a small helper in `internal/index`, taking the
   scores, the filter (allowlist or `MatchFilter`) and `N`, and returning the
   candidate IDs.
3. **`HNSWIndex.Search`:** after the graph pass, add keyword-only candidates
   with their `Similarity` dense score, skipping IDs the graph already returned.
4. **Config:** `HybridKeywordCandidates` with default, validation and env
   override. It reaches the engine through `Options`, like the other engine
   settings.
5. **Docs:**
   - README: replace the "HNSW hybrid search only reranks dense matches" limitation.
   - Roadmap: remove the item.
   - `config.example.yaml`: add the field.

## Tests

All new tests follow Arrange / Act / Assert. No existing tests are restyled
(see the deferred test cleanup).

- **Regression, written first:** the example above. A keyword-only document
  ranks in an HNSW hybrid query's top `k`. It fails today.
- **Matches brute force:** on a small dataset, HNSW hybrid results contain the
  documents brute force ranks highly on keywords.
- **Cap:** with `N = 1`, only the single best keyword match is added.
- **Filters:** a keyword candidate that fails the filter is excluded, with the
  metadata index on and off.
- **Off switch:** with `N = 0`, results equal today's.
- **No duplicates:** a document found both ways appears once.
- **Config:** default, validation of negative values, env override. The last
  one is enforced by `TestLoadFromEnv_EveryFieldHasAnOverride`.

## Measuring

In govec-bench, on a hybrid dataset:
- hybrid recall@k of HNSW against brute force, before (`N = 0`) and after;
- p50/p99 query latency for the same two settings;
- the same at `N` = 25, 100 and 400, to choose the default.

## Decisions needed

1. **Default `N`.** 100 is a starting guess. Should it be fixed, or scale with
   `k` (e.g. `max(N, k)`)?
2. **Default on or off.** On by default changes which results some hybrid
   queries return. That's the point of the fix, and there are no users to
   break, but it's still a behaviour change.
3. **Blending method, out of scope here.** The linear `0.7 / 0.3` blend stays.
   Reciprocal rank fusion is the common alternative and could be a follow-up.
