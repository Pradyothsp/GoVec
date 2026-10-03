# HNSW filtered search returns fewer than `k`

**Status:** known, unfixed, deliberately parked
**Date:** 2026-10-02
**Affects:** `index_type: hnsw` with a metadata filter, both metadata-index settings
**Does not affect:** the brute-force engine, or any unfiltered search
**Regression test:** `internal/index/hnsw_filtered_search_test.go` (committed with `t.Skip`)

## Summary

A filtered HNSW search silently returns fewer results than requested. Nothing
errors and nothing is logged -- the caller asks for 10 and receives 5, with no
indication that 5 more were available.

| Metadata index | Falls short when | Severity |
|---|---|---|
| **Off** — the default (`config.go:130`, `config.yaml:37`) | **any** selectivity | severe; returns roughly `k × selectivity`, so selective filters often return nothing |
| On | selectivity just above 50% | moderate; typically 1–2 short |

The brute-force engine is unaffected: it checks every stored vector against the
filter before ranking (`index.go:271-272`), so its results are exact.

## Mechanism

All line numbers are `internal/index/hnsw_index.go` unless stated.

| Line | Code | Effect |
|---|---|---|
| `:385` | `if filters != nil && idx.metaIndex != nil {` | the allowlist exists only when the metadata index does |
| `:395` | `fetchCount := k` | **index off:** allowlist stays `nil`, the selectivity branch at `:399` never runs, only `k` candidates are ever fetched |
| `:393` | `const selectivityThreshold = 0.5` | **index on:** above 50% the non-selective branch is taken |
| `:403` | `fetchCount = k * 2` | …which fetches a flat `2k` |
| `:426` | `if !core.MatchFilter(...) { continue }` | both paths drop non-matching candidates |
| `:445` | `if k > 0 && len(results) > k { results = results[:k] }` | **the defect:** truncates when there are too many, does nothing when there are too few |

With the index off, the `k` nearest vectors are fetched without regard to the
filter, so on average `k × selectivity` survive. With it on at 51% selectivity,
about `1.02 × k` of the `2k` candidates are expected to match -- barely `k`, so
it frequently comes up short.

## This is arithmetic, not approximation

The tempting explanation is "HNSW is approximate, so of course it sometimes
returns fewer." That is wrong here, and it matters because it points at the
wrong fix.

Measured: the graph returns **exactly** the number of candidates requested,
every time -- unfiltered, and raw through `SearchWithDistance` at both `k` and
`2k`, across 20 queries, zero shortfall. HNSW's approximation shows up in
*which* neighbours come back, not *how many*.

So 100% of the shortfall is the post-filter. There is no recall confound, and
no amount of `efSearch` tuning addresses it.

## Observed behaviour

From the skipped regression test: 400 documents, `k=10`, 20 seeded queries per
row, graph RNG pinned so results are reproducible.

| Case | Matching | Result |
|---|---|---|
| `meta_index_off/90pct` | 360 | FAIL — 14/20 queries short, as low as 6 |
| `meta_index_off/51pct` | 204 | FAIL — 20/20 short, as low as 2 |
| `meta_index_off/10pct` | 40 | FAIL — 20/20 short, several returned 0 |
| `meta_index_off/1pct` | 4 | FAIL — 20/20 short, 17 returned 0 |
| `meta_index_on/51pct` | 204 | FAIL — 5/20 short |
| `meta_index_on/49pct_control` | 196 | pass |
| `meta_index_on/1pct_control` | 4 | pass |

Both controls take the allowlist path and pass across six different graph
seeds, so they are not seed luck.

## Why the suite missed it

- **Index-off path:** no HNSW filter test runs with the metadata index
  disabled. `newTestHNSWIndex` (`hnsw_index_test.go:32`) passes
  `core.NewMetadataIndex()`, the integration `newHNSWEngine` sets
  `EnableMetadataIndex: true`, and the REST filter tests in `api_test.go` use
  the brute-force engine. The default configuration -- the worst one -- was
  untested.
- **Index-on path:** `TestHNSWIndex_Search_WithFilter`
  (`hnsw_index_test.go:175`) does filter 3 of 5 documents (60%), but uses
  `k=10` against a 5-node index, so the `2k` fetch returns the entire graph.
  It then asserts only `LessOrEqual(len(results), 3)`, which passes with zero
  results.

## Fixing it: two paths, two different fixes

The paths differ in what they can know, and that difference drives everything.

### Metadata index on — size the fetch correctly, no retry

`len(allowlist)` gives exact selectivity before the search runs, so the right
fetch size is computable up front:

```go
fetchCount = int(float64(k)/selectivity * safetyMargin)
```

At 51% selectivity with `k=10` that is ~24 rather than the current flat 20.
One search, no retries.

### Metadata index off — fetch, count, refetch

There is no allowlist, so **selectivity cannot be known before filtering**. The
engine only discovers a filter was 1%-selective after discarding 99% of its
candidates. The only correct approach is to widen and retry:

```
fetch = k
loop:
  candidates = SearchWithDistance(query, fetch, nil, 0)
  kept       = candidates after MatchFilter
  if kept >= k            -> done
  if fetch >= graph.Len() -> done   (whole graph seen; kept is provably all there is)
  fetch = grow(fetch)
```

Two properties make this workable:

1. **Growing `fetch` widens exploration automatically.** `graph.go:284` sets
   `resultCap := max(k, efSearch)`, so a larger `fetch` raises the exploration
   bound on its own -- `efSearch` does not need separate scaling.
2. **The `fetch >= graph.Len()` exit makes it correct, not just best-effort.**
   Once the whole graph has been visited, whatever survived the filter is
   genuinely everything that matches.

Verified on the 1%-selectivity case (4 matching of 400, index off), quadrupling
each round:

```
q0: fetch=10 got=10 kept=0 -> fetch=40 got=40 kept=0 -> fetch=160 got=160 kept=2 -> fetch=400 got=400 kept=4
q1: fetch=10 got=10 kept=0 -> fetch=40 got=40 kept=0 -> fetch=160 got=160 kept=1 -> fetch=400 got=400 kept=4
q2: fetch=10 got=10 kept=0 -> fetch=40 got=40 kept=1 -> fetch=160 got=160 kept=3 -> fetch=400 got=400 kept=4
```

All five probed queries recovered the full 4 of 4.

### The cost caveat

Count the work in that trace: **10 + 40 + 160 + 400 = 610** candidate-searches
to find 4 documents in a 400-node graph. Brute-forcing all 400 once costs 400.

At low selectivity, blind geometric retry is **more expensive than not using
the index at all**, because every round restarts from scratch and the last
round scans everything anyway. `SearchWithDistance` is not a cursor; there is
no "give me the next 10."

The cheaper shape is to estimate selectivity from the first round rather than
quadrupling blindly:

```
round 1: fetch=10,  kept=1  -> observed selectivity ~0.1
                            -> need ~k/0.1 = 100, plus margin -> fetch=200
round 2: fetch=200, kept=10 -> done        (total 210, not 610)
```

This also degrades to roughly a single full scan at worst, instead of
overshooting past it.

## Why this is parked

- `govec-bench` never issues a filtered query -- the only `filter` matches in
  that repo are a `docker volume ls --filter` and a tarfile
  `extractall(filter="data")`. Published benchmark numbers are unaffected, so
  parking cannot distort them.
- The broken path is off by default, and GoVec is explicitly single-node and
  not production-targeted (see *Scope & Non-Goals* in `AGENTS.md`).
- The fix is two coordinated changes plus a cost tradeoff, not a cleanup.

## Loose ends to pick up with it

- Unskip `internal/index/hnsw_filtered_search_test.go`.
- That test demands exactly `min(k, matching)` on every query, which commits to
  exhaustive top-up: a bounded over-fetch, however large, still fails
  `meta_index_off/1pct`. Relax it if best-effort is the intended contract.
- The test has no unfiltered control row. The "graph returns exactly `fetch`"
  claim above was verified separately but is not encoded in the suite, so a
  future change to `efSearch` or the graph could make the test fail for an
  unrelated reason while still blaming the filter.
- There is no `meta_index_on/90pct` row. The `>50%` branch is tested only at
  51%, right where `2k` is most strained; its mild end is unexercised.
- The test's document vectors and query vectors are drawn from one RNG stream,
  so changing `numDocs` silently changes every query vector.
- `README.md:385` states `- [x] Metadata filtered search
  (enable_metadata_index: true)` with no qualification. Even the path it names
  under-returns above 50% selectivity.
- Consider whether `enable_metadata_index` should default to `true`. Note this
  is a severity question, not a fix: `meta_index_on/51pct` fails either way.

---

A longer treatment, in the context of a full test-suite audit, is in the
uncommitted local `docs/test-suite-review.md` (§11.5.2).
