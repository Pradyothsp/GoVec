# Telling the OS how the mmap store is read

**Status:** done in code; the measurement that shows its effect is still to run
**Date:** 2026-10-22
**Code:** `internal/storage/access.go`, `MmapStore.SetAccessPattern`, set in `internal/index/factory.go`
**Tests:** `TestMmapStore_SetAccessPattern_ReachesEveryChunk`, `TestNewEngine_MmapAccessPattern`

## Summary

With `storage.enable_mmap: true`, vectors live in chunk files the OS loads into memory on demand.
An HNSW index now tells the OS those files are read at random, so it stops reading ahead around
each vector. Brute force keeps the kernel's default. It only matters when the vectors don't all
fit in memory, which is when mmap mode is used.

## Why

When a program touches a page of a mapped file that isn't in memory, the OS reads it from disk,
and usually reads ahead too, loading the pages after it on the bet that the file is read front to
back.

An HNSW search doesn't read front to back. It hops from one vector to another wherever the graph's
links lead, touching one vector (6 KB at 1536 float32 dimensions) per hop. Read-ahead loads
neighbouring vectors nobody asked for. With memory to spare that only wastes some disk reads.
When memory is short, the wasted pages push out vectors that are actually in use, and the next
search goes back to disk for them.

`madvise(MADV_RANDOM)` tells the OS not to read ahead for that mapping. The store had a method
wrapping it, but nothing called it.

## What it does

- `storage.AccessPattern` has two values. `AccessNormal` is the kernel's default read-ahead.
  `AccessRandom` is one vector at a time, anywhere. It's a type of its own so `index` never needs
  Unix-only code; the stub for other platforms accepts it and does nothing.
- `MmapStore.SetAccessPattern` applies the pattern to every mapped chunk and keeps it: `madvise`
  applies per mapping, so each chunk created later gets it as it is mapped.
- `NewEngine` sets `AccessRandom` when `index_type` is `hnsw`.

**Why not "sequential" for brute force.** Brute force reads every vector on every query, but in Go
map order, not file order, so a sequential hint wouldn't describe it. On Linux that hint also drops
pages soon after they are read, which hurts a scan that comes back to every page on the next query.
The kernel default fits better.

## Still to measure

The unit tests can only check that the hint is given, not that it helps: with every vector cached,
it changes nothing. The case it is for is a govec-bench run of the `govec-mmap` configuration at
DBpedia 100k with the container's memory limit set below the vectors' size, before and after.
Earlier runs disagreed on what mmap costs queries (~27% slower on 2026-10-06, level on 2026-10-08),
so measure rather than assume a gain.
