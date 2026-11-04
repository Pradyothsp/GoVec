#!/usr/bin/env bash
# Downloads a run of the Kernels workflow (.github/workflows/kernels.yml) and compares the SIMD
# and Go kernels on each runner with benchstat:
#
#   scripts/kernels-results.sh               # the latest successful run
#   scripts/kernels-results.sh 41894085901   # that run
#
# Results go in bench-results/kernels/<run id>/ (gitignored, or OUT=<dir>): one folder per runner
# with its bench.txt and machine.txt, run.json saying which commit and run they came from, and
# summary.txt with what this prints. A run already downloaded is summarised again, not fetched.
set -euo pipefail

repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
run=${1:-$(gh run list -R "$repo" --workflow kernels.yml --status success -L 1 --json databaseId -q '.[0].databaseId')}
if [ -z "$run" ]; then
  echo "no successful Kernels run in $repo" >&2
  exit 1
fi
out=${OUT:-bench-results/kernels/$run}

if [ -d "$out" ] && [ -n "$(ls -A "$out")" ]; then
  echo "run $run already in $out; summarising it"
else
  mkdir -p "$out"
  gh run download "$run" -R "$repo" -D "$out"
  gh run view "$run" -R "$repo" --json databaseId,headBranch,headSha,createdAt,url > "$out/run.json"
fi

if ! command -v benchstat > /dev/null; then
  echo "benchstat not found: go install golang.org/x/perf/cmd/benchstat@latest" >&2
  exit 1
fi

{
  echo "Kernels run $run ($repo)"
  for dir in "$out"/kernels-*/; do
    runner=$(basename "$dir")
    runner=${runner#kernels-}
    # lscpu's "Model name:" on Linux; on macOS the sysctl brand string, the line after go version.
    cpu=$(sed -n -e 's/^Model name: *//p' "$dir/machine.txt" | head -1)
    if [ -z "$cpu" ]; then
      cpu=$(grep -A1 '^go version' "$dir/machine.txt" | tail -1)
    fi
    commit=$(sed -n -e 's/^commit: //p' "$dir/machine.txt" | head -1)

    echo
    echo "== $runner: $cpu (commit $commit)"
    benchstat -col kernel "$dir/bench.txt"
  done
} | tee "$out/summary.txt"
