# Contributing to GoVec

Thanks for your interest. Bug reports, fixes and focused features are all welcome. For anything
large, open an issue first so we can agree on the approach before you write the code. GoVec is
deliberately small and single-node (see [AGENTS.md](AGENTS.md#what-govec-is-and-isnt)), so some
features are out of scope by design.

## Development setup

Requirements:

- [Go](https://go.dev/dl/) 1.26+
- [Task](https://taskfile.dev/installation/) (task runner)
- [pre-commit](https://pre-commit.com/) (git hooks)

```bash
git clone https://github.com/Pradyothsp/govec.git
cd govec

task install-tools    # gotestsum, golangci-lint, govulncheck, swag, buf
task deps             # download and tidy Go modules
pre-commit install    # run fmt, vet, lint and vulncheck on every commit

task test             # make sure everything passes before you start
```

## Everyday commands

Run `task --list` for the full set.

| Command | What it does |
|---|---|
| `task run` | Start the server on `:8000` using `config.yaml` |
| `task build` | Build the `govec` binary |
| `task test` | Run all tests |
| `task test:pkg PKG=internal/index` | Test one package |
| `task test:watch` | Re-run tests on file changes |
| `task test:all` | Race detector and coverage report (`coverage.html`) |
| `task test:benchmark` | Go benchmarks |
| `task check` | vet, gofmt, golangci-lint, govulncheck, buf lint |
| `task fmt:fix` | Apply formatting |
| `task gen` | Regenerate Swagger docs and gRPC stubs |
| `task clean` | Remove the binary, coverage files and local data files |

## Project conventions

[AGENTS.md](AGENTS.md) is the reference for how the code is organized, the conventions it
follows, and the known traps. It is written for AI agents and humans alike. The essentials:

- **Keep REST and gRPC in step.** Any operation you add or change must behave identically over
  both, and `internal/test/integration/transport_parity_test.go` must cover it.
- **Never hand-edit generated code.** Edit `proto/` or the handler Swagger annotations, then run
  `task gen`.
- **New config fields** need a default, validation, and a `GOVEC_*` environment override. A
  test fails if the override is missing.
- **Tests** use testify. Cross-package behaviour goes in `internal/test/integration/`. Use
  `t.TempDir()` for any file the test writes.

## Submitting a pull request

Before you open it:

1. `task fmt:fix && task check` passes.
2. `task test:all` passes, including under the race detector.
3. New behaviour has tests, and bug fixes have a regression test.
4. User-visible changes are reflected in `README.md` or `config.example.yaml`.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) with a
scope, for example `fix(index): ...` or `feat(api): ...`. Write the subject as the big-picture
change, not a list of files. Add a body only when the reason isn't obvious from the subject.

Keep each PR to one logical change. A small, focused PR gets reviewed much faster than a large
mixed one.

## Reporting bugs

Open an issue with:

- the GoVec version or commit, and how you run it (binary, Docker, source)
- your config (index type, quantization, distance metric)
- the exact request and response, or a minimal reproduction
- what you expected instead

For security issues, please don't open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/Pradyothsp/govec/security/advisories/new)
instead.

## License

By contributing, you agree that your contributions are licensed under the project's
[AGPL-3.0 license](LICENSE).
