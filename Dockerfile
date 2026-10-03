# Build stage
#
# Runs on the build machine's own architecture ($BUILDPLATFORM) and cross-compiles for the
# target one. CGO is off, so Go cross-compiles natively -- far faster than building the arm64
# image under QEMU emulation, which is what a plain `FROM golang` would do on an amd64 runner.
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS builder

ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev

WORKDIR /src

# Download dependencies first (cached layer)
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w -X github.com/Pradyothsp/govec/internal/release.Version=${VERSION}" -o govec ./cmd/server

# The repo's config.yaml logs at debug for local development; an image should not. Fail the
# build if the line ever changes shape, rather than silently shipping debug logging.
RUN sed -i 's/^  log_level: "debug"/  log_level: "info"/' config.yaml \
    && grep -q '^  log_level: "info"' config.yaml

# Runtime stage
FROM alpine:3.21

ARG VERSION=dev

LABEL org.opencontainers.image.title="GoVec" \
      org.opencontainers.image.description="Compact vector search engine: HNSW, int8 quantization, hybrid search, REST + gRPC" \
      org.opencontainers.image.source="https://github.com/Pradyothsp/govec" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"

RUN apk --no-cache add ca-certificates tzdata

# nobody must own /data to write the snapshot, WAL and mmap files there at runtime.
# chown here, before copying anything in: chowning an empty directory is free, whereas
# `RUN chown -R` *after* COPY duplicates the already-copied files into a new layer (each
# layer is immutable, so touching an existing file's metadata re-stores the whole file).
# COPY --chown sets ownership at copy time instead, no extra layer.
RUN mkdir -p /app /data && chown nobody:nobody /data

COPY --chown=nobody:nobody --from=builder /src/govec /app/govec
COPY --chown=nobody:nobody --from=builder /src/config.yaml /app/config.yaml

# Storage paths in config.yaml are relative (./govec_data.bin, ./govec.wal, ./govec_mmap/),
# so running from /data puts all state on the volume. Deliberately not done with
# GOVEC_DATA_PATH-style env vars: those override the config file, so a user mounting their
# own config.yaml would find its storage paths silently ignored.
WORKDIR /data
ENV GOVEC_CONFIG_PATH=/app/config.yaml
VOLUME /data

EXPOSE 8000 50051

# Run as non-root user
USER nobody:nobody

# Probes the default port; if you move the HTTP port with GOVEC_SERVER_PORT, set it here too.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- "http://localhost:${GOVEC_SERVER_PORT:-8000}/health" >/dev/null || exit 1

CMD ["/app/govec"]
