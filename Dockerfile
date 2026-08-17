# Build stage
FROM golang:1.26-alpine AS builder

WORKDIR /app

# Download dependencies first (cached layer)
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -ldflags="-s -w" -o govec cmd/server/main.go

# Runtime stage
FROM alpine:3.21

RUN apk --no-cache add ca-certificates tzdata

# nobody must own /app to write data_path/wal_path files there at runtime.
# chown here, before copying the binary in: chowning an empty directory is
# free, whereas `RUN chown -R /app` *after* COPY duplicates the already-copied
# binary into this layer (each layer is immutable, so touching an existing
# file's metadata re-stores the whole file) -- doubled the image size for no
# reason. COPY --chown sets ownership at copy time instead, no extra layer.
RUN mkdir -p /app && chown nobody:nobody /app
WORKDIR /app

COPY --chown=nobody:nobody --from=builder /app/govec .
COPY --chown=nobody:nobody --from=builder /app/config.yaml .

EXPOSE 8000

# Run as non-root user
USER nobody:nobody

CMD ["./govec"]
