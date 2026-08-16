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

WORKDIR /app

COPY --from=builder /app/govec .
COPY --from=builder /app/config.yaml .

EXPOSE 8000

# nobody must own /app to write data_path/wal_path files there at runtime
RUN chown -R nobody:nobody /app

# Run as non-root user
USER nobody:nobody

CMD ["./govec"]
