FROM golang:1.22-alpine AS builder

WORKDIR /app

RUN apk add --no-cache gcc musl-dev

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o trader ./cmd/trader

FROM alpine:3.19

WORKDIR /app

RUN apk add --no-cache ca-certificates wget

RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && mv onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ \
    && rm -rf onnxruntime-linux-x64-1.16.3* \
    && ldconfig /usr/local/lib

COPY --from=builder /app/trader .
COPY configs/ ./configs/
COPY models/ ./models/

EXPOSE 8080

ENV LD_LIBRARY_PATH=/usr/local/lib

ENTRYPOINT ["./trader"]
CMD ["--config=configs/paper.yaml", "--capital=10000"]
