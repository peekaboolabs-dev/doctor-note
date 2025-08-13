#!/bin/bash

MODEL=${1:-"models/gpt-oss-20b-Q4_K_M.gguf"}
PORT=${2:-8080}

echo "Starting llama.cpp server with clean output settings..."
echo "Model: $MODEL"
echo "Port: $PORT"

# 깔끔한 출력을 위한 설정
./llama.cpp/build/bin/llama-server \
    -m "$MODEL" \
    -c 8192 \
    --host 127.0.0.1 \
    --port $PORT \
    -ngl -1 \
    --n-predict -1 \
    --threads 8 \
    --batch-size 512 \
    --parallel 4 \
    --cont-batching \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --log-disable