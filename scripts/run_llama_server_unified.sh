#!/bin/bash

# 모델과 포트 설정
MODEL=${1:-"models/gpt-oss-20b-Q4_K_M.gguf"}
PORT=${2:-8080}

# .env 파일 로드 (있을 경우)
ENV_FILE=".env.llamacpp.gpt-oss-20b"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
fi

# 환경변수에서 값 가져오기 (Ollama 기본값과 동일하게)
TEMPERATURE=${LLM_TEMPERATURE:-1.0}
TOP_P=${LLM_TOP_P:-0.95}
TOP_K=${LLM_TOP_K:-40}
MAX_TOKENS=${LLM_MAX_TOKENS:-2048}
REPEAT_PENALTY=${LLM_REPEAT_PENALTY:-1.1}

echo "Starting llama.cpp server with unified settings..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Max tokens: $MAX_TOKENS"
echo "Repeat penalty: $REPEAT_PENALTY"

# gpt-oss 모델용 템플릿 설정
# Ollama의 템플릿 형식을 참고하여 적용
CHAT_TEMPLATE='<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: $(date +%Y-%m-%d)

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
<|start|>user<|message|>{{prompt}}<|end|>
<|start|>assistant'

# llama.cpp 서버 실행
./llama.cpp/build/bin/llama-server \
    -m "$MODEL" \
    -c 8192 \
    --host 127.0.0.1 \
    --port $PORT \
    -ngl -1 \
    --n-predict $MAX_TOKENS \
    --threads 8 \
    --batch-size 512 \
    --parallel 4 \
    --cont-batching \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --log-disable \
    --reasoning-format none \
    --temp $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --repeat-penalty $REPEAT_PENALTY \
    --no-mmap false