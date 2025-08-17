#!/bin/bash

# gpt-oss:20b 모델 전용 llama.cpp 서버 스크립트

# 모델과 포트 설정
MODEL=${1:-"models/gpt-oss-20b-Q4_K_M.gguf"}
PORT=${2:-8080}

# .env 파일 로드 (있을 경우)
ENV_FILE=".env.llamacpp.gpt-oss-20b"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    source "$ENV_FILE"
fi

# 환경변수에서 값 가져오기 (Ollama gpt-oss:20b와 동일)
TEMPERATURE=${LLM_TEMPERATURE:-1.0}
TOP_P=${LLM_TOP_P:-0.95}
TOP_K=${LLM_TOP_K:-40}
MAX_TOKENS=${LLM_MAX_TOKENS:-2048}
REPEAT_PENALTY=${LLM_REPEAT_PENALTY:-1.1}
CONTEXT_SIZE=${LLM_CONTEXT_SIZE:-8192}

echo "========================================"
echo "Starting llama.cpp server for gpt-oss:20b"
echo "========================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Max tokens: $MAX_TOKENS"
echo "Repeat penalty: $REPEAT_PENALTY"
echo "Context size: $CONTEXT_SIZE"
echo "========================================"

# llama.cpp 서버 실행 (gpt-oss 템플릿 적용)
./llama.cpp/build/bin/llama-server \
    -m "$MODEL" \
    -c $CONTEXT_SIZE \
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
    --temp $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --repeat-penalty $REPEAT_PENALTY \
    --chat-template "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: $(date +%Y-%m-%d)\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\n{{#each messages}}{{#ifUser}}<|start|>user<|message|>{{content}}<|end|>{{/ifUser}}{{#ifAssistant}}<|start|>assistant<|channel|>final<|message|>{{content}}<|end|>{{/ifAssistant}}{{/each}}<|start|>assistant" \
    --samplers top_k,top_p,temp \
    --seed -1