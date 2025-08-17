#!/bin/bash

# llama.cpp 서버 설정 스크립트

echo "======================================"
echo "llama.cpp 설정 스크립트"
echo "======================================"

# 1. llama.cpp 클론 및 빌드
if [ ! -d "llama.cpp" ]; then
    echo "1. llama.cpp 클론 중..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    
    # Mac 감지
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Mac 감지 - Metal 빌드"
        make clean && make LLAMA_METAL=1
    else
        # Linux/WSL
        if command -v nvcc &> /dev/null; then
            echo "CUDA 감지 - CUDA 빌드"
            make clean && make LLAMA_CUDA=1
        else
            echo "CPU 빌드"
            make clean && make
        fi
    fi
    cd ..
else
    echo "llama.cpp 이미 존재"
fi

# 2. Python 바인딩 설치
echo ""
echo "2. Python 바인딩 설치..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Mac용 llama-cpp-python 설치 (Metal)"
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
else
    if command -v nvcc &> /dev/null; then
        echo "CUDA용 llama-cpp-python 설치"
        CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    else
        echo "CPU용 llama-cpp-python 설치"
        pip install llama-cpp-python
    fi
fi

# 3. 모델 다운로드 예시
echo ""
echo "3. 모델 다운로드 안내"
echo "======================================"
echo "GGUF 모델 다운로드 예시:"
echo ""
echo "# Qwen 2.5 7B (한국어 잘함)"
echo "wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf -P models/"
echo ""
echo "# Llama 3.2 3B (작고 빠름)"
echo "wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -P models/"
echo ""

# 4. 서버 실행 스크립트 생성
cat > run_llama_server.sh << 'EOF'
#!/bin/bash

MODEL=${1:-"models/qwen2.5-7b-instruct-q4_k_m.gguf"}
PORT=${2:-8080}

echo "Starting llama.cpp server..."
echo "Model: $MODEL"
echo "Port: $PORT"

./llama.cpp/server \
    -m "$MODEL" \
    -c 4096 \
    --host 0.0.0.0 \
    --port $PORT \
    -ngl 999 \
    --parallel 4 \
    --cont-batching \
    --verbose
EOF

chmod +x run_llama_server.sh

echo ""
echo "======================================"
echo "설정 완료!"
echo "======================================"
echo ""
echo "서버 실행:"
echo "  ./run_llama_server.sh [모델경로] [포트]"
echo ""
echo "Python에서 사용:"
echo "  model_type='llamacpp_server'"
echo "  llama_server_port=8080"