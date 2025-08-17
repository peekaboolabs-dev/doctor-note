"""
설정 관리 모듈
"""

import os

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def load_config():
    """
    환경변수에서 설정 로드

    Returns:
        dict: 설정 딕셔너리
    """
    # 환경변수에서 가져오기 (기본값 포함)
    config = {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask"),
        "chroma_persist_dir": os.getenv(
            "CHROMA_PERSIST_DIR", "data/embeddings/chroma_medical_db"
        ),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "collection_name": os.getenv("COLLECTION_NAME", "korean_medical_qa"),
        "batch_size": int(os.getenv("BATCH_SIZE", "500")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        # LLM 설정
        "llm_type": os.getenv("LLM_TYPE", "ollama"),  # "ollama", "llamacpp_server"
        "llm_model": os.getenv("LLM_MODEL", "solar"),  # 모델 이름
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
        "llm_top_p": float(os.getenv("LLM_TOP_P", "0.9")),
        "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
        "llm_streaming": os.getenv("LLM_STREAMING", "true").lower()
        in ["true", "1", "yes"],
        # Ollama 전용
        "ollama_model": os.getenv("OLLAMA_MODEL", "solar"),
        "ollama_host": os.getenv("OLLAMA_HOST", "localhost"),
        "ollama_port": int(os.getenv("OLLAMA_PORT", "11434")),
        # llama.cpp 서버 전용
        "llama_server_host": os.getenv("LLAMA_SERVER_HOST", "localhost"),
        "llama_server_port": int(os.getenv("LLAMA_SERVER_PORT", "8080")),
    }

    return config
