"""
설정 관리 모듈
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def load_config(config_path=None):
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로 (기본값: configs/config.json)
    
    Returns:
        dict: 설정 딕셔너리
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.json"
    
    # 기본 설정
    default_config = {
        "embedding_model": "jhgan/ko-sroberta-multitask",
        "chroma_persist_dir": "data/embeddings/chroma_medical_db",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "collection_name": "korean_medical_qa",
        "batch_size": 500,
        "log_level": "INFO",
        "ollama_model": "solar",
        "ollama_host": "localhost",
        "ollama_port": 11434
    }
    
    # 설정 파일이 있으면 로드
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                default_config.update(file_config)
        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
    
    # 환경 변수 오버라이드
    env_mapping = {
        "EMBEDDING_MODEL": "embedding_model",
        "CHROMA_PERSIST_DIR": "chroma_persist_dir",
        "LOG_LEVEL": "log_level",
        "OLLAMA_MODEL": "ollama_model",
        "OLLAMA_HOST": "ollama_host",
        "OLLAMA_PORT": "ollama_port"
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            default_config[config_key] = os.environ[env_var]
    
    return default_config


def save_config(config, config_path=None):
    """
    설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        config_path: 저장할 경로
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.json"
    
    # 디렉토리 생성
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)