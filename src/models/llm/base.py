"""
LLM 베이스 클래스 및 공통 타입 정의
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 설정 클래스"""

    model_type: str  # "ollama", "llamacpp", "llamacpp_server"
    model_name: str  # 모델 이름
    temperature: float | None = None  # None이면 .env에서 가져옴
    top_p: float | None = None  # None이면 .env에서 가져옴
    max_tokens: int | None = None  # None이면 .env에서 가져옴
    streaming: bool = True

    # Ollama 전용
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    # llama.cpp 서버 전용
    llama_server_host: str = "localhost"
    llama_server_port: int = 8080

    # llama.cpp 로컬 전용 (향후 필요시)
    model_path: str | None = None
    n_gpu_layers: int = -1  # -1 = 모든 레이어 GPU
    n_ctx: int = 4096


@dataclass
class LLMResponse:
    """LLM 응답 클래스"""

    text: str
    model: str
    usage: dict | None = None
    metadata: dict | None = None


class BaseLLM(ABC):
    """LLM 베이스 추상 클래스"""

    def __init__(self, config: LLMConfig):
        from src.utils.config import load_config

        self.config = config
        self.model_name = config.model_name

        # .env 설정 로드
        env_config = load_config()

        # temperature, top_p, max_tokens가 None이면 .env에서 가져오기
        self.temperature = (
            config.temperature
            if config.temperature is not None
            else env_config["llm_temperature"]
        )
        self.top_p = (
            config.top_p if config.top_p is not None else env_config["llm_top_p"]
        )
        self.max_tokens = (
            config.max_tokens
            if config.max_tokens is not None
            else env_config["llm_max_tokens"]
        )
        self.streaming = config.streaming

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """동기 텍스트 생성"""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 텍스트 생성"""
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 텍스트 생성"""
        pass

    @abstractmethod
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """비동기 스트리밍 텍스트 생성"""
        pass

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "type": self.config.model_type,
            "name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def update_params(self, **kwargs):
        """파라미터 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
