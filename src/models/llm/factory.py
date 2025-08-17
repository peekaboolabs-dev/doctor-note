"""
LLM Factory - 설정에 따라 적절한 LLM 인스턴스 생성
"""

from src.utils.logger import get_logger

from .base import BaseLLM, LLMConfig
from .llamacpp_server import LlamaCppServerLLM
from .ollama_llm import OllamaLLM

logger = get_logger(__name__)


class LLMFactory:
    """LLM 팩토리 클래스"""

    @staticmethod
    def create(config: LLMConfig) -> BaseLLM:
        """설정에 따라 LLM 인스턴스 생성"""

        model_type = config.model_type.lower()

        if model_type == "ollama":
            logger.info(f"Creating Ollama LLM with model: {config.model_name}")
            return OllamaLLM(config)

        elif model_type == "llamacpp_server":
            logger.info(f"Creating llama.cpp server client for: {config.model_name}")
            return LlamaCppServerLLM(config)

        else:
            raise ValueError(
                f"Unknown model type: {config.model_type}. Supported: ollama, llamacpp_server"
            )

    @staticmethod
    def from_dict(config_dict: dict) -> BaseLLM:
        """딕셔너리로부터 LLM 생성"""
        config = LLMConfig(**config_dict)
        return LLMFactory.create(config)

    @staticmethod
    def create_ollama(
        model_name: str = "solar", host: str = "localhost", port: int = 11434, **kwargs
    ) -> BaseLLM:
        """Ollama LLM 빠른 생성"""
        config = LLMConfig(
            model_type="ollama",
            model_name=model_name,
            ollama_host=host,
            ollama_port=port,
            **kwargs,
        )
        return OllamaLLM(config)

    @staticmethod
    def create_llamacpp_server(
        model_name: str = "model", host: str = "localhost", port: int = 8080, **kwargs
    ) -> BaseLLM:
        """llama.cpp 서버 클라이언트 빠른 생성"""
        config = LLMConfig(
            model_type="llamacpp_server",
            model_name=model_name,
            llama_server_host=host,
            llama_server_port=port,
            **kwargs,
        )
        return LlamaCppServerLLM(config)
