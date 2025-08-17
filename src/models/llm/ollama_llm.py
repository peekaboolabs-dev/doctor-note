"""
Ollama LLM 래퍼 - 기존 Ollama 호환성 유지
"""

from collections.abc import AsyncIterator, Iterator

from langchain_ollama import OllamaLLM as LangChainOllama

from src.utils.logger import get_logger

from .base import BaseLLM, LLMConfig, LLMResponse

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama 기반 LLM"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = f"http://{config.ollama_host}:{config.ollama_port}"
        self._init_client()

    def _init_client(self):
        """Ollama 클라이언트 초기화"""
        try:
            self.client = LangChainOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                top_p=self.top_p,
                num_predict=self.max_tokens,
            )
            logger.info(f"Ollama client initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """동기 텍스트 생성"""
        try:
            # 파라미터 업데이트
            if kwargs:
                self.client.temperature = kwargs.get("temperature", self.temperature)
                self.client.top_p = kwargs.get("top_p", self.top_p)
                self.client.num_predict = kwargs.get("max_tokens", self.max_tokens)

            # 생성
            response = self.client.invoke(prompt)

            return LLMResponse(
                text=response,
                model=self.model_name,
                metadata={"base_url": self.base_url},
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 텍스트 생성"""
        try:
            # 스트리밍 클라이언트 생성
            streaming_client = LangChainOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                num_predict=kwargs.get("max_tokens", self.max_tokens),
                streaming=True,
            )

            # 스트리밍 생성
            yield from streaming_client.stream(prompt)

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 텍스트 생성"""
        try:
            # 파라미터 업데이트
            if kwargs:
                self.client.temperature = kwargs.get("temperature", self.temperature)
                self.client.top_p = kwargs.get("top_p", self.top_p)
                self.client.num_predict = kwargs.get("max_tokens", self.max_tokens)

            # 비동기 생성
            response = await self.client.ainvoke(prompt)

            return LLMResponse(
                text=response,
                model=self.model_name,
                metadata={"base_url": self.base_url},
            )
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """비동기 스트리밍 텍스트 생성"""
        try:
            # 스트리밍 클라이언트 생성
            streaming_client = LangChainOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                num_predict=kwargs.get("max_tokens", self.max_tokens),
                streaming=True,
            )

            # 비동기 스트리밍 생성
            async for chunk in streaming_client.astream(prompt):
                yield chunk

        except Exception as e:
            logger.error(f"Async streaming failed: {e}")
            raise
