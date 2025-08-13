"""
llama.cpp 서버 클라이언트 - 가장 효율적인 방법
llama.cpp 서버와 통신 (Ollama와 동일한 아키텍처)
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator

import httpx

from src.utils.logger import get_logger

from .base import BaseLLM, LLMConfig, LLMResponse

logger = get_logger(__name__)


class LlamaCppServerLLM(BaseLLM):
    """llama.cpp 서버 기반 LLM (별도 프로세스, 메모리 효율적)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # llama.cpp 서버 주소
        self.base_url = f"http://{config.llama_server_host}:{config.llama_server_port}"

        # 타임아웃 설정 개선
        timeout = httpx.Timeout(
            connect=5.0,  # 연결 타임아웃
            read=300.0,  # 읽기 타임아웃 (5분)
            write=10.0,  # 쓰기 타임아웃
            pool=5.0,  # 연결 풀 타임아웃
        )

        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

        # 서버 연결 테스트
        self._test_connection()
        logger.info(f"llama.cpp server client initialized: {self.base_url}")

    def _test_connection(self):
        """서버 연결 테스트"""
        try:
            response = self.client.get(f"{self.base_url}/health", timeout=2.0)
            if response.status_code != 200:
                logger.warning(f"Server health check returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Server may not be ready: {e}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """동기 텍스트 생성 (Chat API 사용)"""
        try:
            # 스트리밍 여부 확인
            use_streaming = kwargs.get("streaming", self.streaming)

            if use_streaming:
                # 스트리밍으로 수집 후 전체 반환
                logger.debug("Using streaming mode to collect response")
                collected_text = []
                for chunk in self.stream(prompt, **kwargs):
                    collected_text.append(chunk)
                    if kwargs.get("show_progress", False):
                        print(chunk, end="", flush=True)

                full_text = "".join(collected_text)
                return LLMResponse(
                    text=full_text,
                    model=self.config.model_name,
                    metadata={"backend": "llama.cpp-server", "streamed": True},
                )
            else:
                # Chat API 사용
                system_prompt = (
                    "당신은 의료 전문가입니다. 질문에 정확하고 간결하게 답변하세요."
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                payload = {
                    "model": "gpt-oss",
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.3),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "stream": False,
                }

                logger.debug("Using chat API")
                start_time = time.time()

                # Chat completions 엔드포인트 사용
                response = self.client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload
                )
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Chat API 응답 형식 처리
                content = ""
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                elif "content" in data:
                    content = data.get("content", "")

                logger.debug(f"Response received in {elapsed:.2f}s")

                return LLMResponse(
                    text=content,
                    model=self.config.model_name,
                    usage=data.get("usage"),
                    metadata={"backend": "llama.cpp-server", "elapsed_time": elapsed},
                )

        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {e}")
            raise RuntimeError(
                f"서버 응답 시간 초과. 서버 상태를 확인하세요: {self.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise RuntimeError(f"서버 오류: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 텍스트 생성"""
        try:
            payload = {
                "prompt": prompt,
                "n_predict": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "stop": kwargs.get("stop", ["</s>", "\n\n\n", "User:", "Assistant:"]),
                "stream": True,
                "cache_prompt": True,
            }

            logger.debug("Starting streaming request")

            # SSE (Server-Sent Events) 스트리밍
            with self.client.stream(
                "POST",
                f"{self.base_url}/completion",
                json=payload,
                timeout=httpx.Timeout(None),  # 스트리밍은 타임아웃 없음
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    # SSE 형식 파싱
                    if line.startswith("data: "):
                        json_str = line[6:]  # "data: " 제거

                        if json_str == "[DONE]":
                            break

                        try:
                            data = json.loads(json_str)
                            if "content" in data:
                                content = data["content"]
                                # 특별 토큰이 나타나면 이후 내용만 반환
                                if "<|message|>" in content:
                                    parts = content.split("<|message|>")
                                    if len(parts) > 1:
                                        yield parts[-1]
                                else:
                                    yield content

                            # 종료 조건 체크
                            if data.get("stop", False):
                                break

                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON decode error: {e}, line: {line}")
                            continue

        except httpx.TimeoutException as e:
            logger.error(f"Streaming timeout: {e}")
            raise RuntimeError("스트리밍 타임아웃") from e
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 텍스트 생성"""
        try:
            use_streaming = kwargs.get("streaming", self.streaming)

            if use_streaming:
                # 비동기 스트리밍으로 수집
                collected_text = []
                async for chunk in self.astream(prompt, **kwargs):
                    collected_text.append(chunk)

                return LLMResponse(
                    text="".join(collected_text),
                    model=self.config.model_name,
                    metadata={"backend": "llama.cpp-server", "streamed": True},
                )
            else:
                payload = {
                    "prompt": prompt,
                    "n_predict": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                    "stop": kwargs.get(
                        "stop",
                        [
                            "</s>",
                            "\n\n\n",
                            "User:",
                            "Assistant:",
                            ">>>",
                            "<thinking>",
                            "</thinking>",
                            "We need",
                            "Let's parse",
                            "Let's craft",
                            "Thus we",
                            "<|start|>assistant",
                        ],
                    ),
                    "stream": False,
                    "cache_prompt": True,
                }

                response = await self.async_client.post(
                    f"{self.base_url}/completion", json=payload
                )
                response.raise_for_status()

                data = response.json()

                return LLMResponse(
                    text=data.get("content", ""),
                    model=self.config.model_name,
                    usage=data.get("timings"),
                    metadata={"backend": "llama.cpp-server"},
                )

        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """비동기 스트리밍"""
        try:
            payload = {
                "prompt": prompt,
                "n_predict": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "stop": kwargs.get("stop", ["</s>", "\n\n\n", "User:", "Assistant:"]),
                "stream": True,
                "cache_prompt": True,
            }

            async with self.async_client.stream(
                "POST",
                f"{self.base_url}/completion",
                json=payload,
                timeout=httpx.Timeout(None),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        json_str = line[6:]

                        if json_str == "[DONE]":
                            break

                        try:
                            data = json.loads(json_str)
                            if "content" in data:
                                yield data["content"]

                            if data.get("stop", False):
                                break

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Async streaming failed: {e}")
            raise

    def __del__(self):
        """클라이언트 정리"""
        try:
            if hasattr(self, "client"):
                self.client.close()
            if hasattr(self, "async_client"):
                asyncio.run(self.async_client.aclose())
        except Exception:
            pass
