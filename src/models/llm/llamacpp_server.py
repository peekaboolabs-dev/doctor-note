"""
llama.cpp 서버 클라이언트 - v1 기반 개선
"""

import asyncio
import json
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
        """동기 텍스트 생성 - v1 기반"""
        try:
            # 기본 stop 토큰
            stop_tokens = ["<|eot_id|>", "<|im_end|>", "</s>", "[END_NOTE]", "\n\n\n"]

            # JSON 추출 모드
            if kwargs.get("extract_json", False) or "JSON" in prompt:
                modified_prompt = f"""{prompt}

Output ONLY the JSON object. No explanation. Start directly with {{ and end with }}.
JSON:"""

                payload = {
                    "prompt": modified_prompt,
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "n_predict": 800,
                    "stop": stop_tokens,
                    "repeat_penalty": 1.2,
                    "cache_prompt": True,
                }
            else:
                payload = {
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", 0.3),
                    "top_p": kwargs.get("top_p", 0.9),
                    "n_predict": kwargs.get("max_tokens", self.max_tokens),
                    "stop": stop_tokens,
                    "repeat_penalty": 1.2,
                    "cache_prompt": True,
                }

            response = self.client.post(
                f"{self.base_url}/completion", json=payload, timeout=60.0
            )
            response.raise_for_status()

            data = response.json()
            raw_content = data.get("content", "")

            # CoT 제거 - 간단한 버전
            cleaned_content = self._clean_cot_output(raw_content)

            return LLMResponse(
                text=cleaned_content,
                model=self.config.model_name,
                metadata={"backend": "llama.cpp-server"},
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _clean_cot_output(self, text: str) -> str:
        """CoT 추론 과정 제거 - 간단한 버전"""
        import re

        # [BEGIN_NOTE] 블록 추출
        if "[BEGIN_NOTE]" in text:
            start = text.find("[BEGIN_NOTE]")
            end = text.find("[END_NOTE]")
            if end != -1:
                return text[start : end + len("[END_NOTE]")]
            else:
                return text[start:]

        # JSON 블록 추출
        json_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            # 가장 완전한 JSON 선택
            return max(matches, key=len)

        # >>> 마커 이후 내용
        if ">>>" in text:
            parts = text.split(">>>")
            if len(parts) > 1:
                return parts[-1].strip()

        return text

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 - v1 기반 개선"""
        try:
            stop_tokens = ["<|eot_id|>", "<|im_end|>", "</s>", "[END_NOTE]"]

            # JSON 추출 모드
            if kwargs.get("extract_json", False):
                modified_prompt = f"{prompt}\n\nOutput only JSON:"
                payload = {
                    "prompt": modified_prompt,
                    "n_predict": kwargs.get("max_tokens", 800),
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "repeat_penalty": 1.2,
                    "stop": stop_tokens,
                    "stream": True,
                    "cache_prompt": True,
                }
            else:
                payload = {
                    "prompt": prompt,
                    "n_predict": kwargs.get("max_tokens", 800),
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "stop": stop_tokens,
                    "stream": True,
                    "cache_prompt": True,
                }

            with self.client.stream(
                "POST",
                f"{self.base_url}/completion",
                json=payload,
                timeout=httpx.Timeout(None),
            ) as response:
                response.raise_for_status()

                json_mode = kwargs.get("extract_json", False)
                brace_count = 0
                skip_until_json = json_mode  # JSON 모드에서는 처음부터 스킵

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break

                    try:
                        data = json.loads(json_str)
                        content = data.get("content", "")

                        if json_mode:
                            # JSON 모드: { 찾을 때까지 스킵
                            if skip_until_json:
                                if "{" in content:
                                    skip_until_json = False
                                    idx = content.index("{")
                                    content = content[idx:]  # { 이전 제거
                                    yield content
                                    brace_count += content.count("{") - content.count(
                                        "}"
                                    )
                                else:
                                    continue  # { 전까지는 출력하지 않음
                            else:
                                yield content
                                brace_count += content.count("{") - content.count("}")
                                if brace_count == 0:
                                    break
                        else:
                            # 일반 모드: 모든 내용 출력
                            yield content

                        if data.get("stop", False):
                            break

                    except json.JSONDecodeError:
                        continue

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
                stop_tokens = [
                    "<|eot_id|>",
                    "<|im_end|>",
                    "</s>",
                    "\n\n\n",
                    "[END_NOTE]",
                ]

                payload = {
                    "prompt": prompt,
                    "n_predict": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.2),
                    "stop": stop_tokens,
                    "stream": False,
                    "cache_prompt": True,
                }

                response = await self.async_client.post(
                    f"{self.base_url}/completion", json=payload
                )
                response.raise_for_status()

                data = response.json()
                raw_content = data.get("content", "")

                # CoT 제거
                cleaned_content = self._clean_cot_output(raw_content)

                return LLMResponse(
                    text=cleaned_content,
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
            stop_tokens = ["<|eot_id|>", "<|im_end|>", "</s>", "\n\n\n", "[END_NOTE]"]

            payload = {
                "prompt": prompt,
                "n_predict": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.2),
                "stop": stop_tokens,
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
