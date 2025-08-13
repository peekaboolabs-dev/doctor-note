"""
llama.cpp 서버 클라이언트 - 안정성 개선 버전
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Iterator

import httpx

from src.utils.logger import get_logger

from .base import BaseLLM, LLMConfig, LLMResponse

logger = get_logger(__name__)


class LlamaCppServerLLM(BaseLLM):
    """llama.cpp 서버 기반 LLM (안정성 개선)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = f"http://{config.llama_server_host}:{config.llama_server_port}"

        timeout = httpx.Timeout(300.0)
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

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

    def _validate_json_response(self, text: str, expected_keys: list = None) -> bool:
        """JSON 응답 유효성 검증"""
        try:
            data = json.loads(text)
            if expected_keys:
                # 예상 키들이 있는지 확인
                for key in expected_keys:
                    if key not in data:
                        return False
            # 템플릿 값인지 확인
            if "값" in str(data) or "증상1" in str(data):
                return False  # 템플릿 그대로면 무효
            return True
        except Exception:
            return False

    def _extract_json_from_text(self, text: str) -> str:
        """텍스트에서 JSON 추출 (개선된 버전)"""
        # JSON 패턴 찾기
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            # 뒤에서부터 검사 (보통 최종 답변)
            for match in reversed(matches):
                # 템플릿이 아닌 실제 JSON인지 확인
                if self._validate_json_response(match, ["symptoms"]):
                    return match

        # 유효한 JSON을 못 찾은 경우
        logger.warning("No valid JSON found in response")
        return "{}"

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """동기 텍스트 생성 (재시도 로직 포함)"""
        max_retries = kwargs.get("max_retries", 2)

        for attempt in range(max_retries):
            try:
                # JSON 추출 모드
                if kwargs.get("extract_json", False):
                    # 더 명확한 프롬프트
                    modified_prompt = f"""{prompt}

CRITICAL: Output ONLY the JSON with actual values, not template placeholders.
Replace "값", "증상1", etc. with real data from the dialogue.
Start with {{ and end with }}.

JSON:"""

                    payload = {
                        "prompt": modified_prompt,
                        "temperature": 0.2,  # 더 결정적으로
                        "top_p": 0.5,
                        "n_predict": 800,
                        "stop": ["</s>", "<|im_end|>", "<|eot_id|>", "\n\n\n"],
                        "stream": False,
                        "seed": attempt * 42,  # 다른 시드로 재시도
                        "cache_prompt": False,  # 캐시 사용 안함
                    }
                else:
                    # 일반 텍스트 생성
                    payload = {
                        "prompt": prompt,
                        "temperature": kwargs.get(
                            "temperature", self.temperature or 0.3
                        ),
                        "top_p": kwargs.get("top_p", self.top_p or 0.9),
                        "n_predict": kwargs.get("max_tokens", self.max_tokens or 2048),
                        "stop": ["</s>", "<|im_end|>", "<|eot_id|>"],
                        "stream": False,
                        "cache_prompt": True,
                    }

                response = self.client.post(f"{self.base_url}/completion", json=payload)
                response.raise_for_status()

                data = response.json()
                content = data.get("content", "")

                # JSON 모드에서는 JSON 추출 및 검증
                if kwargs.get("extract_json", False):
                    json_text = self._extract_json_from_text(content)
                    # 유효성 검증
                    if self._validate_json_response(json_text, ["symptoms"]):
                        return LLMResponse(
                            text=json_text,
                            model=self.config.model_name,
                            metadata={
                                "backend": "llama.cpp-server",
                                "attempt": attempt + 1,
                            },
                        )
                    else:
                        logger.warning(
                            f"Invalid JSON on attempt {attempt + 1}, retrying..."
                        )
                        time.sleep(0.5)  # 짧은 대기
                        continue
                else:
                    return LLMResponse(
                        text=content,
                        model=self.config.model_name,
                        metadata={"backend": "llama.cpp-server"},
                    )

            except Exception as e:
                logger.error(f"Generation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5)

        # 모든 재시도 실패 시 빈 JSON 반환
        if kwargs.get("extract_json", False):
            return LLMResponse(
                text="{}",
                model=self.config.model_name,
                metadata={"backend": "llama.cpp-server", "failed": True},
            )
        else:
            raise Exception("All generation attempts failed")

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 텍스트 생성"""
        try:
            # JSON 추출 모드는 generate 사용 (안정성)
            if kwargs.get("extract_json", False):
                response = self.generate(prompt, **kwargs)
                # 전체를 한 번에 yield
                yield response.text
                return

            # 일반 스트리밍
            payload = {
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature or 0.3),
                "top_p": kwargs.get("top_p", self.top_p or 0.9),
                "n_predict": kwargs.get("max_tokens", self.max_tokens or 2048),
                "stop": ["</s>", "<|im_end|>", "<|eot_id|>"],
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

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break

                    try:
                        data = json.loads(json_str)
                        content = data.get("content", "")
                        if content:
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
        # 동기 버전과 동일한 로직
        return self.generate(prompt, **kwargs)

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """비동기 스트리밍"""
        # JSON 모드는 agenerate 사용
        if kwargs.get("extract_json", False):
            response = await self.agenerate(prompt, **kwargs)
            yield response.text
            return

        try:
            payload = {
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature or 0.3),
                "top_p": kwargs.get("top_p", self.top_p or 0.9),
                "n_predict": kwargs.get("max_tokens", self.max_tokens or 2048),
                "stop": ["</s>", "<|im_end|>", "<|eot_id|>"],
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
                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break

                    try:
                        data = json.loads(json_str)
                        content = data.get("content", "")
                        if content:
                            yield content

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
