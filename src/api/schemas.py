"""
API 스키마 정의 (타 팀과의 인터페이스 표준)
"""

import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """검색 요청 스키마"""

    query: str = Field(..., description="검색 쿼리", example="고혈압 치료 방법")
    k: int = Field(5, description="반환할 결과 수", ge=1, le=20)
    filter: dict[str, Any] | None = Field(None, description="메타데이터 필터")

    class Config:
        schema_extra = {
            "example": {
                "query": "당뇨병 관리",
                "k": 5,
                "filter": {"category": "doctor"},
            }
        }


class SearchResult(BaseModel):
    """검색 결과 아이템"""

    content: str = Field(..., description="검색된 내용")
    score: float = Field(..., description="유사도 점수")
    metadata: dict[str, Any] = Field(..., description="메타데이터")


class SearchResponse(BaseModel):
    """검색 응답 스키마"""

    success: bool = Field(True, description="요청 성공 여부")
    query: str = Field(..., description="원본 쿼리")
    results: list[SearchResult] = Field(..., description="검색 결과 목록")
    total_results: int = Field(..., description="전체 결과 수")
    processing_time: float = Field(..., description="처리 시간(초)")


class PatientNoteRequest(BaseModel):
    """환자 노트 추가 요청"""

    patient_id: str = Field(..., description="환자 ID", example="P12345")
    note_text: str = Field(..., description="노트 내용")
    metadata: dict[str, Any] | None = Field(
        None,
        description="추가 메타데이터",
        example={"date": "2024-01-15", "doctor": "김의사", "department": "내과"},
    )


class PatientNoteResponse(BaseModel):
    """환자 노트 응답"""

    success: bool = Field(True, description="요청 성공 여부")
    patient_id: str = Field(..., description="환자 ID")
    note_id: str = Field(..., description="생성된 노트 ID")
    created_at: datetime = Field(..., description="생성 시간")


class RAGRequest(BaseModel):
    """RAG 생성 요청"""

    query: str = Field(..., description="사용자 질문")
    context_k: int = Field(5, description="검색할 컨텍스트 수")
    max_tokens: int = Field(500, description="최대 응답 토큰 수")
    temperature: float = Field(
        float(os.getenv("LLM_TEMPERATURE", "0.3")), description="생성 온도", ge=0, le=1
    )
    stream: bool = Field(False, description="스트리밍 응답 여부")


class RAGResponse(BaseModel):
    """RAG 생성 응답"""

    success: bool = Field(True, description="요청 성공 여부")
    query: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="생성된 답변")
    contexts: list[dict[str, Any]] = Field(..., description="사용된 컨텍스트")
    model: str = Field(..., description="사용된 모델")
    processing_time: float = Field(..., description="처리 시간(초)")


class HealthCheckResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field("healthy", description="서비스 상태")
    version: str = Field(..., description="API 버전")
    chroma_connected: bool = Field(..., description="ChromaDB 연결 상태")
    ollama_connected: bool = Field(..., description="Ollama 연결 상태")
    total_documents: int = Field(..., description="총 문서 수")


class ErrorResponse(BaseModel):
    """에러 응답"""

    success: bool = Field(False, description="요청 실패")
    error: str = Field(..., description="에러 메시지")
    error_code: str = Field(..., description="에러 코드")
    detail: dict[str, Any] | None = Field(None, description="상세 정보")
