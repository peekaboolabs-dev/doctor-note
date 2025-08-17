"""
하이브리드 검색 RAG 시스템 구현
BM25(키워드) + Dense(시맨틱) 검색을 결합한 의학 도메인 특화 RAG
"""

import json
import os
import re
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# from konlpy.tag import Mecab  # 시스템 의존성 때문에 옵션으로 처리
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MedicalTokenizer:
    """의학 도메인 특화 토크나이저"""

    def __init__(self):
        """
        Mecab 기반 한국어 토크나이저
        의학 용어 사전과 약어 보존 기능 포함
        """
        try:
            from konlpy.tag import Mecab

            self.mecab = Mecab()
        except Exception:
            logger.warning("Mecab을 사용할 수 없습니다. 기본 토크나이저 사용")
            self.mecab = None

        # 의학 약어 사전
        self.medical_abbr = {
            "BP",
            "HR",
            "ECG",
            "MRI",
            "CT",
            "WBC",
            "RBC",
            "HGB",
            "PLT",
            "AST",
            "ALT",
            "BUN",
            "Cr",
            "CRP",
            "ESR",
            "PT",
            "aPTT",
            "INR",
            "Na",
            "K",
            "Cl",
            "CO2",
            "Ca",
            "Mg",
            "P",
            "Alb",
            "T-bil",
        }

        # 의학 용어 보존 패턴 (정규식)
        self.medical_patterns = [
            r"\d+\.?\d*\s?(mg|g|ml|L|mcg|IU|mEq|mmol|μg)",  # 용량 단위
            r"[A-Z]{2,}",  # 대문자 약어
            r"q\.?d\.?",
            r"b\.?i\.?d\.?",
            r"t\.?i\.?d\.?",
            r"q\.?i\.?d\.?",  # 투약 빈도
            r"p\.?o\.?",
            r"i\.?v\.?",
            r"i\.?m\.?",
            r"s\.?c\.?",  # 투약 경로
        ]

        # 의학 용어 사전 로드
        self.medical_terms = self._load_medical_dictionary()

    def _load_medical_dictionary(self) -> set:
        """의학 용어 사전 로드"""
        terms = set()
        dict_path = "data/medical_terms.json"

        if os.path.exists(dict_path):
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
                terms.update(data.get("terms", []))

        # 기본 의학 용어 추가
        default_terms = {
            "두통",
            "발열",
            "기침",
            "호흡곤란",
            "흉통",
            "복통",
            "구토",
            "설사",
            "아스피린",
            "타이레놀",
            "부루펜",
            "페니실린",
            "아목시실린",
            "당뇨",
            "고혈압",
            "천식",
            "폐렴",
            "위염",
            "간염",
        }
        terms.update(default_terms)

        return terms

    def tokenize(self, text: str) -> list[str]:
        """
        의학 텍스트 토큰화

        Args:
            text: 토큰화할 텍스트

        Returns:
            토큰 리스트
        """
        tokens = []

        # 1. 의학 패턴 보존하며 토큰화
        preserved_tokens = []
        remaining_text = text

        for pattern in self.medical_patterns:
            matches = re.finditer(pattern, remaining_text, re.IGNORECASE)
            for match in matches:
                preserved_tokens.append(match.group())

        # 2. Mecab으로 기본 토큰화
        if self.mecab:
            # Mecab 형태소 분석
            morphs = self.mecab.morphs(text)
            tokens.extend(morphs)
        else:
            # 기본 공백 분리
            tokens.extend(text.split())

        # 3. 의학 약어 보존
        final_tokens = []
        for token in tokens:
            if token.upper() in self.medical_abbr:
                final_tokens.append(token.upper())
            elif token in self.medical_terms:
                final_tokens.append(token)
            else:
                final_tokens.append(token.lower())

        # 4. 보존된 패턴 토큰 추가
        final_tokens.extend(preserved_tokens)

        return final_tokens


class HybridMedicalRAG:
    """
    하이브리드 의학 RAG 시스템
    BM25 키워드 검색 + Dense 벡터 검색 결합
    """

    def __init__(
        self,
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        chroma_persist_dir: str = "data/embeddings/chroma_medical_db",
        collection_name: str = "korean_medical_qa",
        alpha: float = 0.5,
    ):
        """
        Args:
            embedding_model: 임베딩 모델명
            chroma_persist_dir: ChromaDB 저장 경로
            collection_name: ChromaDB 컬렉션명
            alpha: BM25 가중치 (0~1, 높을수록 BM25 중시)
        """
        self.alpha = alpha
        self.collection_name = collection_name

        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self.chroma_store = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_persist_dir,
        )

        # 의학 토크나이저
        self.tokenizer = MedicalTokenizer()

        # BM25 인덱스
        self.documents = []
        self.document_ids = []
        self.bm25 = None
        self._initialize_bm25()

        # ICD-10 코드 매핑
        self.icd10_map = self._load_icd10_codes()

    def _initialize_bm25(self):
        """BM25 인덱스 초기화"""
        try:
            # ChromaDB에서 모든 문서 가져오기
            collection = self.chroma_client.get_collection(self.collection_name)

            # 문서 가져오기 (페이지네이션 처리)
            limit = 1000
            offset = 0
            all_docs = []

            while True:
                result = collection.get(
                    limit=limit, offset=offset, include=["documents", "metadatas"]
                )

                if not result["documents"]:
                    break

                all_docs.extend(result["documents"])
                self.document_ids.extend(result["ids"])
                offset += limit

                if len(result["documents"]) < limit:
                    break

            self.documents = all_docs

            # BM25 인덱스 생성
            if self.documents:
                tokenized_docs = [
                    self.tokenizer.tokenize(doc) for doc in self.documents
                ]
                self.bm25 = BM25Okapi(tokenized_docs)
                logger.info(f"BM25 인덱스 생성 완료: {len(self.documents)}개 문서")
            else:
                logger.warning("BM25 인덱스 생성할 문서가 없습니다")

        except Exception as e:
            logger.error(f"BM25 초기화 실패: {e}")
            self.documents = []
            self.bm25 = None

    def _load_icd10_codes(self) -> dict[str, str]:
        """ICD-10 질병 코드 매핑 로드"""
        icd10_map = {}
        icd10_path = "data/icd10_korean.json"

        if os.path.exists(icd10_path):
            with open(icd10_path, encoding="utf-8") as f:
                icd10_map = json.load(f)
        else:
            # 기본 ICD-10 코드 (예시)
            icd10_map = {
                "J00": "급성 비인두염(감기)",
                "J11": "인플루엔자",
                "J18": "폐렴",
                "K29": "위염 및 십이지장염",
                "E11": "제2형 당뇨병",
                "I10": "본태성 고혈압",
                "J45": "천식",
                "M79.3": "근육통",
            }

        return icd10_map

    def hybrid_search(
        self, query: str, top_k: int = 10, filter_dict: dict | None = None
    ) -> list[dict[str, Any]]:
        """
        하이브리드 검색 실행

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            filter_dict: 메타데이터 필터

        Returns:
            검색 결과 리스트
        """
        # 1. BM25 키워드 검색
        bm25_results = self._bm25_search(query, top_k * 2)

        # 2. Dense 벡터 검색
        dense_results = self._dense_search(query, top_k * 2, filter_dict)

        # 3. 점수 정규화
        bm25_normalized = self._normalize_scores(bm25_results)
        dense_normalized = self._normalize_scores(dense_results)

        # 4. 가중치 결합
        combined_scores = self._combine_scores(
            bm25_normalized, dense_normalized, self.alpha
        )

        # 5. 의학 도메인 리랭킹
        reranked_results = self._rerank_medical(combined_scores, query, top_k)

        return reranked_results

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        BM25 키워드 기반 검색

        Returns:
            [(doc_id, score), ...]
        """
        if not self.bm25 or not self.documents:
            logger.warning("BM25 인덱스가 없습니다")
            return []

        tokenized_query = self.tokenizer.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 K개 선택
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.document_ids):
                doc_id = self.document_ids[idx]
                score = scores[idx]
                results.append((doc_id, float(score)))

        return results

    def _dense_search(
        self, query: str, top_k: int, filter_dict: dict | None = None
    ) -> list[tuple[str, float]]:
        """
        Dense 벡터 유사도 검색

        Returns:
            [(doc_id, score), ...]
        """
        if filter_dict:
            results = self.chroma_store.similarity_search_with_score(
                query, k=top_k, filter=filter_dict
            )
        else:
            results = self.chroma_store.similarity_search_with_score(query, k=top_k)

        # (doc_id, similarity_score) 형태로 변환
        doc_scores = []
        for doc, score in results:
            # ChromaDB의 거리를 유사도로 변환
            similarity = 1 / (1 + score)  # 거리가 작을수록 유사도 높음
            doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
            doc_scores.append((doc_id, similarity))

        return doc_scores

    def _normalize_scores(self, scores: list[tuple[str, float]]) -> dict[str, float]:
        """점수를 0-1 범위로 정규화"""
        if not scores:
            return {}

        values = [score for _, score in scores]
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {doc_id: 0.5 for doc_id, _ in scores}

        normalized = {}
        for doc_id, score in scores:
            normalized[doc_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def _combine_scores(
        self,
        bm25_scores: dict[str, float],
        dense_scores: dict[str, float],
        alpha: float,
    ) -> dict[str, float]:
        """
        두 검색 결과를 가중치 결합

        Args:
            alpha: BM25 가중치 (0~1)
        """
        combined = {}
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0)
            dense_score = dense_scores.get(doc_id, 0)

            # 가중 평균
            combined[doc_id] = alpha * bm25_score + (1 - alpha) * dense_score

        return combined

    def _rerank_medical(
        self, scores: dict[str, float], query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """
        의학 도메인 특화 리랭킹

        Returns:
            최종 검색 결과
        """
        # 의학적 관련성 보너스 점수 계산
        medical_relevance = self._calculate_medical_relevance(query, scores)

        # 최종 점수로 정렬
        final_scores = {}
        for doc_id, base_score in scores.items():
            bonus = medical_relevance.get(doc_id, 0)
            final_scores[doc_id] = base_score + bonus

        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # 문서 내용과 메타데이터 포함하여 반환
        results = []
        for doc_id, score in sorted_docs:
            # 문서 내용 가져오기
            doc_content = self._get_document_content(doc_id)

            results.append(
                {
                    "doc_id": doc_id,
                    "content": doc_content,
                    "score": score,
                    "retrieval_info": {
                        "bm25_score": scores.get(doc_id, 0),
                        "medical_bonus": medical_relevance.get(doc_id, 0),
                        "method": self._get_retrieval_method(doc_id, scores),
                    },
                }
            )

        return results

    def _calculate_medical_relevance(
        self, query: str, scores: dict[str, float]
    ) -> dict[str, float]:
        """
        의학적 관련성 추가 점수 계산

        Returns:
            문서별 보너스 점수
        """
        relevance_scores = {}

        # 쿼리에서 의학 키워드 추출
        medical_keywords = self._extract_medical_keywords(query)

        for doc_id in scores:
            bonus = 0.0
            doc_content = self._get_document_content(doc_id)

            if not doc_content:
                continue

            # ICD-10 코드 매칭
            if self._has_icd10_match(query, doc_content):
                bonus += 0.1

            # 증상-질병 매칭
            if self._has_symptom_disease_match(medical_keywords, doc_content):
                bonus += 0.15

            # 약물 정보 정확도
            if self._has_drug_info_match(query, doc_content):
                bonus += 0.2

            # 용량 정보 포함 여부
            if self._has_dosage_info(doc_content):
                bonus += 0.05

            relevance_scores[doc_id] = bonus

        return relevance_scores

    def _extract_medical_keywords(self, text: str) -> list[str]:
        """의학 키워드 추출"""
        keywords = []

        # 토큰화
        tokens = self.tokenizer.tokenize(text)

        # 의학 용어 필터링
        for token in tokens:
            if token in self.tokenizer.medical_terms:
                keywords.append(token)
            elif token.upper() in self.tokenizer.medical_abbr:
                keywords.append(token.upper())

        # 용량 패턴 추출
        dosage_pattern = r"\d+\.?\d*\s?(mg|g|ml|L|mcg|IU)"
        dosage_matches = re.findall(dosage_pattern, text, re.IGNORECASE)
        keywords.extend(dosage_matches)

        return keywords

    def _has_icd10_match(self, query: str, document: str) -> bool:
        """ICD-10 코드 매칭 확인"""
        # ICD-10 패턴: 대문자 1개 + 숫자 2개 + 선택적 소수점과 숫자
        icd10_pattern = r"[A-Z]\d{2}(?:\.\d+)?"

        query_codes = re.findall(icd10_pattern, query)
        doc_codes = re.findall(icd10_pattern, document)

        # 코드 매칭 또는 질병명 매칭
        for code in query_codes:
            if code in doc_codes:
                return True

        # 질병명으로 매칭
        for code, disease_name in self.icd10_map.items():
            if disease_name in query and (code in document or disease_name in document):
                return True

        return False

    def _has_symptom_disease_match(self, symptoms: list[str], document: str) -> bool:
        """증상과 질병의 연관성 확인"""
        # 증상-질병 연관 매핑 (예시)
        symptom_disease_map = {
            "두통": ["편두통", "긴장성 두통", "뇌수막염", "고혈압"],
            "발열": ["감기", "독감", "폐렴", "요로감염"],
            "기침": ["감기", "폐렴", "천식", "기관지염"],
            "흉통": ["협심증", "심근경색", "늑막염", "역류성 식도염"],
            "복통": ["위염", "장염", "충수염", "담낭염"],
        }

        doc_lower = document.lower()

        for symptom in symptoms:
            if symptom in symptom_disease_map:
                related_diseases = symptom_disease_map[symptom]
                for disease in related_diseases:
                    if disease in doc_lower:
                        return True

        return False

    def _has_drug_info_match(self, query: str, document: str) -> bool:
        """약물 정보 정확성 확인"""
        # 약물명 + 용량 패턴
        drug_pattern = r"(\w+)\s*(\d+\.?\d*\s?(mg|g|ml|mcg|IU))"

        query_drugs = re.findall(drug_pattern, query, re.IGNORECASE)
        doc_drugs = re.findall(drug_pattern, document, re.IGNORECASE)

        # 약물명과 용량이 모두 일치하는지 확인
        for q_drug in query_drugs:
            drug_name = q_drug[0].lower()
            dosage = q_drug[1].lower()

            for d_drug in doc_drugs:
                if drug_name in d_drug[0].lower() and dosage == d_drug[1].lower():
                    return True

        return False

    def _has_dosage_info(self, document: str) -> bool:
        """용량 정보 포함 여부 확인"""
        dosage_pattern = r"\d+\.?\d*\s?(mg|g|ml|L|mcg|IU|mEq|mmol)"
        return bool(re.search(dosage_pattern, document, re.IGNORECASE))

    def _get_document_content(self, doc_id: str) -> str:
        """문서 ID로 내용 가져오기"""
        try:
            # document_ids에서 인덱스 찾기
            if doc_id in self.document_ids:
                idx = self.document_ids.index(doc_id)
                if idx < len(self.documents):
                    return self.documents[idx]
        except Exception:
            pass

        # ChromaDB에서 직접 가져오기
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            result = collection.get(ids=[doc_id], include=["documents"])
            if result["documents"]:
                return result["documents"][0]
        except Exception:
            pass

        return ""

    def _get_retrieval_method(self, doc_id: str, scores: dict) -> str:
        """어떤 검색 방법으로 찾았는지 표시"""
        # 디버깅용: 주요 검색 방법 판별
        # 실제로는 BM25와 Dense 점수를 별도로 추적해야 함
        return "hybrid"

    def update_alpha(self, new_alpha: float):
        """
        BM25 가중치 업데이트

        Args:
            new_alpha: 새로운 alpha 값 (0~1)
        """
        if 0 <= new_alpha <= 1:
            self.alpha = new_alpha
            logger.info(f"Alpha 값 업데이트: {new_alpha}")
        else:
            logger.error(f"잘못된 alpha 값: {new_alpha} (0~1 사이여야 함)")
