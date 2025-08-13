"""
의사-환자 대화 분석 및 요약 시스템 - 개선된 버전
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain.prompts import PromptTemplate

from src.models.llm import LLMConfig, LLMFactory
from src.models.rag.hybrid_rag_system import HybridMedicalRAG
from src.models.rag.medical_rag_system import MedicalRAGSystem
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MedicalEntity:
    """의학적 개체 정보"""

    symptoms: list[str] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    examinations: list[str] = field(default_factory=list)
    vital_signs: dict[str, str] = field(default_factory=dict)
    duration: str | None = None
    severity: str | None = None


@dataclass
class ConsultationSummary:
    """상담 요약 정보"""

    chief_complaint: str
    present_illness: str
    medical_entities: MedicalEntity
    assessment: str
    plan: list[str]
    follow_up: str | None = None
    warnings: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    references: list[dict[str, str]] = field(default_factory=list)


class DialogueSummarizer:
    """의사-환자 대화 요약 시스템"""

    def __init__(
        self,
        rag_system: MedicalRAGSystem | None = None,
        hybrid_rag_system: HybridMedicalRAG | None = None,
        llm_config: LLMConfig | None = None,
        llm_instance: Any | None = None,  # 기존 LLM 인스턴스 재사용
        # 기존 Ollama 호환성을 위한 파라미터
        ollama_model: str | None = None,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        streaming: bool = True,
        use_hybrid: bool = True,
    ):
        """
        Args:
            rag_system: 의학 RAG 시스템 (기존 방식)
            hybrid_rag_system: 하이브리드 RAG 시스템 (BM25+Dense)
            llm_config: LLM 설정 (새로운 방식)
            ollama_model: Ollama 모델명 (하위 호환성)
            ollama_host: Ollama 서버 호스트
            ollama_port: Ollama 서버 포트
            streaming: 스트리밍 출력 여부
            use_hybrid: 하이브리드 RAG 사용 여부
        """
        self.use_hybrid = use_hybrid
        if use_hybrid and hybrid_rag_system:
            self.hybrid_rag = hybrid_rag_system
            self.rag_system = None
        else:
            self.rag_system = rag_system or MedicalRAGSystem()
            self.hybrid_rag = None
        self.streaming = streaming

        # LLM 초기화 (새로운 방식)
        if llm_instance:
            # 기존 LLM 인스턴스 재사용 (메모리 절약)
            self.llm = llm_instance
            self.llm_config = getattr(llm_instance, "config", None)
            logger.info(f"Reusing existing LLM instance: {self.llm.get_model_info()}")
        elif llm_config:
            # 새로운 설정 방식
            self.llm_config = llm_config
            self.llm = LLMFactory.create(self.llm_config)
            logger.info(f"Created new LLM: {self.llm.get_model_info()}")
        elif ollama_model:
            # 기존 Ollama 호환 모드
            self.llm_config = LLMConfig(
                model_type="ollama",
                model_name=ollama_model,
                ollama_host=ollama_host,
                ollama_port=ollama_port,
                temperature=0.3,
                top_p=0.9,
                streaming=streaming,
            )
            self.llm = LLMFactory.create(self.llm_config)
            logger.info(f"Created Ollama LLM: {self.llm.get_model_info()}")
        else:
            # 기본값
            self.llm_config = LLMConfig(
                model_type="ollama",
                model_name="solar",
                ollama_host=ollama_host,
                ollama_port=ollama_port,
                temperature=0.3,
                top_p=0.9,
                streaming=streaming,
            )
            self.llm = LLMFactory.create(self.llm_config)
            logger.info(f"Created default LLM: {self.llm.get_model_info()}")

        # 프롬프트 템플릿
        # llama.cpp와 ollama 모두 호환되도록 명확한 지시
        self.entity_extraction_prompt = PromptTemplate(
            input_variables=["dialogue"],
            template="""다음 의사-환자 대화에서 의학적 정보를 추출하세요.

대화:
{dialogue}

중요: 한국어로 작성하세요.

다음 형식으로 JSON을 반환하세요:
{{
    "symptoms": ["증상1", "증상2"],
    "duration": "증상 지속 기간",
    "severity": "경증/중등증/중증",
    "vital_signs": {{"체온": "값", "혈압": "값"}},
    "examinations": ["검사1", "검사2"],
    "diagnoses": ["진단명1", "진단명2"],
    "medications": ["약물명1", "약물명2"]
}}

JSON:""",
        )

        self.summary_generation_prompt = PromptTemplate(
            input_variables=["dialogue", "entities", "medical_context"],
            template="""당신은 의료 전문가입니다. 다음 의사-환자 대화를 구조화된 상담 노트로 요약하세요.

대화:
{dialogue}

추출된 의학 정보:
{entities}

관련 의학 지식:
{medical_context}

지시사항:
- 추론 과정이나 설명 없이 바로 상담 노트를 작성하세요
- 아래 형식을 정확히 따라 작성하세요
- [BEGIN_NOTE]와 [END_NOTE] 태그 사이에 노트를 작성하세요

[BEGIN_NOTE]

**주호소 (Chief Complaint)**
환자가 방문한 주된 이유를 한 문장으로 작성

**현병력 (Present Illness)**
증상의 발생 시기, 양상, 악화/완화 요인 등을 시간 순서대로 기술

**평가 (Assessment)**
수집된 정보를 바탕으로 한 의학적 평가와 감별 진단

**계획 (Plan)**
1. 처방 및 치료 계획
2. 필요한 추가 검사
3. 생활 습관 권고사항

**추적 관찰 (Follow-up)**
재방문 시기 및 주의사항

[END_NOTE]""",
        )

        # 대화 파서 패턴
        self.dialogue_pattern = re.compile(
            r"(의사|환자|Doctor|Patient|Dr\.|D|P)[\s:：]\s*(.+?)(?=(?:의사|환자|Doctor|Patient|Dr\.|D|P)[\s:：]|$)",
            re.DOTALL | re.IGNORECASE,
        )

    def parse_dialogue(self, dialogue: str) -> list[tuple[str, str]]:
        """
        대화를 화자별로 파싱

        Args:
            dialogue: 원본 대화 텍스트

        Returns:
            [(화자, 발화내용)] 리스트
        """
        turns = []
        matches = self.dialogue_pattern.findall(dialogue)

        for speaker, utterance in matches:
            # 화자 정규화
            if speaker.lower() in ["의사", "doctor", "dr.", "d"]:
                speaker = "의사"
            else:
                speaker = "환자"

            # 발화 내용 정리
            utterance = utterance.strip()
            if utterance:
                turns.append((speaker, utterance))

        return turns

    def extract_medical_entities(self, dialogue: str) -> MedicalEntity:
        """의학 개체 추출 - 프롬프트 개선"""
        try:
            # 구체적인 예시를 포함한 프롬프트
            prompt = f"""Extract medical information from the dialogue below and return a complete JSON object.

Dialogue: {dialogue}

Return this exact format filled with actual values (no placeholders like [...]):
{{
    "symptoms": ["기침", "발열", "가래"],
    "duration": "3일",
    "severity": "중증",
    "vital_signs": {{"체온": "38.5", "혈압": ""}},
    "examinations": ["청진", "흉부 X-ray"],
    "diagnoses": ["폐렴 의심"],
    "medications": []
}}

Important: Replace example values with actual information from the dialogue.
JSON:"""

            # extract_json 플래그 추가
            if self.streaming:
                print(">>> ", end="", flush=True)
                result_parts = []
                json_started = False
                brace_count = 0
                skip_count = 0

                for chunk in self.llm.stream(prompt, extract_json=True):
                    # 처음 몇 청크는 스킵 (>>> 등 제거)
                    if skip_count < 2 and (">>>" in chunk or not chunk.strip()):
                        skip_count += 1
                        continue

                    # JSON 시작 감지
                    if "{" in chunk and not json_started:
                        json_started = True
                        idx = chunk.find("{")
                        chunk = chunk[idx:]  # { 이전 부분 제거

                    if json_started:
                        result_parts.append(chunk)
                        brace_count += chunk.count("{") - chunk.count("}")

                        # JSON 완료 시 중단
                        if brace_count == 0 and len(result_parts) > 0:
                            break

                    if json_started:  # JSON 시작 후에만 출력
                        print(chunk, end="", flush=True)

                print()
                result = "".join(result_parts)
            else:
                response = self.llm.generate(prompt, extract_json=True)
                result = response.text

            # JSON 추출 및 파싱
            result = self._extract_clean_json(result)

            # [...] 같은 placeholder 체크
            if "[...]" in result or "..." in result:
                logger.warning("Placeholder detected in JSON, using fallback")
                return self._fallback_entity_extraction(dialogue)

            entities_dict = json.loads(result)

            # 빈 값 체크
            if not any(
                [
                    entities_dict.get("symptoms"),
                    entities_dict.get("diagnoses"),
                    entities_dict.get("vital_signs", {}).get("체온"),
                ]
            ):
                logger.warning("Empty JSON values, using fallback")
                return self._fallback_entity_extraction(dialogue)

            return MedicalEntity(
                symptoms=entities_dict.get("symptoms", []),
                duration=entities_dict.get("duration"),
                severity=entities_dict.get("severity"),
                vital_signs=entities_dict.get("vital_signs", {}),
                examinations=entities_dict.get("examinations", []),
                diagnoses=entities_dict.get("diagnoses", []),
                medications=entities_dict.get("medications", []),
            )

        except Exception as e:
            logger.error(f"의학 개체 추출 실패: {e}")
            # 폴백 사용
            return self._fallback_entity_extraction(dialogue)

    def _fallback_entity_extraction(self, dialogue: str) -> MedicalEntity:
        """대화에서 직접 의학 정보 추출 (폴백)"""
        import re

        entity = MedicalEntity()

        # 증상 추출
        if "기침" in dialogue or "cough" in dialogue.lower():
            entity.symptoms.append("기침")
        if "열" in dialogue or "fever" in dialogue.lower():
            entity.symptoms.append("발열")
        if "가래" in dialogue or "sputum" in dialogue.lower():
            entity.symptoms.append("가래")
        if (
            "답답" in dialogue
            or "dyspnea" in dialogue.lower()
            or "breath" in dialogue.lower()
        ):
            entity.symptoms.append("호흡곤란")

        # 체온 추출
        temp_match = re.search(r"(\d+\.?\d*)\s*도|℃", dialogue)
        if temp_match:
            entity.vital_signs["체온"] = temp_match.group(1) + "℃"

        # 기간 추출
        duration_match = re.search(r"(\d+)\s*일", dialogue)
        if duration_match:
            entity.duration = f"{duration_match.group(1)}일"

        # 진단 추출
        if "폐렴" in dialogue or "pneumonia" in dialogue.lower():
            entity.diagnoses.append("폐렴 의심")

        # 검사 추출
        if "X-ray" in dialogue or "엑스레이" in dialogue or "x-ray" in dialogue.lower():
            entity.examinations.append("흉부 X-ray")
        if "수포음" in dialogue or "crackles" in dialogue.lower():
            entity.examinations.append("폐 수포음")
        if "청진" in dialogue:
            entity.examinations.append("청진")

        # 심각도 추출
        if "38.5" in dialogue or "심하" in dialogue:
            entity.severity = "중증"
        elif "조금" in dialogue:
            entity.severity = "경증"

        logger.info(f"폴백 추출 완료: {entity}")
        return entity

    def _extract_clean_json(self, text: str) -> str:
        """텍스트에서 깨끗한 JSON만 추출"""
        import re

        # 모든 공백 정리
        text = text.strip()

        # JSON 객체 찾기
        match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            # 불필요한 이스케이프 제거
            json_str = json_str.replace('\\"', '"')
            return json_str

        # { 로 시작하는 경우
        if text.startswith("{"):
            # 마지막 } 찾기
            last_brace = text.rfind("}")
            if last_brace != -1:
                return text[: last_brace + 1]

        return text

    def get_medical_context(self, entities: MedicalEntity) -> str:
        """
        추출된 개체를 기반으로 관련 의학 정보 검색

        Args:
            entities: 의학적 개체 정보

        Returns:
            관련 의학 컨텍스트
        """
        # 검색 쿼리 구성
        queries = []

        if entities.symptoms:
            queries.append(" ".join(entities.symptoms[:3]))  # 주요 증상

        if entities.diagnoses:
            queries.append(" ".join(entities.diagnoses))

        if entities.medications:
            # 약물명과 용량 정보 포함
            med_query = " ".join(entities.medications[:2])
            queries.append(med_query)

        # 하이브리드 RAG 또는 기존 RAG 사용
        all_contexts = []

        if self.use_hybrid and self.hybrid_rag:
            # 하이브리드 검색 사용
            for query in queries:
                if query:
                    results = self.hybrid_rag.hybrid_search(query, top_k=3)
                    for result in results:
                        context = f"[관련도: {result['score']:.2f}] {result['content']}"
                        all_contexts.append(context)
        else:
            # 기존 RAG 검색
            for query in queries:
                if query:
                    context = self.rag_system.get_relevant_medical_context(query, k=3)
                    all_contexts.append(context)

        return "\n\n".join(all_contexts)

    def generate_summary(
        self, dialogue: str, entities: MedicalEntity, medical_context: str
    ) -> ConsultationSummary:
        """
        구조화된 상담 요약 생성 - CoT 필터링 강화
        """
        try:
            # 프롬프트 생성 - 더 명확한 지시
            prompt = f"""당신은 의료 전문가입니다. 다음 의사-환자 대화를 한국어 상담 노트로 요약하세요.

대화:
{dialogue}

추출된 의학 정보:
{json.dumps(entities.__dict__, ensure_ascii=False)}

지시사항:
- 영어 설명이나 추론 과정 없이 바로 한국어 노트를 작성하세요
- [BEGIN_NOTE]로 시작하고 [END_NOTE]로 끝내세요
- 다음 형식을 정확히 따르세요:

[BEGIN_NOTE]

**주호소 (Chief Complaint)**
(환자가 방문한 주된 이유)

**현병력 (Present Illness)**
(증상의 발생 시기와 경과)

**평가 (Assessment)**
(의학적 평가와 감별 진단)

**계획 (Plan)**
1. 처방 및 치료
2. 추가 검사
3. 생활 습관 권고

**추적 관찰 (Follow-up)**
(재방문 시기 및 주의사항)

[END_NOTE]"""

            # 요약 생성
            if self.streaming:
                summary_parts = []
                note_started = False
                skip_cot = True

                for chunk in self.llm.stream(prompt):
                    # [BEGIN_NOTE] 감지
                    if "[BEGIN_NOTE]" in chunk:
                        note_started = True
                        skip_cot = False
                        idx = chunk.find("[BEGIN_NOTE]")
                        chunk = chunk[idx:]  # 이전 내용 제거

                    # 노트 시작 후에만 출력
                    if note_started or (not skip_cot):
                        print(chunk, end="", flush=True)
                        summary_parts.append(chunk)

                    # [END_NOTE] 감지 시 중단
                    if "[END_NOTE]" in chunk:
                        break

                print()  # 줄바꿈
                summary_text = "".join(summary_parts)
            else:
                response = self.llm.generate(prompt)
                summary_text = response.text

            # [BEGIN_NOTE] ~ [END_NOTE] 추출
            if "[BEGIN_NOTE]" in summary_text:
                start = summary_text.find("[BEGIN_NOTE]")
                end = summary_text.find("[END_NOTE]")
                if end != -1:
                    summary_text = summary_text[start : end + len("[END_NOTE]")]
                else:
                    summary_text = summary_text[start:] + "\n[END_NOTE]"

            # 요약 파싱
            summary = self._parse_summary_text(summary_text, entities)

            # 신뢰도 점수 계산
            summary.confidence_score = self._calculate_confidence(entities, summary)

            return summary

        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            # 기본 요약 반환
            return ConsultationSummary(
                chief_complaint=f"{', '.join(entities.symptoms[:2]) if entities.symptoms else '증상 정보 없음'}",
                present_illness=f"증상 지속 기간: {entities.duration or '정보 없음'}",
                medical_entities=entities,
                assessment="의학적 평가가 필요합니다",
                plan=["추가 검사 필요", "증상에 따른 처방 고려"],
                confidence_score=0.5,
            )

    def _parse_summary_text(
        self, summary_text: str, entities: MedicalEntity
    ) -> ConsultationSummary:
        """상담 요약 텍스트 파싱"""
        sections = {
            "chief_complaint": "",
            "present_illness": "",
            "assessment": "",
            "plan": [],
            "follow_up": "",
        }

        # 섹션별 파싱
        current_section = None
        lines = summary_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 섹션 헤더 확인
            if "주호소" in line or "Chief Complaint" in line:
                current_section = "chief_complaint"
            elif "현병력" in line or "Present Illness" in line:
                current_section = "present_illness"
            elif "평가" in line or "Assessment" in line:
                current_section = "assessment"
            elif "계획" in line or "Plan" in line:
                current_section = "plan"
            elif "추적" in line or "Follow-up" in line:
                current_section = "follow_up"
            else:
                # 내용 추가
                if current_section == "plan" and line.startswith(
                    ("1.", "2.", "3.", "-", "•")
                ):
                    sections["plan"].append(line.lstrip("123.-• "))
                elif current_section and current_section != "plan":
                    sections[current_section] += line + " "

        # 주호소가 비어있으면 증상에서 생성
        if not sections["chief_complaint"].strip() and entities.symptoms:
            sections["chief_complaint"] = f"{', '.join(entities.symptoms[:2])}"
            if entities.duration:
                sections["chief_complaint"] += f" ({entities.duration})"

        return ConsultationSummary(
            chief_complaint=sections["chief_complaint"].strip() or "증상 정보 없음",
            present_illness=sections["present_illness"].strip() or "현병력 정보 없음",
            medical_entities=entities,
            assessment=sections["assessment"].strip() or "평가 정보 없음",
            plan=sections["plan"] or ["추가 평가 필요"],
            follow_up=sections["follow_up"].strip() if sections["follow_up"] else None,
        )

    def _calculate_confidence(
        self, entities: MedicalEntity, summary: ConsultationSummary
    ) -> float:
        """요약 신뢰도 점수 계산"""
        score = 0.0

        # 개체 추출 완성도
        if entities.symptoms:
            score += 0.3
        if entities.diagnoses:
            score += 0.2
        if entities.medications:
            score += 0.2

        # 요약 완성도
        if summary.chief_complaint != "주호소 없음":
            score += 0.1
        if summary.assessment != "평가 없음":
            score += 0.1
        if len(summary.plan) > 1:
            score += 0.1

        return min(score, 1.0)

    def summarize_dialogue(
        self,
        dialogue: str,
        patient_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, any]:
        """
        의사-환자 대화를 요약

        Args:
            dialogue: 대화 내용
            patient_id: 환자 ID
            session_id: 세션 ID

        Returns:
            요약 결과 딕셔너리
        """
        logger.info(f"대화 요약 시작 - Patient: {patient_id}, Session: {session_id}")

        # 1. 대화 파싱
        print("\n[1/4] 대화 파싱 중...")
        turns = self.parse_dialogue(dialogue)
        logger.info(f"파싱된 대화 턴 수: {len(turns)}")

        # 2. 의학 개체 추출
        print("\n[2/4] 의학 정보 추출 중...")
        if self.streaming:
            print(">>> ", end="", flush=True)
        entities = self.extract_medical_entities(dialogue)

        # 3. 관련 의학 정보 검색
        print("\n[3/4] 관련 의학 지식 검색 중...")
        medical_context = self.get_medical_context(entities)

        # 4. 요약 생성
        print("\n[4/4] 상담 노트 생성 중...")
        if self.streaming:
            print(">>> ", end="", flush=True)
        summary = self.generate_summary(dialogue, entities, medical_context)

        # 5. 결과 구성
        result = {
            "patient_id": patient_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "dialogue_turns": len(turns),
            "llm_model": self.llm.get_model_info(),
            "summary": {
                "chief_complaint": summary.chief_complaint,
                "present_illness": summary.present_illness,
                "symptoms": entities.symptoms,
                "duration": entities.duration,
                "severity": entities.severity,
                "vital_signs": entities.vital_signs,
                "examinations": entities.examinations,
                "diagnoses": entities.diagnoses,
                "medications": entities.medications,
                "assessment": summary.assessment,
                "plan": summary.plan,
                "follow_up": summary.follow_up,
                "warnings": summary.warnings,
            },
            "confidence_score": summary.confidence_score,
            "references": summary.references,
        }

        logger.info(f"대화 요약 완료 - 신뢰도: {summary.confidence_score:.2f}")

        return result


if __name__ == "__main__":
    # 테스트
    summarizer = DialogueSummarizer()

    test_dialogue = """
    의사: 안녕하세요. 어떤 증상으로 오셨나요?
    환자: 3일 전부터 기침이 심하고 열이 나요.
    의사: 열은 몇 도까지 올라갔나요?
    환자: 어제 저녁에 38.5도까지 올라갔어요.
    의사: 가래는 나오나요?
    환자: 네, 노란색 가래가 나와요.
    의사: 숨쉬기는 괜찮으신가요?
    환자: 조금 답답한 느낌이 있어요.
    의사: 청진 결과 폐에 수포음이 들립니다. 폐렴이 의심되니 흉부 X-ray를 찍어보겠습니다.
    """

    result = summarizer.summarize_dialogue(test_dialogue, "P12345", "S67890")
    print(json.dumps(result, ensure_ascii=False, indent=2))
