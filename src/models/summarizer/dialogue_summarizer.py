"""
의사-환자 대화 분석 및 요약 시스템
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
class MentalStatusExam:
    """정신상태검사 (MSE) 정보"""

    appearance: str | None = None  # 외모 및 태도
    behavior: str | None = None  # 행동 특징
    speech: str | None = None  # 말의 속도와 양
    mood_affect: str | None = None  # 기분/정동
    thought_process: str | None = None  # 사고 과정
    thought_content: str | None = None  # 사고 내용
    perception: str | None = None  # 지각 이상
    cognition: str | None = None  # 인지 기능
    insight: str | None = None  # 병식


@dataclass
class MedicalEntity:
    """정신과 의학적 개체 정보"""

    chief_complaint: str | None = None  # 주호소
    psychiatric_symptoms: list[str] = field(default_factory=list)  # 정신과 증상
    mood: str | None = None  # 기분 상태
    sleep_pattern: str | None = None  # 수면 패턴
    appetite: str | None = None  # 식욕 변화
    duration: str | None = None  # 증상 지속 기간
    severity: str | None = None  # 심각도
    onset: str | None = None  # 발병 시기
    triggers: list[str] = field(default_factory=list)  # 유발 요인
    mental_status_exam: MentalStatusExam = field(
        default_factory=MentalStatusExam
    )  # MSE
    suicidal_ideation: str | None = None  # 자살 사고
    homicidal_ideation: str | None = None  # 타해 사고
    substance_use: list[str] = field(default_factory=list)  # 물질 사용력
    past_psychiatric_history: str | None = None  # 과거 정신과 병력
    medications: list[str] = field(default_factory=list)  # 현재 복용 약물
    diagnoses: list[str] = field(default_factory=list)  # 진단명
    family_history: str | None = None  # 가족력
    vital_signs: dict[str, str] = field(default_factory=dict)  # 활력징후
    physical_exam: str | None = None  # 신체 검사


@dataclass
class SubjectiveSection:
    """SOAP - Subjective 섹션"""

    chief_complaint: str  # 주호소
    history_present_illness: str  # 현병력
    psychiatric_review: str  # 정신과적 증상 검토
    patient_perception: str  # 환자의 질병 인식


@dataclass
class ObjectiveSection:
    """SOAP - Objective 섹션"""

    vital_signs: dict[str, str] = field(default_factory=dict)  # 활력징후
    mental_status_exam: MentalStatusExam = field(
        default_factory=MentalStatusExam
    )  # MSE
    physical_exam: str | None = None  # 신체 검사


@dataclass
class AssessmentSection:
    """SOAP - Assessment 섹션"""

    diagnosis: list[str] = field(default_factory=list)  # 진단 (DSM 코드 포함)
    progress: str | None = None  # 경과 (이전 상태와 비교)
    risk_factors: list[str] = field(default_factory=list)  # 위험 요인


@dataclass
class PlanSection:
    """SOAP - Plan 섹션"""

    medications: list[str] = field(default_factory=list)  # 약물치료
    psychotherapy: str | None = None  # 정신치료
    education: list[str] = field(default_factory=list)  # 환자 교육
    follow_up: str | None = None  # 추적관찰
    referrals: list[str] = field(default_factory=list)  # 의뢰
    safety_planning: str | None = None  # 안전 계획


@dataclass
class ConsultationSummary:
    """정신과 SOAP 상담 요약 정보"""

    subjective: SubjectiveSection  # Subjective 섹션
    objective: ObjectiveSection  # Objective 섹션
    assessment: AssessmentSection  # Assessment 섹션
    plan: PlanSection  # Plan 섹션
    medical_entities: MedicalEntity  # 추출된 의학 정보
    confidence_score: float = 0.0  # 신뢰도 점수
    references: list[dict[str, str]] = field(default_factory=list)  # 참고자료


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
            llm_instance: 기존 LLM 인스턴스
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
                # temperature, top_p는 None으로 설정하여 .env에서 가져오기
                temperature=None,
                top_p=None,
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
                # temperature, top_p는 None으로 설정하여 .env에서 가져오기
                temperature=None,
                top_p=None,
                streaming=streaming,
            )
            self.llm = LLMFactory.create(self.llm_config)
            logger.info(f"Created default LLM: {self.llm.get_model_info()}")

        # 프롬프트 템플릿
        self.entity_extraction_prompt = PromptTemplate(
            input_variables=["dialogue"],
            template="""다음 정신과 의사-환자 대화에서 의학적 정보를 추출하세요.

대화:
{dialogue}

중요:
1. 추론 과정이나 설명 없이 JSON만 출력하세요
2. 한국어로 작성하세요
3. JSON 형식만 반환하세요

출력 형식:
{{
    "chief_complaint": "주호소",
    "psychiatric_symptoms": ["우울", "불안", "불면", "자살사고 등"],
    "mood": "기분 상태",
    "sleep_pattern": "수면 패턴",
    "appetite": "식욕 변화",
    "duration": "증상 지속 기간",
    "severity": "경증/중등증/중증",
    "onset": "발병 시기",
    "triggers": ["유발 요인들"],
    "mental_status_exam": {{
        "appearance": "외모 및 태도",
        "behavior": "행동 특징",
        "speech": "말의 속도와 양",
        "mood_affect": "기분/정동",
        "thought_process": "사고 과정",
        "thought_content": "사고 내용",
        "perception": "지각 이상",
        "cognition": "인지 기능",
        "insight": "병식"
    }},
    "suicidal_ideation": "자살 사고 유무",
    "homicidal_ideation": "타해 사고 유무",
    "substance_use": ["물질 사용력"],
    "past_psychiatric_history": "과거 정신과 병력",
    "medications": ["현재 복용 약물"],
    "diagnoses": ["진단명 또는 추정 진단"],
    "family_history": "가족력"
}}

JSON만 출력:""",
        )

        self.summary_generation_prompt = PromptTemplate(
            input_variables=["dialogue", "entities", "medical_context"],
            template="""당신은 정신건강의학과 전문의입니다. 다음 의사-환자 대화를 정신과 SOAP 노트 형식으로 작성하세요.

중요: 반드시 한국어로 작성하세요. 영어를 사용하지 마세요.

대화:
{dialogue}

추출된 의학 정보:
{entities}

관련 의학 지식:
{medical_context}

아래 정신과 SOAP 형식에 따라 한국어로 상담 노트를 작성하세요. 각 섹션을 빠짐없이 작성하세요:

## 정신과 상담 노트 (SOAP)

### Subjective (주관적 정보)
**주호소 (Chief Complaint):** [환자가 자신의 언어로 표현한 주된 문제]
**현병력 (History of Present Illness):** [증상의 발생 시기, 지속 기간, 심각도, 유발/완화 요인]
**정신과적 증상 검토 (Review of Systems - Psychiatric):** [환자가 보고한 기분, 불안, 수면, 식욕, 자살사고 등]
**환자의 질병 인식 (Patient's Perception):** [환자가 자신의 상태를 어떻게 이해하고 있는지]

### Objective (객관적 정보)
**활력징후 (Vital Signs):** [혈압, 맥박, 체온 등 (해당 시)]
**정신상태검사 (Mental Status Examination):**
- 외모 및 태도: [복장, 위생, 협조도]
- 정신운동활동: [안절부절, 지체, 틱 등]
- 기분 및 정동: [주관적 기분 / 객관적 정동]
- 사고과정: [사고의 흐름, 연상의 이완 등]
- 사고내용: [망상, 강박사고, 자살/타해사고]
- 지각이상: [환각, 착각 등]
- 인지기능: [지남력, 주의력, 기억력]
- 병식: [질병에 대한 인식 수준]
**신체 검사 (Physical Examination):** [관련 신체 소견 (해당 시)]

### Assessment (평가)
**진단 (Diagnosis):** [주진단 및 감별진단, DSM-5 코드 포함]
**경과 (Progress):** [이전 방문과 비교한 현재 상태]
**위험 요인 (Risk Factors):** [자해, 타해, 치료 불순응 위험성 평가]

### Plan (계획)
**약물치료 (Medications):** [처방 변경, 추가, 유지 사항]
**정신치료 (Psychotherapy):** [치료 유형, 초점, 빈도]
**환자 교육 (Education):** [질병, 약물, 대처방법에 대한 교육]
**추적관찰 (Follow-Up):** [다음 방문 시기 및 목표]
**의뢰 (Referrals):** [타과 또는 타 전문가 의뢰 (필요시)]
**안전 계획 (Safety Planning):** [위험 평가에 따른 안전 조치]

---
노트 작성 완료""",
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
        """
        대화에서 의학적 개체 추출

        Args:
            dialogue: 대화 텍스트

        Returns:
            추출된 의학적 개체 정보
        """
        try:
            # 프롬프트 생성
            prompt = self.entity_extraction_prompt.format(dialogue=dialogue)

            # LLM을 통한 개체 추출
            if self.streaming:
                print(">>> ", end="", flush=True)
                result_parts = []
                for chunk in self.llm.stream(prompt, extract_json=True):
                    print(chunk, end="", flush=True)
                    result_parts.append(chunk)
                print()  # 줄바꿈
                result = "".join(result_parts)
            else:
                response = self.llm.generate(prompt, extract_json=True)
                result = response.text

            # JSON 파싱 (코드블록 제거 및 정리)
            if result.strip().startswith("```"):
                # 코드블록 제거
                lines = result.strip().split("\n")
                # json 또는 JSON 태그가 있는 경우 제거
                if lines[0].lower().endswith(("json", "```")):
                    lines = lines[1:]
                if lines[-1] == "```":
                    lines = lines[:-1]
                json_content = "\n".join(lines)
            else:
                json_content = result

            # JSON 문자열 정리
            json_content = json_content.strip()
            # 제어 문자 제거
            json_content = json_content.replace("\t", "    ").replace("\r", "")

            # 불완전한 JSON 처리
            if json_content and not json_content.endswith("}"):
                open_braces = json_content.count("{") - json_content.count("}")
                json_content += "}" * open_braces

            # JSON의 첫 {부터 마지막 }까지만 추출
            start_idx = json_content.find("{")
            end_idx = json_content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_content = json_content[start_idx : end_idx + 1]

            entities_dict = json.loads(json_content)

            # MSE 정보 처리
            mse_dict = entities_dict.get("mental_status_exam", {})
            mse = MentalStatusExam(
                appearance=mse_dict.get("appearance"),
                behavior=mse_dict.get("behavior"),
                speech=mse_dict.get("speech"),
                mood_affect=mse_dict.get("mood_affect"),
                thought_process=mse_dict.get("thought_process"),
                thought_content=mse_dict.get("thought_content"),
                perception=mse_dict.get("perception"),
                cognition=mse_dict.get("cognition"),
                insight=mse_dict.get("insight"),
            )

            # MedicalEntity 객체로 변환
            entities = MedicalEntity(
                chief_complaint=entities_dict.get("chief_complaint"),
                psychiatric_symptoms=entities_dict.get("psychiatric_symptoms", []),
                mood=entities_dict.get("mood"),
                sleep_pattern=entities_dict.get("sleep_pattern"),
                appetite=entities_dict.get("appetite"),
                duration=entities_dict.get("duration"),
                severity=entities_dict.get("severity"),
                onset=entities_dict.get("onset"),
                triggers=entities_dict.get("triggers", []),
                mental_status_exam=mse,
                suicidal_ideation=entities_dict.get("suicidal_ideation"),
                homicidal_ideation=entities_dict.get("homicidal_ideation"),
                substance_use=entities_dict.get("substance_use", []),
                past_psychiatric_history=entities_dict.get("past_psychiatric_history"),
                medications=entities_dict.get("medications", []),
                diagnoses=entities_dict.get("diagnoses", []),
                family_history=entities_dict.get("family_history"),
                vital_signs=entities_dict.get("vital_signs", {}),
                physical_exam=entities_dict.get("physical_exam"),
            )

            logger.info(f"추출된 정신과 의학 개체: {entities}")
            return entities

        except Exception as e:
            logger.error(f"의학 개체 추출 실패: {e}")
            return MedicalEntity()

    def get_medical_context(self, entities: MedicalEntity) -> str:
        """
        추출된 개체를 기반으로 관련 정신과 의학 정보 검색

        Args:
            entities: 정신과 의학적 개체 정보

        Returns:
            관련 의학 컨텍스트
        """
        # 검색 쿼리 구성
        queries = []

        if entities.chief_complaint:
            queries.append(entities.chief_complaint)

        if entities.psychiatric_symptoms:
            queries.append(
                " ".join(entities.psychiatric_symptoms[:3])
            )  # 주요 정신과 증상

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
        구조화된 상담 요약 생성

        Args:
            dialogue: 대화 내용
            entities: 추출된 의학 개체
            medical_context: 관련 의학 정보

        Returns:
            상담 요약
        """
        try:
            # entities를 JSON 직렬화 가능한 형태로 변환
            entities_dict = {
                "chief_complaint": entities.chief_complaint,
                "psychiatric_symptoms": entities.psychiatric_symptoms,
                "mood": entities.mood,
                "sleep_pattern": entities.sleep_pattern,
                "appetite": entities.appetite,
                "duration": entities.duration,
                "severity": entities.severity,
                "onset": entities.onset,
                "triggers": entities.triggers,
                "suicidal_ideation": entities.suicidal_ideation,
                "homicidal_ideation": entities.homicidal_ideation,
                "substance_use": entities.substance_use,
                "past_psychiatric_history": entities.past_psychiatric_history,
                "medications": entities.medications,
                "diagnoses": entities.diagnoses,
                "family_history": entities.family_history,
                "vital_signs": entities.vital_signs,
                "physical_exam": entities.physical_exam,
            }

            # MSE 정보 추가
            if entities.mental_status_exam:
                mse = entities.mental_status_exam
                entities_dict["mental_status_exam"] = {
                    "appearance": mse.appearance,
                    "behavior": mse.behavior,
                    "speech": mse.speech,
                    "mood_affect": mse.mood_affect,
                    "thought_process": mse.thought_process,
                    "thought_content": mse.thought_content,
                    "perception": mse.perception,
                    "cognition": mse.cognition,
                    "insight": mse.insight,
                }

            # 프롬프트 생성
            prompt = self.summary_generation_prompt.format(
                dialogue=dialogue,
                entities=json.dumps(entities_dict, ensure_ascii=False),
                medical_context=medical_context,
            )

            # 요약 생성
            if self.streaming:
                print(">>> ", end="", flush=True)
                summary_parts = []
                for chunk in self.llm.stream(prompt):
                    print(chunk, end="", flush=True)
                    summary_parts.append(chunk)
                print()  # 줄바꿈
                summary_text = "".join(summary_parts)
            else:
                response = self.llm.generate(prompt)
                summary_text = response.text

            # 요약 파싱
            summary = self._parse_summary_text(summary_text, entities)

            # 신뢰도 점수 계산 (간단한 휴리스틱)
            summary.confidence_score = self._calculate_confidence(entities, summary)

            return summary

        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            # 기본 SOAP 요약 반환
            return ConsultationSummary(
                subjective=SubjectiveSection(
                    chief_complaint="요약 생성 실패",
                    history_present_illness="대화 내용을 요약할 수 없습니다.",
                    psychiatric_review="정신과 증상 검토 불가",
                    patient_perception="환자 인식 파악 불가",
                ),
                objective=ObjectiveSection(),
                assessment=AssessmentSection(),
                plan=PlanSection(),
                medical_entities=entities,
                confidence_score=0.0,
            )

    def _parse_summary_text(
        self, summary_text: str, entities: MedicalEntity
    ) -> ConsultationSummary:
        """정신과 SOAP 상담 요약 텍스트 파싱"""
        # 각 섹션 초기화
        subjective_data = {
            "chief_complaint": "",
            "history_present_illness": "",
            "psychiatric_review": "",
            "patient_perception": "",
        }
        objective_data = {
            "vital_signs": {},
            "mental_status_exam": {},
            "physical_exam": "",
        }
        assessment_data = {
            "diagnosis": [],
            "progress": "",
            "risk_factors": [],
        }
        plan_data = {
            "medications": [],
            "psychotherapy": "",
            "education": [],
            "follow_up": "",
            "referrals": [],
            "safety_planning": "",
        }

        # 섹션별 파싱 - SOAP 형식에 맞게
        current_section = None
        current_subsection = None
        lines = summary_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # SOAP 주요 섹션 확인
            line_lower = line.lower()

            # Subjective 섹션
            if "subjective" in line_lower or "주관적" in line:
                current_section = "subjective"
                current_subsection = None
            elif "주호소" in line or "chief complaint" in line_lower:
                current_section = "subjective"
                current_subsection = "chief_complaint"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        subjective_data["chief_complaint"] = content
            elif "현병력" in line or "history of present illness" in line_lower:
                current_section = "subjective"
                current_subsection = "history_present_illness"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        subjective_data["history_present_illness"] = content
            elif "정신과적 증상" in line or "psychiatric" in line_lower:
                current_section = "subjective"
                current_subsection = "psychiatric_review"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        subjective_data["psychiatric_review"] = content
            elif "환자의 질병 인식" in line or "patient's perception" in line_lower:
                current_section = "subjective"
                current_subsection = "patient_perception"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        subjective_data["patient_perception"] = content

            # Objective 섹션
            elif "objective" in line_lower or "객관적" in line:
                current_section = "objective"
                current_subsection = None
            elif "활력징후" in line or "vital signs" in line_lower:
                current_section = "objective"
                current_subsection = "vital_signs"
            elif "정신상태검사" in line or "mental status examination" in line_lower:
                current_section = "objective"
                current_subsection = "mental_status_exam"
            elif "신체 검사" in line or "physical examination" in line_lower:
                current_section = "objective"
                current_subsection = "physical_exam"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        objective_data["physical_exam"] = content

            # Assessment 섹션
            elif "assessment" in line_lower or "평가" in line:
                current_section = "assessment"
                current_subsection = None
            elif "진단" in line or "diagnosis" in line_lower:
                current_section = "assessment"
                current_subsection = "diagnosis"
            elif "경과" in line or "progress" in line_lower:
                current_section = "assessment"
                current_subsection = "progress"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        assessment_data["progress"] = content
            elif "위험 요인" in line or "risk factors" in line_lower:
                current_section = "assessment"
                current_subsection = "risk_factors"

            # Plan 섹션
            elif "plan" in line_lower or "계획" in line:
                current_section = "plan"
                current_subsection = None
            elif "약물치료" in line or "medications" in line_lower:
                current_section = "plan"
                current_subsection = "medications"
            elif "정신치료" in line or "psychotherapy" in line_lower:
                current_section = "plan"
                current_subsection = "psychotherapy"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        plan_data["psychotherapy"] = content
            elif "환자 교육" in line or "education" in line_lower:
                current_section = "plan"
                current_subsection = "education"
            elif "추적관찰" in line or "follow-up" in line_lower:
                current_section = "plan"
                current_subsection = "follow_up"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        plan_data["follow_up"] = content
            elif "의뢰" in line or "referrals" in line_lower:
                current_section = "plan"
                current_subsection = "referrals"
            elif "안전 계획" in line or "safety planning" in line_lower:
                current_section = "plan"
                current_subsection = "safety_planning"
                if ":" in line:
                    content = line.split(":", 1)[1].strip().strip("[]")
                    if content:
                        plan_data["safety_planning"] = content

            # 내용 추가
            else:
                if current_section == "subjective" and current_subsection:
                    if subjective_data[current_subsection]:
                        subjective_data[current_subsection] += " " + line
                    else:
                        subjective_data[current_subsection] = line
                elif current_section == "objective":
                    if current_subsection == "mental_status_exam" and "-" in line:
                        # MSE 항목 파싱
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip("- ")
                            value = parts[1].strip()
                            objective_data["mental_status_exam"][key] = value
                elif current_section == "assessment":
                    if (
                        current_subsection == "diagnosis"
                        or current_subsection == "risk_factors"
                    ):
                        # 리스트 항목 추가
                        if line.startswith(("-", "•", "*")) or line[0].isdigit():
                            clean_line = line.lstrip("0123456789.-•* ")
                            if clean_line:
                                assessment_data[current_subsection].append(clean_line)
                elif current_section == "plan":
                    if current_subsection in ["medications", "education", "referrals"]:
                        # 리스트 항목 추가
                        if line.startswith(("-", "•", "*")) or line[0].isdigit():
                            clean_line = line.lstrip("0123456789.-•* ")
                            if clean_line:
                                plan_data[current_subsection].append(clean_line)

        # 기본값 설정 - 빈 필드 채우기
        if not subjective_data["chief_complaint"] and entities.chief_complaint:
            subjective_data["chief_complaint"] = entities.chief_complaint
        elif not subjective_data["chief_complaint"] and entities.psychiatric_symptoms:
            subjective_data["chief_complaint"] = (
                f"{', '.join(entities.psychiatric_symptoms[:2])} 증상"
            )

        if not subjective_data["history_present_illness"]:
            hpi_parts = []
            if entities.onset:
                hpi_parts.append(f"발병: {entities.onset}")
            if entities.duration:
                hpi_parts.append(f"기간: {entities.duration}")
            if entities.severity:
                hpi_parts.append(f"심각도: {entities.severity}")
            subjective_data["history_present_illness"] = (
                " / ".join(hpi_parts) if hpi_parts else "정보 없음"
            )

        if not subjective_data["psychiatric_review"] and entities.psychiatric_symptoms:
            subjective_data["psychiatric_review"] = ", ".join(
                entities.psychiatric_symptoms
            )

        if not subjective_data["patient_perception"]:
            subjective_data["patient_perception"] = "환자의 질병 인식 평가 필요"

        # MSE 정보 통합
        if not objective_data["mental_status_exam"] and entities.mental_status_exam:
            mse = entities.mental_status_exam
            objective_data["mental_status_exam"] = MentalStatusExam(
                appearance=mse.appearance,
                behavior=mse.behavior,
                speech=mse.speech,
                mood_affect=mse.mood_affect,
                thought_process=mse.thought_process,
                thought_content=mse.thought_content,
                perception=mse.perception,
                cognition=mse.cognition,
                insight=mse.insight,
            )
        else:
            # 딕셔너리를 MentalStatusExam 객체로 변환
            mse_dict = objective_data["mental_status_exam"]
            objective_data["mental_status_exam"] = MentalStatusExam(
                appearance=mse_dict.get("외모 및 태도"),
                behavior=mse_dict.get("정신운동활동"),
                speech=mse_dict.get("말의 속도와 양"),
                mood_affect=mse_dict.get("기분 및 정동"),
                thought_process=mse_dict.get("사고과정"),
                thought_content=mse_dict.get("사고내용"),
                perception=mse_dict.get("지각이상"),
                cognition=mse_dict.get("인지기능"),
                insight=mse_dict.get("병식"),
            )

        # Assessment 기본값
        if not assessment_data["diagnosis"] and entities.diagnoses:
            assessment_data["diagnosis"] = entities.diagnoses

        if entities.suicidal_ideation or entities.homicidal_ideation:
            if entities.suicidal_ideation and "자살" not in str(
                assessment_data["risk_factors"]
            ):
                assessment_data["risk_factors"].append(
                    f"자살 사고: {entities.suicidal_ideation}"
                )
            if entities.homicidal_ideation and "타해" not in str(
                assessment_data["risk_factors"]
            ):
                assessment_data["risk_factors"].append(
                    f"타해 사고: {entities.homicidal_ideation}"
                )

        # Plan 기본값
        if not plan_data["medications"] and entities.medications:
            plan_data["medications"] = entities.medications

        if not plan_data["follow_up"]:
            plan_data["follow_up"] = "2-4주 후 재평가"

        # ConsultationSummary 객체 생성
        return ConsultationSummary(
            subjective=SubjectiveSection(
                chief_complaint=subjective_data["chief_complaint"]
                or "주호소 정보 없음",
                history_present_illness=subjective_data["history_present_illness"]
                or "현병력 정보 없음",
                psychiatric_review=subjective_data["psychiatric_review"]
                or "정신과 증상 검토 필요",
                patient_perception=subjective_data["patient_perception"]
                or "환자 인식 평가 필요",
            ),
            objective=ObjectiveSection(
                vital_signs=objective_data["vital_signs"] or entities.vital_signs,
                mental_status_exam=objective_data["mental_status_exam"],
                physical_exam=objective_data["physical_exam"] or entities.physical_exam,
            ),
            assessment=AssessmentSection(
                diagnosis=assessment_data["diagnosis"],
                progress=assessment_data["progress"] or "초진",
                risk_factors=assessment_data["risk_factors"],
            ),
            plan=PlanSection(
                medications=plan_data["medications"],
                psychotherapy=plan_data["psychotherapy"] or "정신치료 고려",
                education=plan_data["education"],
                follow_up=plan_data["follow_up"],
                referrals=plan_data["referrals"],
                safety_planning=plan_data["safety_planning"] or "위험 평가 후 결정",
            ),
            medical_entities=entities,
        )

    def _calculate_confidence(
        self, entities: MedicalEntity, summary: ConsultationSummary
    ) -> float:
        """정신과 SOAP 요약 신뢰도 점수 계산"""
        score = 0.0

        # 정신과 개체 추출 완성도 (40%)
        if entities.chief_complaint:
            score += 0.05
        if entities.psychiatric_symptoms:
            score += 0.1
        if entities.diagnoses:
            score += 0.1
        if entities.medications:
            score += 0.05
        if entities.mental_status_exam and any(
            [
                entities.mental_status_exam.appearance,
                entities.mental_status_exam.mood_affect,
                entities.mental_status_exam.thought_content,
            ]
        ):
            score += 0.1

        # SOAP 요약 완성도 (60%)
        # Subjective 섹션 (15%)
        if (
            summary.subjective.chief_complaint
            and summary.subjective.chief_complaint != "주호소 정보 없음"
        ):
            score += 0.05
        if (
            summary.subjective.history_present_illness
            and summary.subjective.history_present_illness != "현병력 정보 없음"
        ):
            score += 0.05
        if (
            summary.subjective.psychiatric_review
            and summary.subjective.psychiatric_review != "정신과 증상 검토 필요"
        ):
            score += 0.05

        # Objective 섹션 (15%)
        if summary.objective.mental_status_exam:
            score += 0.1
        if summary.objective.vital_signs:
            score += 0.05

        # Assessment 섹션 (15%)
        if summary.assessment.diagnosis:
            score += 0.1
        if summary.assessment.risk_factors:
            score += 0.05

        # Plan 섹션 (15%)
        if summary.plan.medications:
            score += 0.05
        if summary.plan.psychotherapy and summary.plan.psychotherapy != "정신치료 고려":
            score += 0.05
        if (
            summary.plan.safety_planning
            and summary.plan.safety_planning != "위험 평가 후 결정"
        ):
            score += 0.05

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

        # 5. 결과 구성 - SOAP 형식
        result = {
            "patient_id": patient_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "dialogue_turns": len(turns),
            "llm_model": self.llm.get_model_info(),
            "soap_note": {
                "subjective": {
                    "chief_complaint": summary.subjective.chief_complaint,
                    "history_present_illness": summary.subjective.history_present_illness,
                    "psychiatric_review": summary.subjective.psychiatric_review,
                    "patient_perception": summary.subjective.patient_perception,
                },
                "objective": {
                    "vital_signs": summary.objective.vital_signs,
                    "mental_status_exam": {
                        "appearance": summary.objective.mental_status_exam.appearance,
                        "behavior": summary.objective.mental_status_exam.behavior,
                        "speech": summary.objective.mental_status_exam.speech,
                        "mood_affect": summary.objective.mental_status_exam.mood_affect,
                        "thought_process": summary.objective.mental_status_exam.thought_process,
                        "thought_content": summary.objective.mental_status_exam.thought_content,
                        "perception": summary.objective.mental_status_exam.perception,
                        "cognition": summary.objective.mental_status_exam.cognition,
                        "insight": summary.objective.mental_status_exam.insight,
                    }
                    if summary.objective.mental_status_exam
                    else {},
                    "physical_exam": summary.objective.physical_exam,
                },
                "assessment": {
                    "diagnosis": summary.assessment.diagnosis,
                    "progress": summary.assessment.progress,
                    "risk_factors": summary.assessment.risk_factors,
                },
                "plan": {
                    "medications": summary.plan.medications,
                    "psychotherapy": summary.plan.psychotherapy,
                    "education": summary.plan.education,
                    "follow_up": summary.plan.follow_up,
                    "referrals": summary.plan.referrals,
                    "safety_planning": summary.plan.safety_planning,
                },
            },
            "extracted_entities": {
                "chief_complaint": entities.chief_complaint,
                "psychiatric_symptoms": entities.psychiatric_symptoms,
                "mood": entities.mood,
                "sleep_pattern": entities.sleep_pattern,
                "appetite": entities.appetite,
                "duration": entities.duration,
                "severity": entities.severity,
                "onset": entities.onset,
                "triggers": entities.triggers,
                "suicidal_ideation": entities.suicidal_ideation,
                "homicidal_ideation": entities.homicidal_ideation,
                "past_psychiatric_history": entities.past_psychiatric_history,
                "medications": entities.medications,
                "diagnoses": entities.diagnoses,
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
