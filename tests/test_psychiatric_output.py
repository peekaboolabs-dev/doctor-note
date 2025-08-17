"""
정신과 SOAP 템플릿 출력 테스트
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.models.llm import LLMConfig
from src.models.summarizer.dialogue_summarizer import DialogueSummarizer


def main():
    # 샘플 정신과 상담 대화
    dialogue = """
의사: 안녕하세요. 오늘 어떤 일로 오셨나요?
환자: 최근 2주 동안 우울하고 아무것도 하기 싫어요. 잠도 잘 못 자고 있습니다.
의사: 언제부터 이런 증상이 시작되었나요?
환자: 한 달 전에 직장에서 스트레스를 많이 받은 후부터 시작된 것 같아요.
의사: 우울감 외에 다른 증상은 없나요?
환자: 불안하고 가슴이 답답해요. 식욕도 없고 체중이 3kg 정도 빠졌어요.
의사: 혹시 자살에 대한 생각을 한 적이 있나요?
환자: 아니요, 그런 생각은 없습니다. 그냥 모든 게 힘들기만 해요.
의사: 과거에 정신과 치료를 받은 적이 있나요?
환자: 5년 전에 우울증으로 약을 먹은 적이 있어요. 6개월 정도 치료받고 좋아졌었습니다.
의사: 현재 복용 중인 약물이 있나요?
환자: 고혈압 약만 먹고 있습니다.
의사: 알겠습니다. 정신상태 검사를 해보겠습니다. 오늘 날짜가 어떻게 되죠?
환자: 2024년 1월 15일입니다.
의사: 제 질문에 따라 간단한 계산을 해보세요. 100에서 7을 빼면?
환자: 93입니다.
"""

    print("=" * 80)
    print("정신과 SOAP 템플릿 테스트")
    print("=" * 80)

    # DialogueSummarizer 초기화
    config = LLMConfig(
        model_type="ollama",
        model_name="solar",
        streaming=True,  # 스트리밍으로 실시간 출력 확인
    )

    summarizer = DialogueSummarizer(llm_config=config, use_hybrid=False)

    print("\n1. 대화 파싱 결과:")
    print("-" * 40)
    turns = summarizer.parse_dialogue(dialogue)
    for i, (speaker, utterance) in enumerate(turns[:5], 1):  # 처음 5개만 출력
        print(f"{i}. {speaker}: {utterance[:50]}...")
    print(f"(총 {len(turns)}개 대화 턴)")

    print("\n2. 의학 개체 추출 (Entity Extraction):")
    print("-" * 40)
    print("추출 중...")
    entities = summarizer.extract_medical_entities(dialogue)

    print("\n추출된 정신과 정보:")
    print(f"- 주호소: {entities.chief_complaint}")
    print(f"- 정신과 증상: {entities.psychiatric_symptoms}")
    print(f"- 기분 상태: {entities.mood}")
    print(f"- 수면 패턴: {entities.sleep_pattern}")
    print(f"- 식욕: {entities.appetite}")
    print(f"- 증상 지속 기간: {entities.duration}")
    print(f"- 심각도: {entities.severity}")
    print(f"- 발병 시기: {entities.onset}")
    print(f"- 유발 요인: {entities.triggers}")
    print(f"- 자살 사고: {entities.suicidal_ideation}")
    print(f"- 타해 사고: {entities.homicidal_ideation}")
    print(f"- 과거 정신과 병력: {entities.past_psychiatric_history}")
    print(f"- 현재 약물: {entities.medications}")
    print(f"- 진단: {entities.diagnoses}")

    if entities.mental_status_exam:
        print("\n정신상태검사 (MSE):")
        mse = entities.mental_status_exam
        if mse.appearance:
            print(f"  - 외모: {mse.appearance}")
        if mse.behavior:
            print(f"  - 행동: {mse.behavior}")
        if mse.speech:
            print(f"  - 말: {mse.speech}")
        if mse.mood_affect:
            print(f"  - 기분/정동: {mse.mood_affect}")
        if mse.thought_process:
            print(f"  - 사고과정: {mse.thought_process}")
        if mse.thought_content:
            print(f"  - 사고내용: {mse.thought_content}")
        if mse.perception:
            print(f"  - 지각: {mse.perception}")
        if mse.cognition:
            print(f"  - 인지: {mse.cognition}")
        if mse.insight:
            print(f"  - 병식: {mse.insight}")

    print("\n3. SOAP 형식 상담 노트 생성:")
    print("-" * 40)
    print("요약 생성 중...")
    result = summarizer.summarize_dialogue(dialogue)

    # result 딕셔너리에서 summary 정보 추출
    soap_note = result["soap_note"]
    confidence_score = result["confidence_score"]

    print("\n" + "=" * 80)
    print("정신과 SOAP 상담 노트")
    print("=" * 80)

    print("\n## SUBJECTIVE (주관적 정보)")
    print(f"주호소: {soap_note['subjective']['chief_complaint']}")
    print(f"현병력: {soap_note['subjective']['history_present_illness']}")
    print(f"정신과 증상: {soap_note['subjective']['psychiatric_review']}")
    print(f"환자의 질병 인식: {soap_note['subjective']['patient_perception']}")

    print("\n## OBJECTIVE (객관적 정보)")
    if soap_note["objective"]["vital_signs"]:
        print(f"활력징후: {soap_note['objective']['vital_signs']}")

    if soap_note["objective"]["mental_status_exam"]:
        print("정신상태검사:")
        mse = soap_note["objective"]["mental_status_exam"]
        if mse.get("appearance"):
            print(f"  - 외모 및 태도: {mse['appearance']}")
        if mse.get("behavior"):
            print(f"  - 행동: {mse['behavior']}")
        if mse.get("speech"):
            print(f"  - 말: {mse['speech']}")
        if mse.get("mood_affect"):
            print(f"  - 기분/정동: {mse['mood_affect']}")
        if mse.get("thought_process"):
            print(f"  - 사고과정: {mse['thought_process']}")
        if mse.get("thought_content"):
            print(f"  - 사고내용: {mse['thought_content']}")
        if mse.get("perception"):
            print(f"  - 지각: {mse['perception']}")
        if mse.get("cognition"):
            print(f"  - 인지: {mse['cognition']}")
        if mse.get("insight"):
            print(f"  - 병식: {mse['insight']}")

    if soap_note["objective"]["physical_exam"]:
        print(f"신체 검사: {soap_note['objective']['physical_exam']}")

    print("\n## ASSESSMENT (평가)")
    if soap_note["assessment"]["diagnosis"]:
        print(f"진단: {', '.join(soap_note['assessment']['diagnosis'])}")
    print(f"경과: {soap_note['assessment']['progress']}")
    if soap_note["assessment"]["risk_factors"]:
        print(f"위험 요인: {', '.join(soap_note['assessment']['risk_factors'])}")

    print("\n## PLAN (계획)")
    if soap_note["plan"]["medications"]:
        print(f"약물치료: {', '.join(soap_note['plan']['medications'])}")
    print(f"정신치료: {soap_note['plan']['psychotherapy']}")
    if soap_note["plan"]["education"]:
        print(f"환자 교육: {', '.join(soap_note['plan']['education'])}")
    print(f"추적관찰: {soap_note['plan']['follow_up']}")
    if soap_note["plan"]["referrals"]:
        print(f"의뢰: {', '.join(soap_note['plan']['referrals'])}")
    print(f"안전 계획: {soap_note['plan']['safety_planning']}")

    print("\n" + "=" * 80)
    print(f"신뢰도 점수: {confidence_score:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback

        traceback.print_exc()
