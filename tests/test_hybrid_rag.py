#!/usr/bin/env python3
"""
하이브리드 RAG 시스템 테스트 스크립트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hybrid_rag_system import HybridMedicalRAG, MedicalTokenizer
from src.models.medical_rag_system import MedicalRAGSystem
from src.models.dialogue_summarizer import DialogueSummarizer
import time
import json


def test_medical_tokenizer():
    """의학 토크나이저 테스트"""
    print("\n" + "="*60)
    print("의학 토크나이저 테스트")
    print("="*60)
    
    tokenizer = MedicalTokenizer()
    
    test_texts = [
        "환자가 아스피린 100mg을 하루 2회 복용 중입니다",
        "BP 140/90, HR 72, 체온 38.5도",
        "두통과 발열이 3일째 지속되고 있습니다",
        "폐렴 의심되어 chest X-ray 시행 예정",
        "Amoxicillin 500mg t.i.d. p.o. 처방"
    ]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"\n원문: {text}")
        print(f"토큰: {tokens}")


def test_hybrid_search():
    """하이브리드 검색 테스트"""
    print("\n" + "="*60)
    print("하이브리드 검색 테스트")
    print("="*60)
    
    # 하이브리드 RAG 초기화
    hybrid_rag = HybridMedicalRAG(
        alpha=0.5  # BM25와 Dense를 균등하게
    )
    
    test_queries = [
        "두통이 있고 아스피린 100mg 복용",
        "고혈압 환자의 혈압 관리 방법",
        "폐렴 진단 기준과 흉부 X-ray",
        "당뇨병 환자 인슐린 용량 조절"
    ]
    
    for query in test_queries:
        print(f"\n검색 쿼리: {query}")
        print("-" * 40)
        
        start_time = time.time()
        results = hybrid_rag.hybrid_search(query, top_k=3)
        search_time = time.time() - start_time
        
        print(f"검색 시간: {search_time:.2f}초")
        print(f"결과 수: {len(results)}개")
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] 점수: {result['score']:.3f}")
            print(f"    검색 방법: {result['retrieval_info']['method']}")
            print(f"    의학 보너스: {result['retrieval_info']['medical_bonus']:.3f}")
            print(f"    내용: {result['content'][:150]}...")


def test_alpha_comparison():
    """Alpha 값에 따른 검색 결과 비교"""
    print("\n" + "="*60)
    print("Alpha 값 비교 테스트 (BM25 vs Dense 가중치)")
    print("="*60)
    
    query = "환자가 3일 전부터 발열과 기침을 호소하며 아목시실린 500mg 복용 중"
    
    alphas = [0.2, 0.5, 0.8]  # 0.2=Dense 중시, 0.5=균등, 0.8=BM25 중시
    
    for alpha in alphas:
        print(f"\n\nAlpha = {alpha} ({'BM25 중시' if alpha > 0.5 else 'Dense 중시' if alpha < 0.5 else '균등'})")
        print("-" * 40)
        
        hybrid_rag = HybridMedicalRAG(alpha=alpha)
        results = hybrid_rag.hybrid_search(query, top_k=3)
        
        for i, result in enumerate(results[:2], 1):  # 상위 2개만 표시
            print(f"[{i}] 점수: {result['score']:.3f}")
            print(f"    내용: {result['content'][:100]}...")


def test_dialogue_summarization_comparison():
    """기존 RAG vs 하이브리드 RAG 요약 비교"""
    print("\n" + "="*60)
    print("대화 요약 비교 (기존 RAG vs 하이브리드 RAG)")
    print("="*60)
    
    test_dialogue = """
    의사: 안녕하세요. 어떤 증상으로 오셨나요?
    환자: 3일 전부터 심한 두통과 발열이 있어요. 체온이 38.5도까지 올라갔어요.
    의사: 기침이나 가래는 있으신가요?
    환자: 네, 마른 기침이 계속 나고 목도 아파요.
    의사: 최근에 복용하신 약물이 있나요?
    환자: 타이레놀 500mg을 하루 3번 먹었는데 잘 안 들어요.
    의사: 청진 결과 폐에 이상 소견은 없습니다. 급성 상기도 감염으로 보입니다.
    의사: 아목시실린 500mg을 하루 3번, 5일간 처방하겠습니다.
    """
    
    # 1. 기존 RAG로 요약
    print("\n[기존 RAG 시스템]")
    rag_system = MedicalRAGSystem()
    summarizer_old = DialogueSummarizer(
        rag_system=rag_system,
        use_hybrid=False,
        streaming=False
    )
    
    start_time = time.time()
    result_old = summarizer_old.summarize_dialogue(test_dialogue, "P001")
    time_old = time.time() - start_time
    
    print(f"처리 시간: {time_old:.2f}초")
    print(f"주호소: {result_old['summary']['chief_complaint']}")
    print(f"진단: {result_old['summary']['diagnoses']}")
    print(f"처방: {result_old['summary']['medications']}")
    
    # 2. 하이브리드 RAG로 요약
    print("\n[하이브리드 RAG 시스템]")
    hybrid_rag = HybridMedicalRAG(alpha=0.6)  # 약간 BM25 중시
    summarizer_new = DialogueSummarizer(
        hybrid_rag_system=hybrid_rag,
        use_hybrid=True,
        streaming=False
    )
    
    start_time = time.time()
    result_new = summarizer_new.summarize_dialogue(test_dialogue, "P001")
    time_new = time.time() - start_time
    
    print(f"처리 시간: {time_new:.2f}초")
    print(f"주호소: {result_new['summary']['chief_complaint']}")
    print(f"진단: {result_new['summary']['diagnoses']}")
    print(f"처방: {result_new['summary']['medications']}")
    
    # 성능 비교
    print("\n" + "="*60)
    print("성능 비교")
    print("="*60)
    print(f"속도 개선: {((time_old - time_new) / time_old * 100):.1f}%")
    print(f"신뢰도 점수 - 기존: {result_old['confidence_score']:.2f}, 하이브리드: {result_new['confidence_score']:.2f}")


def main():
    """메인 테스트 실행"""
    print("\n하이브리드 RAG 시스템 테스트 시작")
    print("="*80)
    
    # 1. 토크나이저 테스트
    test_medical_tokenizer()
    
    # 2. 하이브리드 검색 테스트
    test_hybrid_search()
    
    # 3. Alpha 값 비교
    test_alpha_comparison()
    
    # 4. 대화 요약 비교
    test_dialogue_summarization_comparison()
    
    print("\n" + "="*80)
    print("테스트 완료!")


if __name__ == "__main__":
    main()