#!/usr/bin/env python3
"""
의료 노트 분석 시스템 메인 엔트리포인트
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.models.medical_rag_system import MedicalRAGSystem, setup_medical_knowledge_base
from src.utils.config import load_config
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="의료 노트 분석 시스템")
    parser.add_argument(
        "--mode", 
        choices=["search", "add_note", "setup"], 
        default="search",
        help="실행 모드 선택"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="검색 쿼리 (search 모드에서 사용)"
    )
    parser.add_argument(
        "--note", 
        type=str, 
        help="추가할 환자 노트 (add_note 모드에서 사용)"
    )
    parser.add_argument(
        "--patient_id", 
        type=str, 
        help="환자 ID (add_note 모드에서 사용)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="검색 결과 개수"
    )
    
    args = parser.parse_args()
    
    try:
        # 설정 로드
        config = load_config()
        
        # RAG 시스템 초기화
        logger.info("의학 RAG 시스템 초기화 중...")
        rag_system = MedicalRAGSystem(
            model_name=config.get("embedding_model", "jhgan/ko-sroberta-multitask"),
            chroma_persist_dir=config.get("chroma_persist_dir", "data/embeddings/chroma_medical_db")
        )
        
        if args.mode == "setup":
            # 초기 설정 모드
            logger.info("의학 지식 베이스 설정 중...")
            setup_medical_knowledge_base()
            logger.info("설정 완료!")
            
        elif args.mode == "search":
            # 검색 모드
            if not args.query:
                parser.error("--query 옵션이 필요합니다")
            
            logger.info(f"검색 중: {args.query}")
            results = rag_system.search(args.query, k=args.k)
            
            print(f"\n검색 결과 (상위 {args.k}개):")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] 점수: {result['score']:.3f}")
                print(f"카테고리: {result['metadata'].get('category', 'N/A')}")
                print(f"내용: {result['content'][:200]}...")
                print("-" * 40)
                
        elif args.mode == "add_note":
            # 환자 노트 추가 모드
            if not args.note or not args.patient_id:
                parser.error("--note와 --patient_id 옵션이 필요합니다")
            
            logger.info(f"환자 노트 추가 중 (ID: {args.patient_id})")
            rag_system.add_patient_note(
                note_text=args.note,
                patient_id=args.patient_id
            )
            logger.info("환자 노트가 추가되었습니다")
            
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()