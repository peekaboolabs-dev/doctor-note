#!/usr/bin/env python3
"""
의료 노트 분석 시스템 메인 엔트리포인트
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.models.rag.medical_rag_system import MedicalRAGSystem, setup_medical_knowledge_base
from src.models.summarizer.dialogue_summarizer import MedicalDialogueSummarizer
from src.utils.config import load_config
from src.utils.logger import setup_logger
import json

# 로거 설정
logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="의료 상담 요약 AI 시스템")
    parser.add_argument(
        "--mode", 
        choices=["search", "add_note", "setup", "summarize", "benchmark"], 
        default="summarize",
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
    parser.add_argument(
        "--dialogue",
        type=str,
        help="의사-환자 대화 내용 (summarize 모드에서 사용)"
    )
    parser.add_argument(
        "--dialogue_file",
        type=str,
        help="대화 파일 경로 (summarize 모드에서 사용)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="테스트 파일 경로 (benchmark 모드에서 사용)"
    )
    parser.add_argument(
        "--dialogue_id",
        type=str,
        help="JSON 파일에서 특정 대화 ID 선택"
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
            
        elif args.mode == "summarize":
            # 대화 요약 모드
            dialogue = None
            
            if args.dialogue_file:
                # 파일에서 읽기
                with open(args.dialogue_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # JSON 파일인지 확인
                if args.dialogue_file.endswith('.json'):
                    data = json.loads(content)
                    # 대화 추출
                    if 'dialogues' in data and len(data['dialogues']) > 0:
                        # 특정 ID로 선택하거나 첫 번째 대화 사용
                        dialogue_data = None
                        if args.dialogue_id:
                            for d in data['dialogues']:
                                if d.get('id') == args.dialogue_id:
                                    dialogue_data = d
                                    break
                            if not dialogue_data:
                                parser.error(f"대화 ID '{args.dialogue_id}'를 찾을 수 없습니다")
                        else:
                            dialogue_data = data['dialogues'][0]  # 첫 번째 대화 사용
                            
                        dialogue = dialogue_data.get('dialogue', '')
                        if not args.patient_id:
                            args.patient_id = dialogue_data.get('patient_id')
                        logger.info(f"JSON에서 대화 ID {dialogue_data.get('id')} 추출")
                    else:
                        parser.error("JSON 파일에 대화가 없습니다")
                else:
                    dialogue = content
            elif args.dialogue:
                dialogue = args.dialogue
            else:
                parser.error("--dialogue 또는 --dialogue_file 옵션이 필요합니다")
            
            logger.info("대화 요약 시작...")
            
            # 요약 시스템 초기화
            ollama_host = config.get("ollama_host", "localhost")
            ollama_model = config.get("ollama_model", "llama2")
            
            summarizer = MedicalDialogueSummarizer(
                rag_system=rag_system,
                ollama_model=ollama_model,
                ollama_host=ollama_host
            )
            
            # 요약 실행
            result = summarizer.summarize_dialogue(
                dialogue=dialogue,
                patient_id=args.patient_id
            )
            
            # 결과 출력
            print("\n" + "="*80)
            print("상담 요약 결과")
            print("="*80)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        elif args.mode == "benchmark":
            # 벤치마크 모드
            from src.models.benchmark.benchmark_runner import BenchmarkRunner
            
            test_file = args.test_file or "data/sample_dialogues.json"
            
            logger.info(f"벤치마크 실행: {test_file}")
            
            # 테스트 데이터 로드
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_dialogues = data.get("dialogues", [])
            
            # 벤치마크 실행
            runner = BenchmarkRunner()
            
            # 현재 설정된 모델로 벤치마크 실행
            model_name = config.get("ollama_model", "solar")
            results = runner.run_model_benchmark(model_name, test_dialogues)
            
            # 결과 저장
            saved_file = runner.save_results(results)
            print(f"\n벤치마크 결과가 저장되었습니다: {saved_file}")
            
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()