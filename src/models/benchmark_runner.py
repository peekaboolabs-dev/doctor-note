"""
모델 벤치마크 실행 및 결과 저장
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import time

from src.models.dialogue_summarizer import DialogueSummarizer
from src.models.medical_rag_system import MedicalRAGSystem
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BenchmarkRunner:
    """모델 성능 비교를 위한 벤치마크 실행기"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # RAG 시스템 초기화
        self.rag_system = MedicalRAGSystem()
        
    def run_model_benchmark(
        self, 
        model_name: str,
        test_dialogues: List[Dict[str, Any]],
        streaming: bool = True
    ) -> Dict[str, Any]:
        """
        특정 모델로 벤치마크 실행
        
        Args:
            model_name: 테스트할 모델명
            test_dialogues: 테스트 대화 리스트
            streaming: 스트리밍 출력 여부
            
        Returns:
            벤치마크 결과
        """
        logger.info(f"벤치마크 시작: {model_name}")
        
        # 요약 시스템 초기화
        summarizer = DialogueSummarizer(
            rag_system=self.rag_system,
            ollama_model=model_name,
            streaming=streaming
        )
        
        results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "test_count": len(test_dialogues),
            "results": []
        }
        
        # 각 대화에 대해 테스트
        for dialogue_data in test_dialogues:
            dialogue_id = dialogue_data.get("id", "unknown")
            logger.info(f"테스트 중: {dialogue_id}")
            
            start_time = time.time()
            
            try:
                # 요약 실행
                summary_result = summarizer.summarize_dialogue(
                    dialogue=dialogue_data["dialogue"],
                    patient_id=dialogue_data.get("patient_id")
                )
                
                execution_time = time.time() - start_time
                
                # 결과 저장
                test_result = {
                    "dialogue_id": dialogue_id,
                    "success": True,
                    "execution_time": execution_time,
                    "summary": summary_result,
                    "expected_summary": dialogue_data.get("expected_summary")
                }
                
            except Exception as e:
                logger.error(f"테스트 실패 {dialogue_id}: {e}")
                test_result = {
                    "dialogue_id": dialogue_id,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            
            results["results"].append(test_result)
            
        # 통계 계산
        successful_tests = [r for r in results["results"] if r["success"]]
        results["statistics"] = {
            "success_rate": len(successful_tests) / len(test_dialogues) if test_dialogues else 0,
            "avg_execution_time": sum(r["execution_time"] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "total_execution_time": sum(r["execution_time"] for r in results["results"])
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        결과를 JSON 파일로 저장
        
        Args:
            results: 벤치마크 결과
            filename: 저장할 파일명 (없으면 자동 생성)
        """
        if not filename:
            model_name = results["model"].replace(":", "_").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{model_name}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장됨: {filepath}")
        return filepath
    
    def compare_models(self, model_names: List[str], test_dialogues: List[Dict[str, Any]]):
        """
        여러 모델을 비교 테스트
        
        Args:
            model_names: 테스트할 모델명 리스트
            test_dialogues: 테스트 대화 리스트
        """
        comparison_results = {
            "comparison_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "models": model_names,
            "test_count": len(test_dialogues),
            "model_results": {}
        }
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"모델 테스트: {model_name}")
            logger.info(f"{'='*50}")
            
            # 모델별 벤치마크 실행
            results = self.run_model_benchmark(model_name, test_dialogues)
            
            # 결과 저장
            self.save_results(results)
            
            # 비교 결과에 추가
            comparison_results["model_results"][model_name] = {
                "success_rate": results["statistics"]["success_rate"],
                "avg_execution_time": results["statistics"]["avg_execution_time"],
                "file": self.save_results(results)
            }
        
        # 비교 결과 저장
        comparison_file = os.path.join(
            self.output_dir, 
            f"comparison_{comparison_results['comparison_id']}.json"
        )
        
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n비교 결과 저장됨: {comparison_file}")
        
        # 간단한 비교 출력
        print("\n" + "="*60)
        print("모델 비교 결과")
        print("="*60)
        for model, stats in comparison_results["model_results"].items():
            print(f"\n{model}:")
            print(f"  - 성공률: {stats['success_rate']*100:.1f}%")
            print(f"  - 평균 실행 시간: {stats['avg_execution_time']:.2f}초")


if __name__ == "__main__":
    # 테스트 실행
    runner = BenchmarkRunner()
    
    # 샘플 대화 로드
    with open("data/sample_dialogues.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        test_dialogues = data["dialogues"]
    
    # Solar 모델 벤치마크
    results = runner.run_model_benchmark("solar", test_dialogues)
    runner.save_results(results)