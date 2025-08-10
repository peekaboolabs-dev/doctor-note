"""
모델 벤치마크 실행 및 결과 저장
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import time
import psutil
import threading
from collections import defaultdict
import statistics as stats

from src.models.summarizer.dialogue_summarizer import MedicalDialogueSummarizer
from src.models.rag.medical_rag_system import MedicalRAGSystem
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResourceMonitor:
    """시스템 리소스 모니터링"""
    
    def __init__(self, include_ollama=True):
        self.process = psutil.Process()
        self.include_ollama = include_ollama
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'memory_percent': [],
            'ollama_memory_mb': [],
            'ollama_cpu_percent': []
        }
        self._lock = threading.Lock()
        
    def start(self):
        """모니터링 시작"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        
    def stop(self) -> Dict[str, float]:
        """모니터링 중지 및 통계 반환"""
        self.monitoring = False
        self.monitor_thread.join()
        
        with self._lock:
            result = {}
            if self.metrics['cpu_percent']:
                result['cpu_percent_avg'] = stats.mean(self.metrics['cpu_percent'])
                result['cpu_percent_max'] = max(self.metrics['cpu_percent'])
            
            if self.metrics['memory_mb']:
                result['memory_mb_avg'] = stats.mean(self.metrics['memory_mb'])
                result['memory_mb_max'] = max(self.metrics['memory_mb'])
                result['memory_percent_avg'] = stats.mean(self.metrics['memory_percent'])
            
            # Ollama 메트릭 추가
            if self.include_ollama and self.metrics['ollama_memory_mb']:
                result['ollama_memory_mb_avg'] = stats.mean(self.metrics['ollama_memory_mb'])
                result['ollama_memory_mb_max'] = max(self.metrics['ollama_memory_mb'])
                result['ollama_cpu_percent_avg'] = stats.mean(self.metrics['ollama_cpu_percent'])
                result['ollama_cpu_percent_max'] = max(self.metrics['ollama_cpu_percent'])
                
            return result
    
    def _monitor(self):
        """백그라운드 모니터링"""
        while self.monitoring:
            try:
                with self._lock:
                    # CPU 사용률
                    cpu = self.process.cpu_percent(interval=0.1)
                    self.metrics['cpu_percent'].append(cpu)
                    
                    # 메모리 사용량
                    mem_info = self.process.memory_info()
                    mem_mb = mem_info.rss / 1024 / 1024  # MB 단위
                    mem_percent = self.process.memory_percent()
                    
                    self.metrics['memory_mb'].append(mem_mb)
                    self.metrics['memory_percent'].append(mem_percent)
                    
                    # Ollama 프로세스 모니터링
                    if self.include_ollama:
                        ollama_memory = 0
                        ollama_cpu = 0
                        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                            try:
                                if 'ollama' in proc.info['name'].lower():
                                    ollama_memory += proc.info['memory_info'].rss / 1024 / 1024
                                    ollama_cpu += proc.cpu_percent(interval=0.1)
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        
                        self.metrics['ollama_memory_mb'].append(ollama_memory)
                        self.metrics['ollama_cpu_percent'].append(ollama_cpu)
                    
            except Exception as e:
                logger.error(f"리소스 모니터링 오류: {e}")
            
            time.sleep(0.5)  # 0.5초마다 측정


class StreamingMetricsCallback:
    """스트리밍 메트릭 수집 콜백"""
    
    def __init__(self):
        self.first_token_time = None
        self.start_time = None
        self.token_times = []
        self.tokens = []
        
    def on_llm_start(self, *args, **kwargs):
        """LLM 시작 시점"""
        self.start_time = time.time()
        
    def on_llm_new_token(self, token: str, **kwargs):
        """새 토큰 생성 시점"""
        current_time = time.time()
        
        if self.first_token_time is None:
            self.first_token_time = current_time - self.start_time
            
        self.token_times.append(current_time)
        self.tokens.append(token)
        
    def get_metrics(self) -> Dict[str, float]:
        """수집된 메트릭 반환"""
        metrics = {}
        
        if self.first_token_time:
            metrics['time_to_first_token'] = self.first_token_time
            
        if len(self.tokens) > 1:
            total_time = self.token_times[-1] - self.token_times[0]
            metrics['tokens_per_second'] = len(self.tokens) / total_time if total_time > 0 else 0
            
        metrics['total_tokens'] = len(self.tokens)
        
        return metrics


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
        
    def _count_tokens(self, text: str) -> int:
        """
        간단한 토큰 카운터 (공백 기준)
        실제로는 tiktoken 또는 모델별 토크나이저 사용 권장
        """
        # 한국어는 보통 문자당 1-2 토큰
        # 영어는 단어당 1-2 토큰
        # 간단히 공백 + 문자수 기반으로 추정
        words = text.split()
        korean_chars = sum(1 for char in text if '가' <= char <= '힣')
        
        # 추정 공식: 영어 단어 + (한국어 문자 / 2)
        estimated_tokens = len(words) + korean_chars // 2
        return max(estimated_tokens, len(words))  # 최소값은 단어 수
        
    def run_model_benchmark(
        self, 
        model_name: str,
        test_dialogues: List[Dict[str, Any]],
        streaming: bool = True,
        measure_resources: bool = True
    ) -> Dict[str, Any]:
        """
        특정 모델로 벤치마크 실행
        
        Args:
            model_name: 테스트할 모델명
            test_dialogues: 테스트 대화 리스트
            streaming: 스트리밍 출력 여부
            measure_resources: 리소스 사용량 측정 여부
            
        Returns:
            벤치마크 결과
        """
        logger.info(f"벤치마크 시작: {model_name}")
        
        # 요약 시스템 초기화
        summarizer = MedicalDialogueSummarizer(
            rag_system=self.rag_system,
            ollama_model=model_name,
            streaming=streaming
        )
        
        results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "test_count": len(test_dialogues),
            "streaming": streaming,
            "results": []
        }
        
        # 각 대화에 대해 테스트
        all_latencies = []
        
        for dialogue_data in test_dialogues:
            dialogue_id = dialogue_data.get("id", "unknown")
            logger.info(f"테스트 중: {dialogue_id}")
            
            # 리소스 모니터링 시작
            resource_monitor = ResourceMonitor() if measure_resources else None
            if resource_monitor:
                resource_monitor.start()
            
            # 입력 토큰 수 계산
            input_tokens = self._count_tokens(dialogue_data["dialogue"])
            
            start_time = time.time()
            
            try:
                # 요약 실행
                summary_result = summarizer.summarize_dialogue(
                    dialogue=dialogue_data["dialogue"],
                    patient_id=dialogue_data.get("patient_id")
                )
                
                execution_time = time.time() - start_time
                all_latencies.append(execution_time)
                
                # 출력 토큰 수 계산
                output_text = json.dumps(summary_result, ensure_ascii=False)
                output_tokens = self._count_tokens(output_text)
                
                # 결과 저장
                test_result = {
                    "dialogue_id": dialogue_id,
                    "success": True,
                    "execution_time": execution_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "tokens_per_second": output_tokens / execution_time if execution_time > 0 else 0,
                    "summary": summary_result,
                    "expected_summary": dialogue_data.get("expected_summary")
                }
                
                # 리소스 사용량 추가
                if resource_monitor:
                    resource_stats = resource_monitor.stop()
                    test_result["resources"] = resource_stats
                
            except Exception as e:
                logger.error(f"테스트 실패 {dialogue_id}: {e}")
                test_result = {
                    "dialogue_id": dialogue_id,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
                
                if resource_monitor:
                    resource_monitor.stop()
            
            results["results"].append(test_result)
            
        # 통계 계산
        successful_tests = [r for r in results["results"] if r["success"]]
        
        statistics = {
            "success_rate": len(successful_tests) / len(test_dialogues) if test_dialogues else 0,
            "total_execution_time": sum(r["execution_time"] for r in results["results"])
        }
        
        if successful_tests:
            # 시간 관련 통계
            execution_times = [r["execution_time"] for r in successful_tests]
            statistics["latency"] = {
                "mean": stats.mean(execution_times),
                "median": stats.median(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "p95": stats.quantiles(execution_times, n=20)[18] if len(execution_times) > 1 else execution_times[0],
                "p99": stats.quantiles(execution_times, n=100)[98] if len(execution_times) > 1 else execution_times[0]
            }
            
            # 토큰 관련 통계
            if "tokens_per_second" in successful_tests[0]:
                tps_values = [r["tokens_per_second"] for r in successful_tests]
                total_input_tokens = sum(r["input_tokens"] for r in successful_tests)
                total_output_tokens = sum(r["output_tokens"] for r in successful_tests)
                
                statistics["tokens"] = {
                    "total_input": total_input_tokens,
                    "total_output": total_output_tokens,
                    "avg_input_per_request": total_input_tokens / len(successful_tests),
                    "avg_output_per_request": total_output_tokens / len(successful_tests),
                    "tokens_per_second": {
                        "mean": stats.mean(tps_values),
                        "median": stats.median(tps_values),
                        "min": min(tps_values),
                        "max": max(tps_values)
                    }
                }
            
            # 리소스 사용량 통계
            if "resources" in successful_tests[0]:
                cpu_avgs = [r["resources"]["cpu_percent_avg"] for r in successful_tests if "cpu_percent_avg" in r["resources"]]
                mem_avgs = [r["resources"]["memory_mb_avg"] for r in successful_tests if "memory_mb_avg" in r["resources"]]
                mem_maxs = [r["resources"]["memory_mb_max"] for r in successful_tests if "memory_mb_max" in r["resources"]]
                
                resources_stats = {}
                
                if cpu_avgs and mem_avgs:
                    resources_stats["python_process"] = {
                        "cpu_percent": {
                            "mean": stats.mean(cpu_avgs),
                            "max": max(cpu_avgs)
                        },
                        "memory_mb": {
                            "mean": stats.mean(mem_avgs),
                            "max": max(mem_maxs)
                        }
                    }
                
                # Ollama 리소스 통계
                ollama_mem_avgs = [r["resources"]["ollama_memory_mb_avg"] for r in successful_tests if "ollama_memory_mb_avg" in r["resources"]]
                ollama_mem_maxs = [r["resources"]["ollama_memory_mb_max"] for r in successful_tests if "ollama_memory_mb_max" in r["resources"]]
                ollama_cpu_avgs = [r["resources"]["ollama_cpu_percent_avg"] for r in successful_tests if "ollama_cpu_percent_avg" in r["resources"]]
                
                if ollama_mem_avgs:
                    resources_stats["ollama_process"] = {
                        "cpu_percent": {
                            "mean": stats.mean(ollama_cpu_avgs),
                            "max": max(ollama_cpu_avgs)
                        },
                        "memory_mb": {
                            "mean": stats.mean(ollama_mem_avgs),
                            "max": max(ollama_mem_maxs)
                        }
                    }
                    
                    # 총 메모리 사용량
                    resources_stats["total_memory_mb"] = {
                        "mean": resources_stats["python_process"]["memory_mb"]["mean"] + resources_stats["ollama_process"]["memory_mb"]["mean"],
                        "max": resources_stats["python_process"]["memory_mb"]["max"] + resources_stats["ollama_process"]["memory_mb"]["max"]
                    }
                
                statistics["resources"] = resources_stats
        
        results["statistics"] = statistics
        
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