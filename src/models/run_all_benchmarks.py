#!/usr/bin/env python3
"""
모든 설치된 모델에 대한 벤치마크 실행
"""
import subprocess
import json
import time
from datetime import datetime

# 테스트할 모델 목록
MODELS = [
    "solar",
    "gpt-oss:20b",
    "gemma3:12b",
    "gemma3:27b",
    "qwen3:8b",
    "qwen3:30b"
]

def run_benchmark(model_name):
    """단일 모델 벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"모델 벤치마크 시작: {model_name}")
    print(f"{'='*60}")
    
    # .env 파일 업데이트
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    with open('.env', 'w') as f:
        for line in lines:
            if line.startswith('OLLAMA_MODEL='):
                f.write(f'OLLAMA_MODEL={model_name}\n')
            else:
                f.write(line)
    
    # 벤치마크 실행
    try:
        result = subprocess.run(
            ['python', 'main.py', '--mode', 'benchmark', '--test_file', 'data/sample_dialogues.json'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {model_name} 벤치마크 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} 벤치마크 실패: {e}")
        if e.stderr:
            print(f"에러 메시지: {e.stderr}")
        return False

def create_comparison_report():
    """모든 벤치마크 결과를 비교하는 리포트 생성"""
    import os
    import glob
    
    # 모든 벤치마크 결과 파일 찾기
    benchmark_files = glob.glob('benchmark_results/benchmark_*.json')
    
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "models": {}
    }
    
    for file_path in benchmark_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        model_name = data['model']
        
        # 주요 지표 추출
        stats = data.get('statistics', {})
        comparison['models'][model_name] = {
            "file": os.path.basename(file_path),
            "timestamp": data['timestamp'],
            "success_rate": stats.get('success_rate', 0),
            "latency": stats.get('latency', {}),
            "tokens": stats.get('tokens', {}),
            "resources": stats.get('resources', {})
        }
    
    # 비교 리포트 저장
    with open('benchmark_results/comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 간단한 요약 출력
    print("\n" + "="*60)
    print("벤치마크 비교 요약")
    print("="*60)
    
    for model, data in comparison['models'].items():
        print(f"\n{model}:")
        print(f"  - 성공률: {data['success_rate']*100:.1f}%")
        if 'mean' in data.get('latency', {}):
            print(f"  - 평균 응답 시간: {data['latency']['mean']:.2f}초")
        if 'tokens_per_second' in data.get('tokens', {}):
            tps = data['tokens']['tokens_per_second'].get('mean', 0)
            print(f"  - 평균 TPS: {tps:.2f}")
        if 'resources' in data:
            # Python 프로세스 리소스
            if 'python_process' in data['resources']:
                py_cpu = data['resources']['python_process'].get('cpu_percent', {}).get('mean', 0)
                py_mem = data['resources']['python_process'].get('memory_mb', {}).get('mean', 0)
                print(f"  - Python CPU 사용률: {py_cpu:.2f}%")
                print(f"  - Python 메모리: {py_mem:.0f}MB")
            
            # Ollama 프로세스 리소스
            if 'ollama_process' in data['resources']:
                ollama_cpu = data['resources']['ollama_process'].get('cpu_percent', {}).get('mean', 0)
                ollama_mem = data['resources']['ollama_process'].get('memory_mb', {}).get('mean', 0)
                print(f"  - Ollama CPU 사용률: {ollama_cpu:.2f}%")
                print(f"  - Ollama 메모리 (LLM): {ollama_mem:.0f}MB")
            
            # 총 메모리
            if 'total_memory_mb' in data['resources']:
                total_mem = data['resources']['total_memory_mb'].get('mean', 0)
                print(f"  - 총 메모리 사용량: {total_mem:.0f}MB")
            
            # 기존 형식 호환성
            elif 'cpu_percent' in data['resources']:
                cpu = data['resources'].get('cpu_percent', {}).get('mean', 0)
                mem = data['resources'].get('memory_mb', {}).get('mean', 0)
                print(f"  - CPU 사용률: {cpu:.2f}%")
                print(f"  - 메모리 사용량: {mem:.0f}MB")

def main():
    print("모든 모델 벤치마크 시작")
    print(f"테스트할 모델: {', '.join(MODELS)}")
    
    success_count = 0
    for model in MODELS:
        if run_benchmark(model):
            success_count += 1
        # 모델 간 쿨다운
        time.sleep(5)
    
    print(f"\n벤치마크 완료: {success_count}/{len(MODELS)} 성공")
    
    # 비교 리포트 생성
    create_comparison_report()

if __name__ == "__main__":
    main()