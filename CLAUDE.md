# Claude AI 작업 지침서

이 문서는 Claude가 이 프로젝트에서 작업할 때 따라야 할 지침을 정의합니다.

## 🚀 세션 시작 시 필수 작업

### 1. README.md 읽기
```bash
# 새 세션 시작 시 항상 실행
cat README.md
```
- 프로젝트 현황 파악
- 최신 업데이트 확인
- 개발 로드맵 확인

### 2. 가상환경 확인 및 활성화
```bash
# 가상환경 상태 확인
which python

# 필요시 활성화
source venv/bin/activate

# 패키지 설치 (requirements.txt 변경 시)
pip install -r requirements.txt
```

## 📝 코드 작업 규칙

### 1. 파일 구조 준수
- **테스트 파일**: 항상 `tests/` 디렉토리에 생성
  ```
  tests/
  ├── test_*.py          # 유닛 테스트
  ├── integration/       # 통합 테스트
  └── fixtures/          # 테스트 데이터
  ```

- **새 기능 추가 시**:
  1. 기능 코드는 `src/` 아래 적절한 위치에
  2. 테스트 코드는 `tests/` 아래 동일한 구조로
  3. 문서 업데이트는 README.md에

### 2. 코드 스타일
- 타입 힌트 사용
- docstring 작성 (Google 스타일)
- 함수명은 snake_case
- 클래스명은 PascalCase
- 상수는 UPPER_SNAKE_CASE

## 🔄 Git 커밋 규칙

### 1. 커밋 전 체크리스트
```bash
# 1. README.md 업데이트
- [ ] 새 기능 추가 시 기능 목록 업데이트
- [ ] API 변경 시 사용법 섹션 업데이트
- [ ] 성능 개선 시 벤치마크 결과 업데이트
- [ ] 로드맵 진행 상황 업데이트

# 2. 테스트 실행
python -m pytest tests/

# 3. 코드 포맷팅 (선택사항)
black src/ tests/
isort src/ tests/
```

### 2. 커밋 메시지 형식
```
<type>: <subject>

<body>

<footer>
```

**Type**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드, 패키지 매니저 등

**예시**:
```bash
git add .
git commit -m "feat: 하이브리드 RAG 시스템 구현

- BM25와 Dense 검색 결합
- 의학 도메인 특화 토크나이저 추가
- Alpha 파라미터로 가중치 조절 가능

🤖 Generated with Claude
Co-Authored-By: Claude <noreply@anthropic.com>"
```

## 🧪 테스트 작성 규칙

### 1. 테스트 파일 명명
```python
# 기능 코드
src/models/hybrid_rag_system.py

# 테스트 코드
tests/models/test_hybrid_rag_system.py
```

### 2. 테스트 구조
```python
import pytest
from src.models.hybrid_rag_system import HybridMedicalRAG

class TestHybridMedicalRAG:
    """하이브리드 RAG 시스템 테스트"""
    
    @pytest.fixture
    def rag_system(self):
        """테스트용 RAG 시스템 인스턴스"""
        return HybridMedicalRAG(alpha=0.5)
    
    def test_search_accuracy(self, rag_system):
        """검색 정확도 테스트"""
        results = rag_system.hybrid_search("두통", top_k=5)
        assert len(results) <= 5
        assert all('score' in r for r in results)
```

## 📦 패키지 관리

### 1. 새 패키지 추가 시
```bash
# 1. 패키지 설치
pip install package-name

# 2. requirements.txt 업데이트
pip freeze > requirements.txt

# 3. README.md 업데이트 (필요시)
- 기술 스택 섹션
- 설치 방법 섹션
```

### 2. 패키지 버전 관리
- 메이저 버전은 고정 (예: `langchain==0.3.21`)
- 보안 업데이트는 즉시 적용
- 호환성 테스트 후 업데이트

## 🔍 작업 시작 전 확인 사항

```python
# 체크리스트
checklist = {
    "README_읽기": False,
    "가상환경_활성화": False,
    "브랜치_확인": False,
    "최신_코드_pull": False,
    "의존성_설치": False
}

# 모든 항목이 True가 되어야 작업 시작
```

## 💡 프로젝트별 특별 지침

### 1. 의학 데이터 처리
- 의학 용어는 항상 표준 용어 사전 참조
- ICD-10 코드는 공식 분류 체계 준수
- 민감한 의료 정보는 절대 로그에 남기지 않음

### 2. 모델 벤치마크
- 새 모델 추가 시 반드시 벤치마크 실행
- 결과는 `benchmark_results/` 디렉토리에 저장
- README.md의 성능 비교 섹션 업데이트

### 3. RAG 시스템
- ChromaDB 인덱스 변경 시 버전 관리
- 임베딩 모델 변경 시 전체 재인덱싱
- 검색 성능 저하 시 청크 크기 조정

## 🚨 주의사항

1. **절대 하지 말아야 할 것**:
   - 환자 개인정보를 코드에 하드코딩
   - 의료 진단을 확정적으로 표현
   - 테스트 없이 main 브랜치에 푸시

2. **항상 해야 할 것**:
   - 의료 면책 조항 포함
   - 프로토타입/실제 데이터 구분 명시
   - 에러 처리 및 로깅

## 📊 작업 우선순위

1. **긴급**: 보안 이슈, 크리티컬 버그
2. **높음**: 사용자 요청 기능, 성능 개선
3. **중간**: 코드 리팩토링, 문서 개선
4. **낮음**: 코드 스타일, 주석 추가

## 🔄 정기 업데이트

### 매 작업 시
- README.md 현황 업데이트
- 테스트 코드 작성/실행
- 커밋 메시지 규칙 준수

### 주요 기능 완료 시
- 벤치마크 실행 및 결과 업데이트
- API 문서 업데이트
- 로드맵 진행 상황 업데이트

---

**Note**: 이 문서는 Claude가 프로젝트 작업 시 자동으로 참조합니다.
새로운 규칙이 필요하면 이 문서를 업데이트해주세요.