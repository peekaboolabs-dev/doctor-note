# 의료 상담 요약 AI 시스템

의사-환자 대화를 자동으로 분석하여 구조화된 상담 요약 노트를 생성하는 AI 시스템입니다.
한국 의료진 국가시험 데이터셋(KorMedMCQA) 기반 RAG를 활용하여 정확하고 전문적인 의료 요약을 제공합니다.

## 주요 기능

- **의사-환자 대화 분석**: 실시간 대화 내용을 분석하여 핵심 정보 추출
- **자동 상담 요약**: 증상, 진단, 처방, 주의사항 등을 구조화된 노트로 생성
- **의학 지식 기반 RAG**: 7,469개 한국 의료진 시험 문제로 정확도 향상
- **전문 의학 용어 처리**: 한국어 의학 용어 인식 및 표준화
- **맥락 기반 정보 보강**: RAG를 통한 관련 의학 정보 자동 추가

## 기술 스택

- **LLM**: Solar (Upstage, Ollama 로컬 실행)
- **Vector DB**: ChromaDB
- **Embedding**: jhgan/ko-sroberta-multitask (한국어 특화)
- **Framework**: LangChain
- **Language**: Python 3.10+
- **Dataset**: KorMedMCQA (한국 의료진 국가시험)
- **Container**: Docker & Docker Compose
- **API**: FastAPI (예정)

## 빠른 시작 (Docker)

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/doctor-note.git
cd doctor-note
```

### 2. 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env
```

### 3. Docker 컨테이너 실행

```bash
# 이미지 빌드
make build

# 의학 데이터베이스 초기화 (최초 1회)
make init-db

# 컨테이너 실행
make up
```

### 4. 개발 모드 실행

```bash
# 개발용 이미지 빌드
make build-dev

# 개발 모드로 실행 (핫 리로드 지원)
make up-dev
```

## 로컬 개발 환경 (Docker 없이)

### 1. 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 활성화
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Ollama 설치 및 실행

```bash
# macOS
brew install ollama

# 모델 다운로드
ollama pull solar  # 6GB, 한국어 최적화

# Ollama 서버 실행
ollama serve
```

### 3. 초기 데이터 설정

```bash
python main.py --mode setup
```

## 프로젝트 구조

```
doctor-note/
│
├── data/
│   ├── raw/                      # 원본 의료 노트
│   ├── processed/                # 전처리된 데이터
│   └── embeddings/               # 벡터 임베딩
│       └── chroma_medical_db/    # ChromaDB 저장소
│
├── src/
│   ├── api/                      # API 인터페이스
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic 스키마
│   ├── data_processing/          # 데이터 전처리
│   │   └── setup_medical_chromadb.py
│   ├── models/                   # 모델 및 RAG 시스템
│   │   └── medical_rag_system.py
│   └── utils/                    # 유틸리티
│       ├── config.py            # 설정 관리
│       └── logger.py            # 로깅
│
├── configs/                      # 설정 파일
├── notebooks/                    # 실험 및 분석  
├── tests/                        # 테스트 코드
├── logs/                         # 로그 파일
│
├── Dockerfile                    # 프로덕션 컨테이너
├── Dockerfile.dev               # 개발용 컨테이너
├── docker-compose.yml           # 기본 구성
├── docker-compose.dev.yml       # 개발 환경 오버라이드
├── .dockerignore               # Docker 제외 파일
├── .env.example                # 환경 변수 예제
├── Makefile                    # 자동화 명령어
├── main.py                     # 메인 엔트리포인트
└── requirements.txt            # Python 의존성
```

## 사용 방법

### CLI 사용

```bash
# 의사-환자 대화 요약
docker-compose run --rm medical-rag python main.py \
  --mode summarize \
  --dialogue "의사: 어떤 증상이 있으신가요? 
환자: 3일 전부터 두통이 심하고 어지러워요." \
  --patient_id "P12345"

# RAG 성능 테스트
docker-compose run --rm medical-rag python main.py \
  --mode benchmark \
  --test_file "test_dialogues.json"
```

### Python API 사용

```python
from src.models.dialogue_summarizer import DialogueSummarizer

# 대화 요약 시스템 초기화
summarizer = DialogueSummarizer()

# 의사-환자 대화 요약
dialogue = """
의사: 어떤 증상으로 오셨나요?
환자: 3일 전부터 기침이 심하고 열이 나요.
의사: 열은 몇 도까지 올라갔나요?
환자: 38.5도까지 올라갔어요.
"""

summary = summarizer.summarize_dialogue(dialogue)
print(summary)
# 출력: {"주증상": "기침, 발열", "기간": "3일", "체온": "38.5도", ...}
```

## API 인터페이스 (통합용)

타 개발팀과의 통합을 위한 표준화된 API 스키마가 `src/api/schemas.py`에 정의되어 있습니다.

### 주요 엔드포인트 (백엔드팀 구현 예정)

```python
# 대화 요약 요청
POST /api/v1/summarize
{
    "dialogue": "의사-환자 대화 내용",
    "patient_id": "P12345",
    "session_id": "S67890"
}

# 요약 결과 응답
{
    "summary": {
        "chief_complaint": "기침, 발열",
        "symptoms": ["3일간 지속된 기침", "38.5도 발열"],
        "diagnosis": "상기도 감염 의심",
        "treatment": ["해열제 처방", "충분한 수분 섭취"],
        "follow_up": "증상 지속시 3일 후 재방문"
    },
    "confidence": 0.92,
    "references": ["관련 의학 문헌 정보"]
}
```

## 환경 변수 설정

`.env` 파일에서 다음 설정을 관리할 수 있습니다:

```bash
# 임베딩 모델
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask

# ChromaDB
CHROMA_PERSIST_DIR=/app/data/embeddings/chroma_medical_db

# Ollama (로컬 또는 컨테이너)
OLLAMA_HOST=host.docker.internal
OLLAMA_PORT=11434
OLLAMA_MODEL=solar

# 로깅
LOG_LEVEL=INFO
```

## 개발 가이드

### Makefile 명령어

```bash
make help        # 도움말 표시
make build       # Docker 이미지 빌드
make up          # 컨테이너 시작
make down        # 컨테이너 중지
make logs        # 로그 확인
make shell       # 컨테이너 쉘 접속
make test        # 테스트 실행
make clean       # 볼륨 및 캐시 정리
```

### 디버깅

개발 모드에서는 Python 디버거를 사용할 수 있습니다:

```bash
# VSCode 디버깅 포트: 5678
make up-dev
```

## 최적화 전략

### RAG 최적화
- **임베딩 모델**: jhgan/ko-sroberta-multitask (768차원)
- **벡터 DB**: ChromaDB (44,814개 의학 문서)
- **청크 전략**: 1000자 단위, 100자 오버랩
- **최적화 목표**:
  - 검색 정확도: 의학 용어 매칭률 90% 이상
  - 응답 시간: 2초 이내
  - 컨텍스트 관련성: 상위 5개 중 3개 이상 관련

### 모델 최적화
- **LLM 경량화**:
  - Ollama 지원 양자화 포맷 (GGUF)
  - 4-bit/8-bit 양자화로 메모리 사용량 감소
  - 의학 도메인 LoRA 어댑터 적용
- **임베딩 최적화**:
  - 의학 코퍼스로 도메인 적응
  - PCA/UMAP으로 차원 축소 실험
  - HNSW 인덱스로 검색 속도 개선
- **서빙 최적화**:
  - vLLM/TGI 통합 검토
  - 동적 배치 처리
  - KV 캐시 최적화

## 문제 해결

### Apple Silicon (M1/M4) 호환성
- FAISS 대신 ChromaDB 사용으로 segmentation fault 해결
- ARM64 네이티브 이미지 빌드

### Docker 관련 이슈
```bash
# 권한 문제
sudo usermod -aG docker $USER

# 포트 충돌
docker ps  # 기존 컨테이너 확인
make down  # 정리
```

### Ollama 연결 실패
```bash
# 로컬 Ollama 사용 시
OLLAMA_HOST=host.docker.internal

# Docker Ollama 사용 시  
OLLAMA_HOST=ollama
```

## 개발 현황 및 로드맵

### 진행 상황 (2025-08-09 기준)

#### ✅ 완료된 작업
- [✓] 의사-환자 대휤 분석 모듈 개발
- [✓] 상담 요약 노트 생성 파이프라인
- [✓] KorMedMCQA 데이터셋 통합 (44,814개 의학 문서)
- [✓] Docker 컨테이너화
- [✓] JSON 파일에서 대화 추출 기능
- [✓] Solar 모델로 전환 (Mac M4 호환)

#### 🚧 진행 중
- [ ] Solar 모델 테스트 및 최적화
- [ ] 프롬프트 엔지니어링

#### 📅 예정 작업

**Phase 1: 로컬 최적화 (1주)**
- [ ] RAG 성능 최적화 (청크 크기, 검색 전략)
- [ ] 벤치마크 및 성능 측정 도구
- [ ] 의학 용어 인식 및 표준화

**Phase 2: 모델 추상화 (3-4일)**
- [ ] LLM Provider 추상화 계층 구현
- [ ] 환경별 자동 전환 (Ollama/Transformers)
- [ ] 모델 비교 테스트 (Solar vs Qwen2.5 vs Llama3.2)

**Phase 3: API 개발 (1주)**
- [ ] FastAPI 서버 구현
- [ ] 비동기 처리 및 배치 요청
- [ ] API 문서화 (OpenAPI/Swagger)

**Phase 4: AWS 배포 (3-4일)**
- [ ] Docker 이미지 최적화 (CPU/GPU 버전)
- [ ] EC2 CloudFormation 템플릿
- [ ] CI/CD 파이프라인

### 기술 의사결정 사항

1. **LLM 모델**
   - 현재: Solar (로컬 테스트)
   - AWS: EXAONE/Solar with Transformers
   - 대안: Qwen2.5, Llama3.2

2. **배포 전략**
   - MVP: EC2 t3.large (CPU)
   - 프로덕션: GPU 인스턴스 or SageMaker

3. **모델 서빙**
   - 로컬: Ollama
   - AWS: Transformers + FastAPI
   - 고성능: vLLM/TGI

### 팀 협업 분담

#### RAG 최적화
- [ ] 청크 크기 및 오버랩 최적화
- [ ] 의학 도메인 특화 검색 전략
- [ ] 동적 컨텍스트 검색 알고리즘
- [ ] 하이브리드 검색 (키워드 + 시맨틱)

#### 모델 최적화
- [ ] **LLM 최적화**
  - [ ] 양자화 (GGUF, GPTQ, AWQ)
  - [ ] 한국어 의학 데이터 파인튜닝
  - [ ] LoRA/QLoRA 어댑터 훈련
  - [ ] 프롬프트 최적화 및 Few-shot 학습
- [ ] **임베딩 모델 최적화**
  - [ ] 의학 도메인 특화 임베딩 훈련
  - [ ] 차원 축소 및 인덱싱 최적화
  - [ ] 다국어 의학 용어 지원
- [ ] **서빙 최적화**
  - [ ] 배치 처리 및 스트리밍
  - [ ] 모델 캐싱 전략
  - [ ] GPU 메모리 최적화
  - [ ] 추론 속도 개선

#### 성능 측정
- [ ] 벤치마크 시스템 구축
- [ ] 정확도/속도 트레이드오프 분석
- [ ] A/B 테스트 프레임워크
- [ ] 실시간 모니터링 대시보드

- **RAG & 모델 최적화**: 현재 개발자
- **백엔드 API**: 타 개발팀 (예정)
- **프론트엔드 UI**: 타 개발팀 (예정)

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

MIT License

## 데이터셋 출처

- [KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA): 한국 의료진 국가시험 데이터셋

## 참고자료

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/)
- [Korean Sentence-BERT](https://huggingface.co/jhgan/ko-sroberta-multitask)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)