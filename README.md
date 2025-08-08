# 의료 노트 분석 시스템

로컬 LLM(Ollama)과 벡터 데이터베이스(ChromaDB)를 활용한 한국어 의료 노트 분석 및 RAG(Retrieval-Augmented Generation) 시스템입니다. 
한국 의료진 국가시험 데이터셋(KorMedMCQA)을 기반으로 구축되었습니다.

## 주요 기능

- 한국어 의료 노트의 자동 분석 및 구조화
- 증상, 진단, 처방 정보 추출
- 7,469개의 한국 의료진 시험 문제 기반 지식베이스
- 의료 지식 기반 질의응답
- 유사 사례 검색
- 의료 문서 요약
- 환자 노트 추가 및 관리

## 기술 스택

- **LLM**: Ollama (로컬 실행)
- **Vector DB**: ChromaDB
- **Embedding**: jhgan/ko-sroberta-multitask (한국어 특화)
- **Framework**: LangChain
- **Language**: Python 3.10+
- **Dataset**: KorMedMCQA (한국 의료진 국가시험)

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/doctor-note.git
cd doctor-note
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
doctor-note/
│
├── data/
│   ├── raw/              # 원본 의료 노트
│   ├── processed/        # 전처리된 데이터
│   └── embeddings/       # 벡터 임베딩
│       └── chroma_medical_db/  # ChromaDB 저장소
│
├── src/
│   ├── data_processing/  # 데이터 전처리
│   │   └── setup_medical_chromadb.py
│   ├── models/          # 모델 및 RAG 시스템
│   │   └── medical_rag_system.py
│   └── utils/           # 유틸리티
│       ├── config.py    # 설정 관리
│       └── logger.py    # 로깅
│
├── configs/            # 설정 파일
├── notebooks/          # 실험 및 분석  
├── tests/             # 테스트 코드
├── main.py            # 메인 엔트리포인트
└── requirements.txt   # 의존성 패키지
```

## 사용 방법

### 1. 초기 설정

```bash
# 가상환경 활성화
source venv/bin/activate

# 의학 지식베이스 구축 (최초 1회)
python main.py --mode setup
```

### 2. Ollama 모델 실행

```bash
# Ollama 설치 (macOS)
brew install ollama

# 한국어 모델 다운로드 및 실행
ollama pull llama2
ollama serve
```

### 3. 의학 정보 검색

```bash
# CLI를 통한 검색
python main.py --mode search --query "고혈압 치료 방법" --k 5

# Python 코드에서 사용
from src.models.medical_rag_system import MedicalRAGSystem

rag = MedicalRAGSystem()
results = rag.search("당뇨병 관리", k=3)
```

### 4. 환자 노트 추가

```bash
# CLI를 통한 추가
python main.py --mode add_note --note "환자 두통 호소" --patient_id "P12345"

# Python 코드에서 사용  
rag.add_patient_note(
    note_text="환자는 최근 두통과 어지러움을 호소",
    patient_id="P12345"
)
```

## 주요 컴포넌트

### 1. 데이터셋
- **KorMedMCQA**: 7,469개의 한국 의료진 국가시험 문제
  - 의사, 간호사, 약사, 치과의사 시험 포함
  - 2012-2024년 기출문제
  - Chain of Thought 추론 포함

### 2. 임베딩 모델
- **jhgan/ko-sroberta-multitask**: 한국어 특화 Sentence-BERT
  - 768차원 벡터 임베딩
  - 의미적 유사도 검색에 최적화

### 3. 벡터 데이터베이스
- **ChromaDB**: 44,814개의 의학 문서 저장
  - 효율적인 유사도 검색
  - 메타데이터 기반 필터링 지원
  - 영구 저장소 지원

### 4. RAG 파이프라인
- 한국어 의학 컨텍스트 검색
- 프롬프트 엔지니어링
- Ollama를 통한 응답 생성

## API 사용 예제

### 검색 API

```python
from src.models.medical_rag_system import MedicalRAGSystem

# 시스템 초기화
rag = MedicalRAGSystem()

# 일반 검색
results = rag.search("폐렴 진단 기준", k=5)

# 카테고리별 검색
doctor_results = rag.search(
    "심장병 치료",
    k=3,
    filter_dict={"category": "doctor"}
)

# 의학적 컨텍스트 생성
context = rag.get_relevant_medical_context("고혈압 관리 방법")
```

### 환자 노트 관리

```python
# 환자 노트 추가
rag.add_patient_note(
    note_text="환자는 3일 전부터 발열과 기침 증상을 보임",
    patient_id="P54321",
    note_metadata={
        "date": "2024-01-15",
        "doctor": "김의사",
        "department": "내과"
    }
)
```

## 성능 최적화

- 청크 크기: 1000자 (오버랩 100자)
- 배치 처리: 500개 문서씩 처리
- ChromaDB 영구 저장소 활용
- 한국어 특화 토크나이저 사용

## 향후 계획

- [ ] 멀티모달 지원 (의료 이미지)
- [ ] 실시간 모니터링
- [ ] API 서버 구축  
- [ ] 웹 인터페이스
- [ ] 더 많은 한국어 의학 데이터셋 통합
- [ ] Fine-tuning된 한국어 의료 LLM 적용

## 문제 해결

### Apple Silicon (M1/M4) 관련 이슈
FAISS 대신 ChromaDB를 사용하여 segmentation fault 문제를 해결했습니다.

### 의존성 설치 실패
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

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