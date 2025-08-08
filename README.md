# 의료 상담 요약 AI 시스템

## 프로젝트 개요
의사-환자 간 대화 내용을 분석하여 자동으로 상담 요약을 생성하는 AI 시스템입니다. LangChain과 Ollama를 활용하여 로컬 환경에서 빠르게 프로토타입을 개발하고 검증합니다.

## 주요 기능
- 의사-환자 대화 내용 자동 분석
- 주요 증상, 진단, 처방 정보 추출
- 구조화된 상담 요약 생성
- RAG를 통한 의학 정보 기반 보강

## 기술 스택
- **LLM 프레임워크**: LangChain
- **모델 실행**: Ollama (로컬 LLM)
- **벡터 DB**: ChromaDB 또는 FAISS
- **프로그래밍 언어**: Python 3.10+
- **웹 프레임워크**: FastAPI (선택사항)

## 시스템 아키텍처
```
[의사-환자 대화] → [전처리] → [LLM + RAG] → [상담 요약]
                                    ↑
                              [의학 정보 DB]
```

## 설치 및 환경 설정

### 1. 필수 도구 설치
```bash
# Ollama 설치 (Mac)
brew install ollama

# Ollama 설치 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요 패키지 설치
pip install langchain langchain-community ollama chromadb pandas numpy transformers
```

### 2. Ollama 모델 다운로드
```bash
# 한국어 지원 모델 추천
ollama pull llama2:13b-chat
# 또는
ollama pull mistral:7b
# 또는 한국어 특화 모델
ollama pull beomi/llama-2-ko-7b
```

## 프로젝트 구조
```
doctor-note/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # 원본 의학 정보 데이터
│   ├── processed/           # 전처리된 데이터
│   └── sample_dialogues/    # 테스트용 대화 샘플
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # 데이터 전처리
│   ├── embeddings.py        # 임베딩 생성
│   ├── rag_engine.py        # RAG 시스템
│   ├── summarizer.py        # 요약 생성
│   └── evaluator.py         # 성능 평가
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_setup.ipynb
│   └── 03_model_evaluation.ipynb
└── tests/
    └── test_summarizer.py
```

## 개발 단계

### Phase 1: 데이터 준비 (1-2일)
1. 의학 정보 5000건 전처리
   - 형식 통일 (JSON/CSV)
   - 메타데이터 추가 (카테고리, 질병코드 등)
   - 품질 검증

2. 샘플 대화 데이터 준비
   - 실제 상담 대화 형식으로 변환
   - 정답 요약문 작성 (평가용)

### Phase 2: RAG 시스템 구축 (2-3일)
1. 벡터 DB 구축
```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 한국어 임베딩 모델 사용
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)

# 벡터 DB 생성
vectorstore = Chroma.from_documents(
    documents=medical_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

2. RAG 파이프라인 구성
```python
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# LLM 설정
llm = Ollama(model="llama2:13b-chat")

# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(k=5)
)
```

### Phase 3: 상담 요약 모델 개발 (3-4일)
1. 프롬프트 엔지니어링
```python
summary_prompt = """
다음은 의사와 환자의 대화 내용입니다:
{dialogue}

관련 의학 정보:
{medical_context}

위 대화를 바탕으로 다음 항목을 포함한 상담 요약을 작성하세요:
1. 주요 증상
2. 추정 진단
3. 권고 사항
4. 추가 검사 필요 여부

요약:
"""
```

2. 요약 생성 파이프라인
```python
def generate_summary(dialogue):
    # 대화에서 핵심 키워드 추출
    keywords = extract_keywords(dialogue)
    
    # RAG로 관련 의학 정보 검색
    medical_context = qa_chain.run(keywords)
    
    # 요약 생성
    summary = llm.generate(
        summary_prompt.format(
            dialogue=dialogue,
            medical_context=medical_context
        )
    )
    return summary
```

### Phase 4: 평가 및 개선 (2-3일)
1. 평가 지표 설정
   - ROUGE 점수
   - 의학 용어 정확도
   - 구조 완성도
   - 인간 평가 (의료진 검토)

2. 모델 튜닝 방법
   - 프롬프트 최적화
   - Few-shot 예제 추가
   - LoRA 파인튜닝 (선택사항)

## 빠른 시작 가이드

### 1. 기본 요약 테스트
```python
from src.summarizer import MedicalSummarizer

# 요약기 초기화
summarizer = MedicalSummarizer()

# 샘플 대화
dialogue = """
의사: 어떤 증상으로 오셨나요?
환자: 3일 전부터 두통이 심하고 열이 나요.
의사: 열은 몇 도까지 올라가나요?
환자: 38.5도까지 올라갔어요.
의사: 다른 증상은 없나요?
환자: 목도 좀 아프고 기침도 나요.
"""

# 요약 생성
summary = summarizer.summarize(dialogue)
print(summary)
```

### 2. RAG 성능 테스트
```python
# 의학 정보 검색 테스트
query = "두통 발열 인후통 증상"
results = vectorstore.similarity_search(query, k=3)
for doc in results:
    print(doc.page_content[:200])
```

## 성능 최적화 팁

1. **모델 선택**
   - 빠른 추론: Mistral 7B
   - 높은 정확도: Llama2 13B
   - 한국어 특화: beomi/llama-2-ko

2. **RAG 최적화**
   - Chunk 크기: 200-500 토큰
   - Overlap: 20-50 토큰
   - Top-k: 3-5개

3. **프롬프트 최적화**
   - 구조화된 출력 형식 정의
   - Few-shot 예제 2-3개 포함
   - 의학 용어 사전 제공

## 예상 결과물

### 입력 (대화)
```
의사: 어떤 증상으로 오셨나요?
환자: 3일 전부터 두통이 심하고 열이 나요...
```

### 출력 (요약)
```
[상담 요약]
- 주요 증상: 두통 (3일), 발열 (38.5°C), 인후통, 기침
- 추정 진단: 급성 상기도 감염
- 권고 사항: 충분한 휴식, 수분 섭취, 해열제 복용
- 추가 검사: 증상 지속 시 인플루엔자 검사 권장
```

## 한계점 및 주의사항
- 의료 진단 목적으로 사용 불가 (보조 도구로만 활용)
- 개인정보 보호 필요 (로컬 실행 권장)
- 의학 전문가의 검증 필수
- 지속적인 데이터 업데이트 필요

## 향후 개선 방향
1. 더 많은 의학 데이터 수집 및 학습
2. 다양한 진료과목별 특화 모델 개발
3. 실시간 음성 인식 연동
4. 의료 영상 분석 기능 추가
5. 클라우드 배포 및 API 서비스화

## 참고 자료
- [LangChain 문서](https://python.langchain.com/)
- [Ollama 공식 사이트](https://ollama.ai/)
- [한국어 임베딩 모델](https://huggingface.co/jhgan/ko-sroberta-multitask)
- [의료 NLP 가이드라인](https://github.com/medical-nlp-kr)

## 라이선스
MIT License (연구 및 교육 목적)

---
**주의**: 이 시스템은 연구 및 교육 목적으로만 사용해야 하며, 실제 의료 진단이나 처방에 사용해서는 안 됩니다.