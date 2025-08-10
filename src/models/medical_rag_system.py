import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from tqdm import tqdm

class MedicalRAGSystem:
    def __init__(self, 
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 chroma_persist_dir: str = "data/embeddings/chroma_medical_db"):
        """
        의학 RAG 시스템 (ChromaDB 기반)
        
        Args:
            model_name: 한국어 임베딩 모델
            chroma_persist_dir: ChromaDB 저장 경로
        """
        # 임베딩 모델 설정
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 저장 경로
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = "korean_medical_qa"
        
        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        
        # Vector store
        self.chroma_store = None
        
    def load_korean_medical_data(self):
        """한국 의학 MCQ 데이터 로드 및 ChromaDB에 저장"""
        # 이미 데이터가 있는지 확인
        if os.path.exists(self.chroma_persist_dir):
            print("기존 ChromaDB 데이터 사용")
            return
            
        print("KorMedMCQA 데이터셋 로드 중...")
        datasets = {}
        categories = ["doctor", "nurse", "pharm", "dentist"]
        
        for category in categories:
            print(f"  - {category} 데이터 로드...")
            dataset = load_dataset("sean0042/KorMedMCQA", name=category)
            datasets[category] = dataset
            
        # Document 객체 생성
        documents = []
        for category, dataset in datasets.items():
            for split in ["train", "dev", "test"]:
                if split in dataset:
                    data = dataset[split]
                    
                    for item in tqdm(data, desc=f"{category}-{split} 처리"):
                        # 질문과 정답 추출
                        question_text = item["question"]
                        answer_idx = int(item["answer"]) - 1
                        answer_options = ["A", "B", "C", "D", "E"]
                        correct_answer = item[answer_options[answer_idx]]
                        
                        # 전체 텍스트 생성
                        full_text = f"질문: {question_text}\n정답: {correct_answer}"
                        if "cot" in item and item["cot"]:
                            full_text += f"\n설명: {item['cot']}"
                        
                        # Document 생성
                        metadata = {
                            "category": category,
                            "year": str(item["year"]),
                            "subject": item["subject"],
                            "q_number": str(item["q_number"]),
                            "split": split
                        }
                        
                        doc = Document(page_content=full_text, metadata=metadata)
                        documents.append(doc)
        
        print(f"총 {len(documents)}개의 문서 생성 완료")
        
        # ChromaDB에 저장
        if not self.chroma_store:
            self.setup_chromadb()
            
        print("ChromaDB에 문서 추가 중...")
        batch_size = 500
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            self.chroma_store.add_documents(batch)
            
        print("데이터 로드 완료")
    
    def setup_chromadb(self):
        """ChromaDB 설정 및 초기화"""
        # ChromaDB 클라이언트 설정
        client = chromadb.PersistentClient(
            path=self.chroma_persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # ChromaDB 컬렉션 생성
        self.chroma_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.chroma_persist_dir
        )
        
        print("ChromaDB 설정 완료")
    
    def add_documents(self, documents: List[Document]):
        """문서를 ChromaDB에 추가"""
        if not self.chroma_store:
            self.setup_chromadb()
        
        # 문서 분할
        split_docs = self.text_splitter.split_documents(documents)
        
        # ChromaDB에 추가
        self.chroma_store.add_documents(split_docs)
        
        print(f"{len(split_docs)}개의 문서 청크가 ChromaDB에 추가됨")
    
    def search(self, query: str, k: int = 5, filter_dict: Dict = None):
        """
        ChromaDB를 활용한 의학 정보 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_dict: 메타데이터 필터
        
        Returns:
            검색 결과
        """
        if not self.chroma_store:
            self.setup_chromadb()
            
        if filter_dict:
            results = self.chroma_store.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.chroma_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return formatted_results
    
    def add_patient_note(self, note_text: str, patient_id: str, note_metadata: Dict[str, Any] = None):
        """
        환자 노트를 시스템에 추가
        
        Args:
            note_text: 환자 노트 텍스트
            patient_id: 환자 ID
            note_metadata: 추가 메타데이터
        """
        if not self.chroma_store:
            self.setup_chromadb()
        
        # 메타데이터 준비
        metadata = {
            "patient_id": patient_id,
            "type": "patient_note",
            "source": "clinical"
        }
        if note_metadata:
            metadata.update(note_metadata)
        
        # Document 생성
        doc = Document(page_content=note_text, metadata=metadata)
        
        # ChromaDB에 추가
        self.add_documents([doc])
    
    def get_relevant_medical_context(self, query: str, k: int = 5):
        """
        쿼리와 관련된 의학적 컨텍스트 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 컨텍스트 수
        
        Returns:
            관련 의학 정보
        """
        results = self.search(query, k=k)
        
        # 컨텍스트 구성
        context_parts = []
        for i, result in enumerate(results):
            content = result["content"]
            category = result["metadata"].get("category", "일반")
            
            context_parts.append(
                f"[참고 {i+1} - {category}]\n{content}"
            )
        
        return "\n\n".join(context_parts)


def setup_medical_knowledge_base():
    """의학 지식 베이스 초기 설정"""
    rag_system = MedicalRAGSystem()
    
    # ChromaDB 설정
    rag_system.setup_chromadb()
    
    # 한국 의학 MCQ 데이터 로드
    rag_system.load_korean_medical_data()
    
    # 추가 의학 지식 예시
    sample_medical_docs = [
        Document(
            page_content="고혈압은 수축기 혈압이 140mmHg 이상이거나 이완기 혈압이 90mmHg 이상인 경우를 말합니다. 주요 위험 요인으로는 비만, 흡연, 과도한 염분 섭취, 스트레스 등이 있습니다.",
            metadata={"category": "cardiovascular", "topic": "hypertension"}
        ),
        Document(
            page_content="당뇨병은 인슐린 분비 부족이나 인슐린 저항성으로 인해 혈당이 상승하는 대사 질환입니다. 제1형과 제2형으로 구분되며, 제2형이 전체의 90% 이상을 차지합니다.",
            metadata={"category": "endocrine", "topic": "diabetes"}
        ),
        Document(
            page_content="폐렴은 세균, 바이러스, 곰팡이 등에 의한 폐 실질의 염증입니다. 주요 증상으로는 발열, 기침, 가래, 호흡곤란 등이 있으며, 흉부 X-ray로 진단합니다.",
            metadata={"category": "respiratory", "topic": "pneumonia"}
        )
    ]
    
    # ChromaDB에 추가
    rag_system.add_documents(sample_medical_docs)
    
    return rag_system


if __name__ == "__main__":
    # 시스템 설정
    print("의학 RAG 시스템 초기화 중...")
    rag_system = setup_medical_knowledge_base()
    
    # 테스트
    print("\n=== ChromaDB 검색 테스트 ===")
    test_queries = [
        "고혈압 치료 방법",
        "당뇨병 관리",
        "폐렴 진단 기준"
    ]
    
    for query in test_queries:
        print(f"\n쿼리: {query}")
        results = rag_system.search(query, k=3)
        
        for i, result in enumerate(results):
            print(f"\n[{i+1}] 점수: {result['score']:.3f}")
            print(f"내용: {result['content'][:150]}...")
            print(f"메타데이터: {result['metadata']}")
    
    # 환자 노트 추가 예시
    print("\n=== 환자 노트 추가 ===")
    rag_system.add_patient_note(
        note_text="환자는 최근 두통과 어지러움을 호소하며 내원. 혈압 측정 결과 150/95mmHg로 고혈압 소견 보임.",
        patient_id="P12345",
        note_metadata={
            "date": "2024-01-15",
            "doctor": "김의사"
        }
    )