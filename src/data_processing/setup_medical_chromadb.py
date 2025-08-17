import os
import pickle

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm


class KorMedChromaProcessor:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask"):
        """
        한국어 의학 데이터 ChromaDB 처리기

        Args:
            model_name: 한국어 임베딩 모델 (기본값: ko-sroberta-multitask)
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.persist_directory = "data/embeddings/chroma_medical_db"
        self.collection_name = "korean_medical_qa"

        # ChromaDB 클라이언트 설정
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )

    def load_kormed_data(self):
        """KorMedMCQA 데이터셋 로드"""
        print("KorMedMCQA 데이터셋 로드 중...")

        datasets = {}
        categories = ["doctor", "nurse", "pharm", "dentist"]

        for category in categories:
            print(f"  - {category} 데이터 로드...")
            dataset = load_dataset("sean0042/KorMedMCQA", name=category)
            datasets[category] = dataset

        return datasets

    def create_documents(self, datasets):
        """
        의학 질문과 답변을 Document 객체로 변환
        """
        documents = []

        for category, dataset in datasets.items():
            for split in ["train", "dev", "test"]:
                if split in dataset:
                    data = dataset[split]

                    for item in tqdm(data, desc=f"{category}-{split} 처리"):
                        # 질문 텍스트
                        question_text = item["question"]

                        # 정답 선택지 추출
                        answer_idx = int(item["answer"]) - 1
                        answer_options = ["A", "B", "C", "D", "E"]
                        correct_answer = item[answer_options[answer_idx]]

                        # 전체 문맥을 포함한 텍스트 생성
                        full_text = f"질문: {question_text}\n정답: {correct_answer}"

                        # Chain of Thought가 있는 경우 추가
                        if "cot" in item and item["cot"]:
                            full_text += f"\n설명: {item['cot']}"

                        # 메타데이터
                        metadata = {
                            "category": category,
                            "year": str(item["year"]),
                            "subject": item["subject"],
                            "q_number": str(item["q_number"]),
                            "split": split,
                            "period": str(item.get("period", "")),
                            "has_cot": bool(item.get("cot")),
                        }

                        # Document 생성
                        doc = Document(page_content=full_text, metadata=metadata)
                        documents.append(doc)

                        # 각 선택지도 별도 문서로 저장 (문맥 확장)
                        for opt in answer_options:
                            if item[opt]:
                                option_text = f"{category} 관련 의학 정보: {item[opt]}"
                                option_doc = Document(
                                    page_content=option_text,
                                    metadata={
                                        "category": category,
                                        "type": "option",
                                        "related_question": str(item["q_number"]),
                                        "year": str(item["year"]),
                                    },
                                )
                                documents.append(option_doc)

        return documents

    def setup_chromadb(self, documents, batch_size=500):
        """ChromaDB에 문서 저장"""
        print(f"\n총 {len(documents)}개의 문서를 ChromaDB에 저장 중...")

        # 기존 컬렉션 삭제 (있는 경우)
        try:
            self.client.delete_collection(self.collection_name)
            print("기존 컬렉션 삭제됨")
        except Exception:
            pass

        # Chroma 벡터스토어 생성
        vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        # 배치로 나눠서 추가 (메모리 효율성)
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i : i + batch_size]
            vectorstore.add_documents(batch)

        print(f"ChromaDB 저장 완료: {self.persist_directory}")
        return vectorstore

    def process_all(self):
        """전체 처리 파이프라인"""
        # 1. 데이터 로드
        datasets = self.load_kormed_data()

        # 2. Document 객체 생성
        print("\n의학 문서 생성 중...")
        documents = self.create_documents(datasets)
        print(f"총 {len(documents)}개의 문서 생성 완료")

        # 3. ChromaDB에 저장
        vectorstore = self.setup_chromadb(documents)

        # 4. 메타데이터 저장
        metadata_info = {
            "total_documents": len(documents),
            "categories": list(datasets.keys()),
            "model_name": self.embeddings.model_name,
        }

        with open(os.path.join(self.persist_directory, "metadata_info.pkl"), "wb") as f:
            pickle.dump(metadata_info, f)

        return vectorstore


class MedicalChromaSearch:
    def __init__(
        self,
        persist_directory="chroma_medical_db",
        model_name="jhgan/ko-sroberta-multitask",
    ):
        """의학 검색 엔진"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.persist_directory = persist_directory
        self.collection_name = "korean_medical_qa"

        # ChromaDB 클라이언트 - 기존 설정과 동일하게
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Chroma 벡터스토어 로드
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def search(self, query, k=5, filter_dict=None):
        """
        쿼리 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_dict: 메타데이터 필터 (예: {"category": "doctor"})
        """
        if filter_dict:
            results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append(
                {
                    "rank": i + 1,
                    "score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        return formatted_results

    def search_by_category(self, query, category, k=5):
        """특정 카테고리 내에서 검색"""
        return self.search(query, k=k, filter_dict={"category": category})


if __name__ == "__main__":
    # 데이터 처리 및 ChromaDB 구축
    processor = KorMedChromaProcessor()
    vectorstore = processor.process_all()

    print("\n=== ChromaDB 검색 엔진 테스트 ===")
    search_engine = MedicalChromaSearch()

    # 테스트 쿼리
    test_queries = ["고혈압 치료", "당뇨병 약물", "심장 질환 진단", "폐렴 증상"]

    for query in test_queries:
        print(f"\n쿼리: {query}")
        results = search_engine.search(query, k=3)
        for result in results:
            print(f"  [{result['rank']}] (점수: {result['score']:.3f})")
            print(f"      {result['content'][:150]}...")
            print(f"      카테고리: {result['metadata'].get('category', 'N/A')}")
            print(f"      연도: {result['metadata'].get('year', 'N/A')}")

    # 카테고리별 검색 테스트
    print("\n=== 의사 카테고리 검색 ===")
    doctor_results = search_engine.search_by_category("심장병", "doctor", k=3)
    for result in doctor_results:
        print(f"  {result['content'][:100]}...")
