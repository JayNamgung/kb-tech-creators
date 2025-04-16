# ollama-chromadb 연동 rag 기능 샘플 코드

import chromadb
from chromadb.utils import embedding_functions

# ChromaDB 클라이언트 생성
client = chromadb.Client()

# Ollama 임베딩 함수 정의
model = "deepcoder"
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name=model)

# 컬렉션 생성
collection = client.create_collection("internal_docs", embedding_function=ollama_ef)

# 회사 내부 비공개 데이터 추가
collection.add(
    documents=[
        "ABC전자 스마트홈 제품은 2024년 2분기에 글로벌 출시 예정이며 현재 90% 개발 완료되었습니다.",
        "ABC전자 직원 복지제도가 개편되어 2024년 5월부터 주 4일 근무제가 시범 도입됩니다.",
        "ABC전자의 2023년 4분기 영업이익은 전년 대비 15% 증가한 3,750억원을 기록했습니다.",
    ],
    ids=["product_roadmap", "hr_policy", "financial_report"],
)

# 의미 기반 검색 실행
results = collection.query(
    query_texts=["ABC전자의 근무제도가 어떻게 바뀌나요?"], n_results=1
)

print("검색 결과:", results["documents"][0])
