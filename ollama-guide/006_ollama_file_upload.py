# Ollama를 사용하여 파일 업로드 및 검색

import ollama
import chromadb
from chromadb.utils import embedding_functions
import os

# 한글 응답 강화를 위한 시스템 프롬프트
system_prompt = """당신은 매우 유능한 한국어 AI 도우미입니다. 
항상 한국어로 자연스럽게 답변하며, 한국어 맞춤법과 문법을 정확하게 사용합니다.
전문적인 내용이라도 한국인이 이해하기 쉽게 설명해주세요.
영어 단어는 필요한 경우에만 사용하고, 가능한 한국어로 설명합니다.
주어진 문서 정보를 기반으로 정확하게 답변하세요."""

# 모델 선택
model = "deepcoder"

# ChromaDB 설정
client = chromadb.Client()
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name=model)
collection = client.create_collection("company_docs", embedding_function=ollama_ef)


# 텍스트 파일 로드 및 청크 분할 함수
def load_text_file(file_path, chunk_size=1000):
    """
    텍스트 파일을 읽고 지정된 크기로 청크를 나눕니다.
    대용량 문서를 효율적으로 처리하기 위한 필수 과정입니다.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 문서를 chunk_size 크기로 분할 (벡터 검색 효율화)
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    return chunks


# 파일 경로 설정
file_path = os.path.join(os.path.dirname(__file__), "006_sample_raw_data.txt")

# 문서 로드 및 청크 분할
chunks = load_text_file(file_path)
print(f"문서를 {len(chunks)}개 청크로 분할했습니다.")

# 고유 ID 생성 후 ChromaDB에 추가
chunk_ids = [f"policy_chunk_{i}" for i in range(len(chunks))]

collection.add(documents=chunks, ids=chunk_ids)


# 문서 기반 질의응답
def answer_from_document(question):
    # 1. 질문과 관련된 문서 청크 검색
    results = collection.query(
        query_texts=[question], n_results=2  # 관련성 높은 상위 2개 청크 가져오기
    )
    context = "\n".join(results["documents"][0])

    # 2. 검색된 정보를 바탕으로 답변 생성
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""다음은 문서의 일부입니다:
        
{context}

위 정보만을 사용하여 다음 질문에 답하세요:
질문: {question}""",
        },
    ]

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


# 사용 예시
questions = [
    "회사의 유연근무제는 어떻게 운영되나요?",
    "연차휴가 정책은 어떻게 되나요?",
    "자격증 취득 시 어떤 혜택이 있나요?",
]

for question in questions:
    print("\n질문:", question)
    print("답변:", answer_from_document(question))
