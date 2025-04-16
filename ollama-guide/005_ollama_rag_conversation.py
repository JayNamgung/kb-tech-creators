# ollama-rag를 사용하여 한국어 문서를 검색하는 프로그램

import ollama
import chromadb
from chromadb.utils import embedding_functions

# 모델 선택
model = "deepcoder"

# 한글 응답 강화를 위한 시스템 프롬프트
system_prompt = """당신은 매우 유능한 한국어 AI 도우미입니다. 
항상 한국어로 자연스럽게 답변하며, 한국어 맞춤법과 문법을 정확하게 사용합니다.
전문적인 내용이라도 한국인이 이해하기 쉽게 설명해주세요.
영어 단어는 필요한 경우에만 사용하고, 가능한 한국어로 설명합니다.
주어진 문서 정보를 기반으로 정확하게 답변하세요."""

# ChromaDB 설정
client = chromadb.Client()
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name=model)
collection = client.create_collection("korean_docs", embedding_function=ollama_ef)

# 회사 내부 문서라고 가정한 데이터 추가
documents = [
    "ABC 회사는 2023년 3월에 클라우드 기반 AI 서비스 '스마트에이전트'를 출시했습니다.",
    "ABC 회사의 직원 복지 제도에는 유연근무제, 연 20일의 휴가, 건강검진 지원이 포함됩니다.",
    "ABC 회사의 주요 경쟁사는 XYZ 테크놀로지와 DEF 소프트웨어입니다.",
]

# 문서 추가
collection.add(documents=documents, ids=["product", "welfare", "competitor"])


# 멀티턴 RAG 채팅 구현
class RAGChatbot:
    def __init__(self, model_name, collection):
        self.model = model_name
        self.collection = collection
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, question):
        # 관련 문서 검색
        results = self.collection.query(query_texts=[question], n_results=2)
        context = "\n".join(results["documents"][0])

        # 검색 결과와 함께 질문 추가
        self.messages.append(
            {
                "role": "user",
                "content": f"""다음 정보를 참고하여 질문에 답해주세요:
            
{context}

질문: {question}""",
            }
        )

        # 모델에 질의
        response = ollama.chat(model=self.model, messages=self.messages)

        # 응답 저장
        self.messages.append(response["message"])

        return response["message"]["content"]


# 챗봇 초기화
chatbot = RAGChatbot(model, collection)

# 첫 번째 질문
question1 = "이 회사의 복지는 무엇인가요?"
answer1 = chatbot.ask(question1)
print("질문:", question1)
print("AI:", answer1)

# 두 번째 질문
question2 = "이 회사의 주요 경쟁사는 어디인가요?"
answer2 = chatbot.ask(question2)
print("질문:", question2)
print("AI:", answer2)

# 세 번째 질문
question3 = "이 회사가 출시한 새로운 서비스는 무엇인가요?"
answer3 = chatbot.ask(question3)
print("질문:", question3)
print("AI:", answer3)
