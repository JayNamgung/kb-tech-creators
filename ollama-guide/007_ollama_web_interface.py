# ollama 웹 인터페이스 샘플

import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions
import os
import tempfile

st.set_page_config(page_title="로컬 문서 기반 챗봇", layout="wide")
st.title("Ollama 문서 기반 질의응답 시스템")

# 한글 응답 강화를 위한 시스템 프롬프트
system_prompt = """당신은 매우 유능한 한국어 AI 도우미입니다. 
항상 한국어로 자연스럽게 답변하며, 한국어 맞춤법과 문법을 정확하게 사용합니다.
전문적인 내용이라도 한국인이 이해하기 쉽게 설명해주세요.
영어 단어는 필요한 경우에만 사용하고, 가능한 한국어로 설명합니다.
주어진 문서 정보를 기반으로 정확하게 답변하세요."""

# 세션 상태 초기화
if "collection" not in st.session_state:
    st.session_state.collection = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사이드바에 모델 선택 옵션
with st.sidebar:
    st.header("설정")
    model = st.selectbox("Ollama 모델 선택", ["llama2", "deepcoder"])
    chunk_size = st.slider(
        "청크 크기 설정",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="텍스트를 나눌 청크의 크기. 작을수록 정확한 검색, 클수록 더 많은 컨텍스트 제공",
    )
    num_chunks = st.slider(
        "검색할 청크 수",
        min_value=1,
        max_value=5,
        value=2,
        help="질문에 답변할 때 검색할 관련 청크 수",
    )

    st.markdown("---")
    st.markdown("### 📌 사용 방법")
    st.markdown("1. 텍스트 파일을 업로드합니다.")
    st.markdown("2. 문서 처리가 완료될 때까지 기다립니다.")
    st.markdown("3. 질문을 입력하고 답변을 받습니다.")
    st.markdown("4. 새 문서로 시작하려면 '새 문서 업로드' 버튼을 클릭합니다.")

# 파일 업로드 섹션
st.header("1. 문서 업로드")

uploaded_file = st.file_uploader(
    "텍스트 파일을 업로드하세요", type=["txt"], key="file_uploader"
)


# ChromaDB 초기화 함수
def initialize_chromadb(model_name):
    client = chromadb.Client()
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name=model_name)
    # 이미 있는 컬렉션은 삭제
    try:
        client.delete_collection("uploaded_docs")
    except:
        pass

    # 새 컬렉션 생성
    collection = client.create_collection("uploaded_docs", embedding_function=ollama_ef)
    return collection


# 텍스트 파일 처리 함수
def process_file(file, chunk_size):
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    # 파일 읽기 및 청킹
    with open(tmp_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 임시 파일 삭제
    os.unlink(tmp_path)

    # 청크 분할
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    return chunks


# 문서 처리 버튼
if uploaded_file is not None and not st.session_state.documents_loaded:
    if st.button("문서 처리 시작"):
        with st.spinner("문서를 처리 중입니다..."):
            # ChromaDB 초기화
            st.session_state.collection = initialize_chromadb(model)

            # 파일 처리
            chunks = process_file(uploaded_file, chunk_size)

            # 청크 ID 생성
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

            # ChromaDB에 추가
            st.session_state.collection.add(documents=chunks, ids=chunk_ids)

            st.session_state.documents_loaded = True
            st.success(f"문서를 {len(chunks)}개 청크로 분할하여 처리 완료했습니다!")

# 새 문서 업로드 옵션
if st.session_state.documents_loaded:
    if st.button("새 문서 업로드"):
        st.session_state.documents_loaded = False
        st.session_state.chat_history = []
        st.experimental_rerun()

# 질문-답변 섹션
st.header("2. 질문하기")

if st.session_state.documents_loaded:
    # 질문 입력 필드
    user_question = st.text_input("질문을 입력하세요:", key="user_question")

    if user_question and st.button("답변 받기"):
        with st.spinner("답변을 생성하는 중..."):
            # 문서 검색
            results = st.session_state.collection.query(
                query_texts=[user_question], n_results=num_chunks
            )
            context = "\n".join(results["documents"][0])

            # 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""다음은 문서의 일부입니다:
                
{context}

위 정보만을 사용하여 다음 질문에 답하세요:
질문: {user_question}""",
                },
            ]

            # Ollama로 답변 생성
            response = ollama.chat(model=model, messages=messages)
            answer = response["message"]["content"]

            # 채팅 기록에 추가
            st.session_state.chat_history.append((user_question, answer))

    # 채팅 기록 표시
    if st.session_state.chat_history:
        st.header("대화 기록")
        for q, a in st.session_state.chat_history:
            st.markdown(f"#### Q: {q}")
            st.markdown(f"{a}")
            st.markdown("---")

        # 검색된 문서 조각 표시 (선택적)
        with st.expander("참조된 문서 조각 보기"):
            st.markdown(f"**최근 질문 '{user_question}'와 관련된 문서 조각:**")
            st.text(context)
else:
    st.info("먼저 문서를 업로드하고 처리해주세요.")

# 페이지 하단 정보
st.markdown("---")
st.caption(
    "이 애플리케이션은 Ollama와 ChromaDB를 사용하여 로컬에서 실행되는 문서 기반 질의응답 시스템입니다."
)
st.caption("모든 처리는 로컬에서 이루어지며 데이터가 외부로 전송되지 않습니다.")
