import streamlit as st # Streamlit: 웹 UI를 만드는 라이브러리
from dotenv import load_dotenv # .env 파일에서 API 키 로드
import os
import pickle  # FAISS 인덱스 저장용
from pathlib import Path #상대경로 처리

# LangChain 관련 모듈: LLM, 텍스트 분할, 벡터 저장소, 문서 로더
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# .env 파일 로드/ 환경변수를 로드
load_dotenv()
# 캐시 리소스로 등록하여 1회만 실행하게 구성.
@st.cache_resource

# FAQ 데이터 로드 및 RAG 설정
def setup_rag():
    # FAISS 인덱스 파일 경로
    faiss_path = Path(__file__).parent / "faiss_index.pkl"
    # 텍스트를 벡터로 변환 (=텍스트를 숫자 리스트로 바꿈), OpenAI 임베딩 사용
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # FAISS 인덱스가 이미 존재하면 로드
    if faiss_path.exists():
        with open(faiss_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        # faq.txt 파일을 절대 경로로 로드, UTF-8 인코딩 명시
        loader = TextLoader(Path(__file__).parent / "faq.txt", encoding='utf-8')
        docs = loader.load()
        # 텍스트를 작은 조각으로 나누기 (검색 효율성 향상, chunk_size 축소)
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        # FAISS로 벡터 저장소 생성 (빠른 검색 가능)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # FAISS 인덱스 저장
        with open(faiss_path, "wb") as f:
            pickle.dump(vectorstore, f)

    # Open AI LLM 설정 (GPT-4o-mini 사용, temperature=0은 정확한 답변 선호)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # 대화 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # ConversationalRetrievalChain: FAQ 검색 + LLM + 대화 메모리
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
#?
# 마크다운으로 답변 포맷팅
def format_response(response):
    # FAQ 답변을 마크다운으로 구조화
    formatted = f"**답변**\n\n{response}\n\n*자세한 문의는 고객 지원팀(1234-5678)으로 연락 주세요.*"
    return formatted
#?
# Streamlit UI 설정
st.set_page_config(page_title="고객 불만 처리 챗봇", layout="centered")
st.title("고객 불만 처리 챗봇")

# 최초 실행 시 setup_rag()를 한 번만 호출, 스피너 표시
if "qa_chain" not in st.session_state:
    with st.spinner("⏳ 초기 설정 중… 잠시만 기다려 주세요"):
        st.session_state.qa_chain = setup_rag()

# 채팅 히스토리 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 입력 폼(엔터 시 입력창 초기화)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "질문을 입력하세요.",
        placeholder="예: 환불 정책이 궁금해요.",
        key="user_input"
    )
    submit_button = st.form_submit_button(label="전송")

# 응답 출력
if submit_button and user_input:
    # RAG 체인으로 FAQ 기반 답변 생성 ( 대화 기록 포함 )
    response = st.session_state.qa_chain({"question": user_input})
    answer = response["answer"]
    # 대화 기록에 추가
    st.session_state.chat_history.append(("사용자", user_input))
    st.session_state.chat_history.append(("챗봇", response["answer"]))
    # 페이지 새로고침으로 대화 즉시 표시

# 대화 기록 표시
for speaker, text in st.session_state.chat_history:
    if speaker == "사용자":
        st.markdown(f"**사용자:** {text}")
    else:
        st.markdown(f"**챗봇:** {text}")
