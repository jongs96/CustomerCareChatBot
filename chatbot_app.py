import streamlit as st # Streamlit: 웹 UI를 만드는 라이브러리
# LangChain 관련 모듈: LLM, 텍스트 분할, 벡터 저장소, 문서 로더
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# .env 파일에서 API 키 로드
from dotenv import load_dotenv
import os
import pickle  # FAISS 인덱스 저장용

# .env 파일 로드
load_dotenv()
# Open AI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FAQ 데이터 로드 및 RAG 설정
def setup_rag():
    # FAISS 인덱스 파일 경로
    faiss_index_path = "C:/Users/flux310/Desktop/Js/CustomerCareBot/faiss_index.pkl"
    # 텍스트를 벡터로 변환 (=텍스트를 숫자 리스트로 바꿈), OpenAI 임베딩 사용
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # FAISS 인덱스가 이미 존재하면 로드
    if os.path.exists(faiss_index_path):
        with open(faiss_index_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        # faq.txt 파일을 절대 경로로 로드, UTF-8 인코딩 명시
        loader = TextLoader("C:/Users/flux310/Desktop/Js/CustomerCareBot/faq.txt", encoding='utf-8')
        documents = loader.load()
        # 텍스트를 작은 조각으로 나누기 (검색 효율성 향상, chunk_size 축소)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        # FAISS로 벡터 저장소 생성 (빠른 검색 가능)
        vectorstore = FAISS.from_documents(texts, embeddings)
        # FAISS 인덱스 저장
        with open(faiss_index_path, "wb") as f:
            pickle.dump(vectorstore, f)

    # Open AI LLM 설정 (GPT-4o-mini 사용, temperature=0은 정확한 답변 선호)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # 대화 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # ConversationalRetrievalChain: FAQ 검색 + LLM + 대화 메모리
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return qa_chain

# 마크다운으로 답변 포맷팅
def format_response(response):
    # FAQ 답변을 마크다운으로 구조화
    formatted = f"**답변**\n\n{response}\n\n*자세한 문의는 고객 지원팀(1234-5678)으로 연락 주세요.*"
    return formatted

# Streamlit UI 설정
st.title("고객 불만 처리 챗봇")
st.write("불만 사항을 입력하세요:")

# 대화 기록 저장
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# RAG 체인 초기화(페이지 로드 시 한 번만 실행)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = setup_rag()

# 대화 기록 표시
with st.container(height=400):
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            if sender == "챗봇":
                st.markdown(format_response(message)) # 챗봇 응답을 마크다운으로 포맷팅
            else:
                st.write(message)

# 입력 폼(엔터 시 입력창 초기화)
with st.form(key="chat_form", clear_on_submit=True):
    # 사용자 입력
    user_input = st.text_input("입력", placeholder="질문을 입력하세요.", key="user_input")
    # 제출 버튼
    submit_button = st.form_submit_button(label="전송")

# 응답 출력
if submit_button and user_input:
    # RAG 체인으로 FAQ 기반 답변 생성 ( 대화 기록 포함 )
    response = st.session_state.qa_chain({"question": user_input})
    # 대화 기록에 추가
    st.session_state.chat_history.append(("사용자", user_input))
    st.session_state.chat_history.append(("챗봇", response["answer"]))
    # 페이지 새로고침으로 대화 즉시 표시
    st.rerun()