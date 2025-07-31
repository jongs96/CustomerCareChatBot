import streamlit as st
# LangChain 관련 모듈: LLM, 텍스트 분할, 벡터 저장소, 문서 로더
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# .env 파일에서 API 키 로드
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
# Open AI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FAQ 데이터 로드 및 RAG 설정
def setup_rag():
    # faq.txt 파일을 로드
    loader = TextLoader("faq.txt")
    documents = loader.load()
    # 텍스트를 작은 조각으로 나누기 (검색 효율성 향상)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # HuggingFace 모델로 텍스트를 벡터로 변환
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # FAISS로 벡터 저장소 생성
    vectorstore = FAISS.from_documents(texts, embeddings)
    # Open AI LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # RetrievalQA 체인: FAQ 검색 + LLM 답변 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# Streamlit UI 설정
st.title("고객 불만 처리 챗봇")
st.write("불만 사항을 입력하세요:")

# RAG 체인 초기화
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = setup_rag()

# 사용자 입력
user_input = st.text_input("입력", key="user_input")

# 응답 출력
if user_input:
    # RAG 체인으로 답변 생성
    response = st.session_state.qa_chain.run(user_input)
    st.write("챗봇 응답:", response)