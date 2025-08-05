# chatbot_app.py
"""
정책문서 기반 고객 불만 자동 분류 & 대응 챗봇

• policy_docs/ 폴더의 .txt/.pdf 문서를 읽어
  변경 감지 시마다 FAISS 인덱스 디렉터리 생성/로드
• LangChain API와 시스템 프롬프트로 RAG 체인 구성
• st.chat_input / st.chat_message 로 카톡·GPT 스타일 UI
• 메시지 지연 없이 즉시 표시
"""

# 0️⃣ 개발자 전용 설정
TEMPERATURE   = 0.2        # 응답의 창의성 (0.0 – 1.0)
N_CANDIDATES  = 3           # 생성 답변 수
RETRIEVE_K    = 2           # 검색할 문서 개수 ↓ 속도↑
CHUNK_SIZE    = 400         # 청크 크기 ↓ 속도↑
CHUNK_OVERLAP = 40          # 청크 오버랩 ↓ 속도↑
MODEL_NAME    = "gpt-4o-mini" # OpenAI 모델 이름

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import hashlib

# LangChain / FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 시스템 프롬프트용
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 1️⃣ 환경 변수 로드
load_dotenv()  # .env에 OPENAI_API_KEY 필요

# 2️⃣ 커스텀 CSS 적용
st.markdown(
    """
    <style>
    /* 배경색 · 폰트 */
    body {background-color: #f4f6f8;}
    .css-18e3th9 {padding: 1rem 2rem;}  /* main container */
    /* 헤더 스타일 */
    .title {font-size:2.5rem; color:#0057a4; font-weight:bold; margin-bottom:0;}
    .subtitle {font-size:1rem; color:#444; margin-top:0.2rem;}
    /* 사이드바 헤더 */
    .sidebar .css-1d391kg h2 {color:#0057a4;}
    /* 채팅 박스 */
    .stChatMessage > div {
        border-radius: 12px !important;
        padding: 0.75rem !important;
        font-size: 0.95rem;
    }
    /* 사용자 메시지 */
    .stChatMessage.stChatMessageUser > div {
        background-color: #e1f5fe !important;
        color: #333 !important;
    }
    /* 챗봇 메시지 */
    .stChatMessage.stChatMessageAssistant > div {
        background-color: #ffffff !important;
        border: 1px solid #ddd !important;
    }
    /* 입력창 */
    .stChatInput>div>div>div>textarea {
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3️⃣ Streamlit 페이지 설정 & 헤더
st.set_page_config(page_title="NCSOFT 운영약관 챗봇", layout="wide")
st.markdown('<h1 class="title">💬 NCSOFT 운영약관 챗봇</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">NCSOFT 공식 운영약관을 기반으로 고객 문의에 답변해 드립니다.</p>', unsafe_allow_html=True)
st.markdown("---")

# 4️⃣ 사이드바 안내
with st.sidebar:
    st.markdown('<h2>⚙️ 사용법</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        - `policy_docs/` 폴더에 NCSOFT 운영약관(.txt/.pdf) 파일을 넣고 새로고침  
        - 검색 문서 수: **%d**, 청크 크기: **%d/%d**  
        - 모델: **%s**, 온도: **%.2f**, 후보 수: **%d**
        """ % (RETRIEVE_K, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, TEMPERATURE, N_CANDIDATES)
    )

# 4️⃣ 문서 변경 감지 해시
def get_docs_hash() -> str:
    docs_dir = Path(__file__).parent / "policy_docs"
    h = hashlib.md5()
    for f in sorted(docs_dir.glob("*")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_mtime).encode())
    return h.hexdigest()

# 5️⃣ FAISS 인덱스 생성/로드
@st.cache_resource
def get_vectorstore(docs_hash: str):
    base = Path(__file__).parent
    docs_dir = base / "policy_docs"
    index_dir = base / f"faiss_index_{docs_hash}"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if index_dir.exists():
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    raw_docs = []
    for f in docs_dir.glob("*"):
        if f.suffix.lower() == ".pdf":
            if PyPDFLoader:
                loader = PyPDFLoader(str(f))
            else:
                st.warning(f"PDF 무시: {f.name}")
                continue
        else:
            loader = TextLoader(str(f), encoding="utf-8")
        raw_docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    return vs

# 6️⃣ 시스템 프롬프트 및 응답 프롬프트 정의
system_template = """
당신은 당신은 NCSOFT 고객 지원 상담사입니다.
답변 내용은 policy_docs 폴더 내 문서를 참고해 전달합니다.
답변시 질문 내용에 대한 정의는 포함하지 않고 해결 방안에 대해서 전달합니다.
정중한 문체를 사용해 답변합니다.
"""

# RAG의 답변 조합 단계에서 사용할 Prompt
combine_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("문서 내용:\n{context}\n\n질문:\n{question}")
])

# 7️⃣ RAG 체인 초기화
@st.cache_resource
def init_chain(docs_hash: str):
    vs = get_vectorstore(docs_hash)
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        n=N_CANDIDATES
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": RETRIEVE_K}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt}
    )

# 8️⃣ 해시 계산 & RAG 체인 로드
docs_hash = get_docs_hash()
with st.spinner("⏳ 인덱스/체인 초기화 중…"):
    qa_chain = init_chain(docs_hash)

# 9️⃣ 대화 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔟 사용자 입력 & 응답
user_input = st.chat_input("질문을 입력하세요…")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    
    # ———리트리버 디버깅 ———
    #docs = qa_chain.retriever.get_relevant_documents(user_input)
    #st.markdown(f"**[디버그]** 검색된 문서 수: {len(docs)}")
    #for i, d in enumerate(docs, 1):
    #    st.markdown(f"- 문서 #{i} (앞 200자): {d.page_content[:200]!r}")
    
    resp = qa_chain({"question": user_input})
    st.session_state.chat_history.append(("assistant", resp["answer"]))

# ⓫ 대화 렌더링
for speaker, msg in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
