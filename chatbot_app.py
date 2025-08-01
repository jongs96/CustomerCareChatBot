# chatbot_app.py
"""
정책문서 기반 고객 불만 자동 분류 & 대응 챗봇

• policy_docs/ 폴더의 .txt/.pdf 문서를 읽어
  변경 감지 시마다 FAISS 인덱스 디렉터리 생성/로드
• LangChain API로 RAG 체인 구성
• st.chat_input / st.chat_message 로 카톡·GPT 스타일 UI
• 메시지 지연 없이 즉시 표시
"""

# 0️⃣ 개발자 전용 설정 (이용자는 UI에서 변경 불가)
TEMPERATURE   = 0.3   # 응답의 창의성 정도 (0.0 – 1.0)
N_CANDIDATES  = 3     # 한번에 생성할 답변 수(LLM 초기화 시 반영)

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import hashlib

# LangChain / FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import TextLoader
try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 1️⃣ 환경 변수 로드
load_dotenv()  # .env에 OPENAI_API_KEY 필수

# 2️⃣ Streamlit 페이지 설정
st.set_page_config(
    page_title="정책문서 기반 고객 불만 챗봇",
    layout="wide",
)
st.title("💬 정책문서 기반 고객 불만 챗봇")
st.markdown("`policy_docs/` 폴더 안의 .txt/.pdf 문서를 기준으로 자동 응답합니다.")
st.markdown("---")

# 3️⃣ 사이드바 안내
with st.sidebar:
    st.header("⚙️ 사용법")
    st.markdown(
        "- `policy_docs/` 폴더에 `.txt` 또는 `.pdf` 파일을 넣으세요.\n"
        "- PDF 지원: `pip install pypdf` → PyPDFLoader 자동 활성화\n"
        "- `.env` 파일에 `OPENAI_API_KEY`를 설정하세요.\n"
        "- 문서 변경 후 새로고침하면 인덱스를 갱신합니다.\n"
        "- 2회차 실행부터는 즉시 로드됩니다."
    )

# 4️⃣ 문서 변경 감지 해시 생성
def get_docs_hash() -> str:
    """
    policy_docs/ 내 파일명+수정시각 리스트로 MD5 해시 생성.
    변경 시마다 해시가 달라져 저장 디렉터리가 분기됩니다.
    """
    docs_dir = Path(__file__).parent / "policy_docs"
    h = hashlib.md5()
    for f in sorted(docs_dir.glob("*")):
        h.update(f.name.encode("utf-8"))
        h.update(str(f.stat().st_mtime).encode("utf-8"))
    return h.hexdigest()

# 5️⃣ FAISS 인덱스 생성/로드 (디렉터리 기반)
@st.cache_resource
def get_vectorstore(docs_hash: str):
    """
    • 인덱스 디렉터리: faiss_index_{docs_hash}
    • 디렉터리 존재 시 FAISS.load_local(..., allow_dangerous_deserialization=True) 로드
    • 없으면 문서 로드→임베딩→FAISS.from_documents→save_local
    """
    base = Path(__file__).parent
    docs_dir = base / "policy_docs"
    index_dir = base / f"faiss_index_{docs_hash}"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1) 기존 인덱스 디렉터리 로드 (pickle 경고 회피)
    if index_dir.exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True  # ⚠️ 신뢰된 로컬 데이터이므로 허용
        )

    # 2) 새 인덱스 생성
    raw_docs = []
    for f in docs_dir.glob("*"):
        if f.suffix.lower() == ".pdf":
            if PyPDFLoader:
                loader = PyPDFLoader(str(f))
            else:
                st.warning(f"PDF 무시(PyPDFLoader 미설치): {f.name}")
                continue
        else:
            loader = TextLoader(str(f), encoding="utf-8")
        raw_docs.extend(loader.load())

    # 문서 분할 → 임베딩 → FAISS 생성
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    vs = FAISS.from_documents(chunks, embeddings)

    # 디렉터리로 저장
    vs.save_local(str(index_dir))
    return vs

# 6️⃣ RAG 체인 초기화 (LangChain API)
@st.cache_resource
def init_chain(docs_hash: str):
    vs = get_vectorstore(docs_hash)

    # LLM 초기화 시 온도 및 후보 수 고정
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=TEMPERATURE,
        model_kwargs={"n": N_CANDIDATES}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

# 7️⃣ 해시 계산 & RAG 체인 로드
docs_hash = get_docs_hash()
with st.spinner("⏳ 인덱스 생성/로드 및 RAG 초기화 중… 잠시만 기다려주세요"):
    qa_chain = init_chain(docs_hash)

# 8️⃣ 대화 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (speaker, message)

# 9️⃣ 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요…")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    resp = qa_chain({"question": user_input})
    st.session_state.chat_history.append(("assistant", resp["answer"]))

# 🔟 대화 렌더링 (즉시 표시)
for speaker, msg in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
