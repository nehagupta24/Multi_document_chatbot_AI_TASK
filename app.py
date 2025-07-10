import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

from tools import run_agent
from rag_index import build_index_from_file
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()

openai_model = os.getenv("OPENAI_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

st.set_page_config(
    page_title="🤖 AI-Powered Multi-Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "run_query" not in st.session_state:
    st.session_state.run_query = False
if "reset_query" not in st.session_state:
    st.session_state.reset_query = False


if st.session_state.reset_query:
    st.session_state.query = ""
    st.session_state.reset_query = False


st.markdown("""
    <div style="background-color:#f0f4f8;padding:30px 50px;border-radius:12px;margin-bottom:30px;">
        <h1 style="color:#1a237e;margin-bottom:10px;font-size:2.8rem;">
            🤖 AI-Powered Multi-Document Chatbot
        </h1>
        <p style="color:#37474f;font-size:1.1rem;">
            Upload PDF, DOCX, or PPTX files and ask your questions.<br>
            Get accurate, real-time answers with clear source highlights using Retrieval-Augmented Generation (RAG) & OpenAI.
        </p>
    </div>
""", unsafe_allow_html=True)


st.sidebar.title("ℹ️ Assistant Info")
st.sidebar.write(f"**Model in use:** `{openai_model}`")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)

try:
    db = FAISS.load_local("rag_faiss_store", embeddings, allow_dangerous_deserialization=True)
    st.sidebar.write(f"**Chunks Indexed:** {db.index.ntotal}")
except:
    st.sidebar.write("**Chunks Indexed:** Not available yet")


st.subheader("📤 Upload Document")
uploaded_file = st.file_uploader("Supported formats: PDF, DOCX, PPTX", type=["pdf", "docx", "pptx"])

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_doc_path = tmp_file.name

    st.success(f"✅ `{uploaded_file.name}` uploaded successfully!")

    with st.spinner("🔍 Indexing document..."):
        build_index_from_file(tmp_doc_path, persist_dir="rag_faiss_store")
    st.success("📚 Document indexed successfully!")

 
    st.markdown("---")
    st.subheader("💬 Ask a Question")

  
    query = st.text_input(
        "Type your question here:",
        value=st.session_state.query,
        key="query"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ Submit"):
            st.session_state.run_query = True
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.run_query = False
            st.session_state.reset_query = True   
            st.rerun()

    
    if st.session_state.run_query and query.strip() != "":
        with st.spinner("🤖 Generating answer..."):
            answer, history, sources = run_agent(query)

        st.session_state.chat_history = history
        st.session_state.run_query = False  

        st.markdown("### 🧠 Assistant's Answer")
        placeholder = st.empty()
        partial = ""

        for sentence in answer.split(". "):
            partial += sentence.strip() + ". "
            placeholder.markdown(partial + "▌")
            time.sleep(0.05)
        placeholder.markdown(partial)

        if sources:
            st.markdown("### 📚 Source Chunks")
            for i, doc in enumerate(sources, 1):
                st.info(f"**Chunk {i}:**\n\n{doc.page_content.strip()}")

        with st.expander("📜 Full Chat History"):
            for msg in st.session_state.chat_history:
                role = msg.get("role", "").capitalize()
                name = msg.get("name", "")
                content = msg.get("content", "")
                if content:
                    st.markdown(f"**{role} ({name})**: {content}")

else:
    st.info("📂 Please upload a document to get started.")
