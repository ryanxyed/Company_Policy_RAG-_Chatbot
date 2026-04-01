import streamlit as st
import os
from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch
from src.chunking import CHUNK_SIZE, CHUNK_OVERLAP

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Company Portal", page_icon="📂", layout="centered")

page = st.sidebar.radio("Navigate", ["📂 Upload Portal", "📚 RAG Q&A"])

# ─────────────────────────────────────────────
# Upload Portal Page
# ─────────────────────────────────────────────
if page == "📂 Upload Portal":
    st.markdown("""
        <style>
        body { background-color: #f0f2f6; }
        .upload-card h1 { color: #2c3e50; font-family: 'Helvetica', sans-serif; margin-bottom: 8px; }
        .upload-card p { color: #555555; font-size: 15px; margin-bottom: 20px; }
        .uploaded-list { background-color: #fafafa; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: left; }
        </style>""", unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/109/109612.png", width=90)
    st.markdown("<h1>Upload Your Files</h1>", unsafe_allow_html=True)
    st.markdown("<p>Drag & drop or browse to select files. They will be saved securely in the <b>data</b> folder.</p>",
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📂 Choose files to upload",
        type=["pdf", "txt", "csv", "xlsx", "docx", "json", "jpg", "png"],
        accept_multiple_files=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} uploaded successfully!")
            st.info(f"📍 Saved at: `{save_path}`")

        st.subheader("📜 Uploaded Files")
        st.markdown('<div class="uploaded-list">', unsafe_allow_html=True)
        for file in os.listdir(UPLOAD_DIR):
            st.write(f"- {file}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No files uploaded yet. Start by selecting a file above.")

# ─────────────────────────────────────────────
# RAG Q&A Page
# ─────────────────────────────────────────────
elif page == "📚 RAG Q&A":
    @st.cache_resource
    def initialize_rag():
        rag = RAGSearch()
        docs = load_all_documents("data")
        rag.vectorstore.build_from_documents(docs)
        return rag

    rag_search = initialize_rag()

    st.title("📚 Company Policy RAG Chatbot")

    st.sidebar.header("System Configuration")
    st.sidebar.text(f"LLM : Groq")
    st.sidebar.text(f"LLM Model : {rag_search.llm_model}")
    st.sidebar.text(f"Chunk Size : {CHUNK_SIZE}")
    st.sidebar.text(f"Chunk Overlap : {CHUNK_OVERLAP}")
    st.sidebar.text(f"Embedding : TF-IDF")
    st.sidebar.text(f"Vector DB : ChromaDB")
    top_k = st.sidebar.slider("Number of results", 1, 10, 3)

    # ── Initialize session state ─────────────
    if "query" not in st.session_state:
        st.session_state["query"] = ""
    if "last_searched" not in st.session_state:
        st.session_state["last_searched"] = ""
    if "answer" not in st.session_state:
        st.session_state["answer"] = ""

    # ── Query Input ──────────────────────────
    st.subheader("Ask a Question")

    query = st.text_input(
        "Type your question:",
        value=st.session_state["query"],
    )
    st.session_state["query"] = query

    # ── Auto search ──────────────────────────
    final_query = st.session_state["query"]

    if final_query.strip() != "" and final_query != st.session_state["last_searched"]:
        st.session_state["last_searched"] = final_query
        with st.spinner("Searching and summarizing..."):
            st.session_state["answer"] = rag_search.search_and_summarize(final_query, top_k=top_k)

    # ── Show Answer ──────────────────────────
    if st.session_state["answer"]:
        st.subheader("📋 Answer")
        st.write(st.session_state["answer"])