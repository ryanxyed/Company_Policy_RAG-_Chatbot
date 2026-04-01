import streamlit as st
import os
import tempfile
from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch
from src.chunking import CHUNK_SIZE, CHUNK_OVERLAP

# ✅ Temp storage (Streamlit safe)
if "temp_dir" not in st.session_state:
    st.session_state["temp_dir"] = tempfile.mkdtemp()

UPLOAD_DIR = st.session_state["temp_dir"]

st.set_page_config(page_title="Company Portal", page_icon="📂", layout="centered")

page = st.sidebar.radio("Navigate", ["📂 Upload Portal", "📚 RAG Q&A"])

# ─────────────────────────────────────────────
# Upload Portal
# ─────────────────────────────────────────────
if page == "📂 Upload Portal":
    st.title("📂 Upload Your Files")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "csv"],  # ✅ removed images
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")

        st.subheader("Files:")
        for file in os.listdir(UPLOAD_DIR):
            st.write(file)
    else:
        st.info("Upload files first")

# ─────────────────────────────────────────────
# RAG Q&A
# ─────────────────────────────────────────────
elif page == "📚 RAG Q&A":

    @st.cache_resource
    def initialize_rag(upload_dir):
        rag = RAGSearch()
        docs = load_all_documents(upload_dir)

        if not docs:
            raise ValueError("No documents loaded")

        rag.vectorstore.build_from_documents(docs)
        return rag

    # ✅ Safe initialization
    try:
        if os.path.exists(UPLOAD_DIR) and len(os.listdir(UPLOAD_DIR)) > 0:
            rag_search = initialize_rag(UPLOAD_DIR)
        else:
            st.warning("⚠️ Please upload documents first.")
            st.stop()
    except Exception:
        st.error("❌ Invalid or empty documents. Upload proper text files.")
        st.stop()

    st.title("📚 Company Policy RAG Chatbot")

    top_k = st.slider("Top results", 1, 10, 3)

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Searching..."):
            answer = rag_search.search_and_summarize(query, top_k=top_k)

        st.subheader("Answer")
        st.write(answer)