import os
from src.vectorstore import ChromaVectorStore
from langchain_cohere import ChatCohere
import streamlit as st

class RAGSearch:
    def __init__(self, llm_model: str = "command-r-08-2024"):
        self.vectorstore = ChromaVectorStore()
        self.llm_model = llm_model

        self.llm = ChatCohere(
            cohere_api_key = st.secrets["COHERE_API_KEY"],
            model=llm_model,
            temperature=0.1
        )
        print(f"[INFO] Cohere LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        context = "\n\n".join(r["metadata"].get("text", "") for r in results if r["metadata"])

        if not context:
            return "No relevant documents found."

        prompt = f"""You are a RAG Assistant.
Rules:
- Use only the document to answer.
- If the answer is not in the document, say "I don't know".

Query: {query}

Context:
{context}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content
