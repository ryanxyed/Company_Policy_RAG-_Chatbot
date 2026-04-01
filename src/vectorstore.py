from typing import List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
from src.chunking import EmbeddingPipeline

class ChromaVectorStore:
    def __init__(self):
        self.emb_pipe = EmbeddingPipeline()
        self.vectorizer = TfidfVectorizer()
        self.fitted = False
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="documents_tfidf")
        print("[INFO] ChromaVectorStore initialized with In-Memory ChromaDB.")

    def build_from_documents(self, documents: List[Any]):
        chunks = self.emb_pipe.chunk_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        self.fitted = True

        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings.tolist(),
            metadatas=[{"text": chunk.page_content} for chunk in chunks]
        )
        print(f"[INFO] Vector store built with {len(chunks)} chunks.")

    def query(self, query_text: str, top_k: int):
        if not self.fitted:
            raise RuntimeError("Vectorizer not fitted. Call build_from_documents first.")
        query_emb = self.vectorizer.transform([query_text]).toarray().tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        return [
            {"id": id_, "distance": dist, "metadata": meta}
            for id_, dist, meta in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0]
            )
        ]