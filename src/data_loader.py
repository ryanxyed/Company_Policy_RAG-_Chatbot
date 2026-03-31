from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader, Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    data_path = Path(data_dir).resolve()
    documents = []

    loaders = [
        ("pdf", PyPDFLoader),
        ("txt", TextLoader),
        ("csv", CSVLoader),
        ("xlsx", UnstructuredExcelLoader),
        ("docx", Docx2txtLoader),
        ("json", lambda path: JSONLoader(path, jq_schema=".", text_content=False)),
    ]

    for ext, LoaderClass in loaders:
        for file in data_path.glob(f"**/*.{ext}"):
            try:
                documents.extend(LoaderClass(str(file)).load())
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")

    print(f"[INFO] Total loaded documents: {len(documents)}")
    return documents