from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document
from pathlib import Path
import os

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

documents = []

def load_and_tag(loader_class, path):
    """Load documents and add source metadata"""
    try:
        loader = loader_class(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["file_path"] = str(path)
            doc.metadata["file_type"] = path.suffix.lower()
        return docs
    except Exception as e:
        print(f" Failed to load {path.name}: {e}")
        return []

def load_documents():
    """Load all documents from the data directory"""
    print(f" Loading documents from {data_dir}")
    
    if not data_dir.exists():
        print(f" Data directory '{data_dir}' does not exist!")
        return []
    
    loaded_docs = []
    supported_extensions = {".pdf", ".csv", ".docx", ".xlsx", ".json", ".txt", ".md"}
    all_files = list(data_dir.glob("*"))
    if not all_files:
        print(f" No files found in {data_dir}")
        return []
    supported_files = [f for f in all_files if f.suffix.lower() in supported_extensions]
    if not supported_files:
        print(f" No supported files found in {data_dir}")
        print(f"Supported extensions: {', '.join(supported_extensions)}")
        print(f"Found files: {[f.name for f in all_files]}")
        return []
    for file_path in supported_files:
        suffix = file_path.suffix.lower()
        print(f" Loading {file_path.name}...")
        
        try:
            if suffix == ".pdf":
                docs = load_and_tag(PyPDFLoader, file_path)
            elif suffix == ".csv":
                docs = load_and_tag(CSVLoader, file_path)
            elif suffix == ".docx":
                docs = load_and_tag(UnstructuredWordDocumentLoader, file_path)
            elif suffix == ".xlsx":
                docs = load_and_tag(UnstructuredExcelLoader, file_path)
            elif suffix in [".json", ".txt", ".md"]:
                docs = load_and_tag(UnstructuredFileLoader, file_path)
            else:
                continue
            
            if docs:
                loaded_docs.extend(docs)
                print(f" Loaded {len(docs)} document(s) from {file_path.name}")
                if docs and docs[0].page_content:
                    preview = docs[0].page_content[:150] + "..." if len(docs[0].page_content) > 150 else docs[0].page_content
                    print(f"   Preview: {preview}")
            else:
                print(f" No content extracted from {file_path.name}")
                
        except Exception as e:
            print(f" Failed to load {file_path.name}: {e}")
    
    print(f" Total documents loaded: {len(loaded_docs)}")
    return loaded_docs

# Load documents on import
documents = load_documents()

if __name__ == "__main__":
    print(" Testing document loader...")
    docs = load_documents()
    print(f" Successfully loaded {len(docs)} documents")
    for doc in docs:
        print(f"{doc.metadata['source']}: {len(doc.page_content)} characters")