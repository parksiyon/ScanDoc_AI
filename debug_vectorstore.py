from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def debug_vectorstore():
    """Debug script to check vectorstore contents"""
    
    VECTORSTORE_PATH = "vectorstore_index"
    
    if not os.path.exists(VECTORSTORE_PATH):
        print(" Vectorstore not found!")
        return
    
    try:
        # Load vectorstore
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        print(" Vectorstore loaded successfully")
        
        # Get all documents
        all_docs = vector_db.similarity_search("", k=1000)
        print(f"Total documents in vectorstore: {len(all_docs)}")
        
        # Check document sources
        sources = {}
        for doc in all_docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        print(f"\n Documents by source:")
        for source, docs in sources.items():
            print(f" {source}: {len(docs)} chunks")
            # Show first chunk preview
            if docs:
                preview = docs[0].page_content[:300] + "..." if len(docs[0].page_content) > 300 else docs[0].page_content
                print(f"    Preview: {preview}")
                print(f"    Metadata: {docs[0].metadata}")
            print()
        
        # Test search for REBit_resume.pdf
        print(f" Testing search for 'REBit_resume.pdf':")
        rebit_results = vector_db.similarity_search("REBit_resume.pdf", k=5)
        print(f"Found {len(rebit_results)} results")
        
        if rebit_results:
            for i, doc in enumerate(rebit_results):
                print(f"  Result {i+1}:")
                print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"    Content: {doc.page_content[:200]}...")
                print()

        print(f" Testing search for 'resume':")
        resume_results = vector_db.similarity_search("resume", k=5)
        print(f"Found {len(resume_results)} results")
        
        if resume_results:
            for i, doc in enumerate(resume_results):
                print(f"  Result {i+1}:")
                print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"    Content: {doc.page_content[:200]}...")
                print()
        
    except Exception as e:
        print(f" Error debugging vectorstore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_vectorstore()
    