from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .loader import documents
import os

def build_index():
    print(" Starting index building process...")
    
    if not documents:
        print(" No documents found! Please check your 'data' directory.")
        return False
    
    print(f" Found {len(documents)} document(s) to process")
    
    # Split documents with better parameters
    print(" Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,  
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  
    )
    docs = splitter.split_documents(documents)
    print(f" Created {len(docs)} document chunks")
    
    # Show sample of what's being indexed
    print("\n Sample of indexed content:")
    for i, doc in enumerate(docs[:3]):
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"  Chunk {i+1} from {source}: {preview}")
    
    # Create embeddings
    print("\n Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector database
    print(" Building vector database...")
    vector_db = FAISS.from_documents(docs, embedding_model)
    
    # Save vector database
    print(" Saving vector database...")
    vector_db.save_local("vectorstore_index")
    
    print(" Index built and saved successfully!")
    print(f" Total chunks indexed: {len(docs)}")
    
    # List source documents with chunk counts
    sources = {}
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        sources[source] = sources.get(source, 0) + 1
    
    print(" Source documents and chunk counts:")
    for source, count in sources.items():
        print(f"   {source}: {count} chunks")
    
    return True

if __name__ == "__main__":
    success = build_index()
    if success:
        print("\n You can now run the app with: python app.py")
    else:
        print("\n Index building failed. Please check your data directory and try again.")

