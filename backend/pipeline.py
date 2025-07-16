from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from .prompt import rag_prompt
import os

llm = OllamaLLM(model="mistral", temperature=0.1)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTORSTORE_PATH = "vectorstore_index"
vector_db, retriever = None, None

if os.path.exists(VECTORSTORE_PATH):
    try:
        vector_db = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "fetch_k": 20})
        print(" Vectorstore loaded successfully")
        
        test_docs = vector_db.similarity_search("sample", k=1)
        print(f" Vectorstore test successful - found {len(test_docs)} documents")
    except Exception as e:
        print(f" Error loading vectorstore: {e}")
else:
    print(" Vectorstore not found. Please run indexer.py first to build it")

def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"Document {i+1} (from {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

def enhanced_rag_invoke(query: str) -> str:
    try:
        if not retriever:
            return " Retriever not available. Check vectorstore."

        relevant_docs = retriever.invoke(query)
        if not relevant_docs:
            return f" No documents matched your query '{query}'. Try asking 'What documents are available?'"

        context = format_docs(relevant_docs)
        prompt_input = {"context": context, "question": query}
        return llm.invoke(rag_prompt.format(**prompt_input))

    except Exception as e:
        return f" Error in RAG chain: {str(e)}"

rag_chain = enhanced_rag_invoke if retriever else None
