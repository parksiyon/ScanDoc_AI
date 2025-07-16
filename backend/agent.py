from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from backend.pipeline import rag_chain, retriever, llm, vector_db
from backend.prompt import agent_prompt
import re

def get_all_docs():
    return list(vector_db.docstore._dict.values()) if vector_db else []

@tool
def document_qa_tool(query: str) -> str:
    """Answers user queries by performing retrieval-augmented generation from loaded documents."""
    try:
        response = rag_chain(query)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f" Error processing query: {str(e)}"

@tool
def list_documents_tool(dummy: str) -> str:
    """Lists all available documents currently in the vectorstore."""
    try:
        all_docs = get_all_docs()
        sources = {doc.metadata.get("source", "Unknown") for doc in all_docs}
        return (
            f" Available documents: {', '.join(sorted(sources))}"
            if sources else " No documents found"
        )
    except Exception as e:
        return f" Error listing documents: {str(e)}"

@tool
def search_by_filename_tool(filename: str) -> str:
    """Searches for document content by filename."""
    try:
        all_docs = get_all_docs()
        matching_docs = [doc for doc in all_docs if filename.lower() in doc.metadata.get("source", "").lower()]
        if matching_docs:
            content_preview = [
                f"Chunk {i+1}: {doc.page_content[:500]}..." for i, doc in enumerate(matching_docs[:3])
            ]
            return f" Found {len(matching_docs)} chunks from {filename}:\n\n" + "\n\n".join(content_preview)
        else:
            return f" No content found for document: {filename}"
    except Exception as e:
        return f" Error searching by filename: {str(e)}"

@tool
def enhanced_document_search_tool(query: str) -> str:
    """Performs a hybrid search using both semantic similarity and filename hints."""
    try:
        semantic_results = vector_db.similarity_search(query, k=5) if vector_db else []
        filename_pattern = r'([^/\\]+\.(pdf|docx?|csv|xlsx|txt|md))'
        filename_match = re.search(filename_pattern, query, re.IGNORECASE)

        results = []

        if semantic_results:
            results.append(" Semantic search results:")
            for i, doc in enumerate(semantic_results):
                source = doc.metadata.get("source", "Unknown")
                results.append(f"{i+1}. From {source}: {doc.page_content[:300]}...")

        if filename_match:
            filename = filename_match.group(1)
            all_docs = get_all_docs()
            filename_results = [doc for doc in all_docs if filename.lower() in doc.metadata.get("source", "").lower()]
            if filename_results:
                results.append(f"\n Content from {filename}:")
                for i, doc in enumerate(filename_results[:3]):
                    results.append(f"Section {i+1}: {doc.page_content[:400]}...")

        return "\n".join(results) if results else f" No relevant information found for: {query}"
    except Exception as e:
        return f" Error in enhanced search: {str(e)}"

@tool
def summarize_document_tool(filename: str) -> str:
    """Summarizes the content of a document by filename."""
    try:
        all_docs = get_all_docs()
        chunks = [doc for doc in all_docs if filename.lower() in doc.metadata.get("source", "").lower()]
        if not chunks:
            return f" No content found for: {filename}"
        combined = "\n\n".join([doc.page_content for doc in chunks[:5]])
        prompt = f"Summarize the following document:\n\n{combined}"
        summary = llm.invoke(prompt)
        return f" Summary of {filename}:\n{summary}"
    except Exception as e:
        return f" Error summarizing document: {str(e)}"

tools = [
    document_qa_tool,
    list_documents_tool,
    search_by_filename_tool,
    enhanced_document_search_tool,
    summarize_document_tool
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate"
)
