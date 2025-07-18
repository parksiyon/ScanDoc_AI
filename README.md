# ScandDoc AI Assistant (V1)

ScandDoc AI is a fully local, privacy-first intelligent document assistant that enables users to ask questions directly about their files — including PDFs, CSVs, DOCX, XLSX, and JSON — and receive contextually accurate, LLM-generated answers. This project demonstrates how retrieval-augmented generation (RAG), agents, and embeddings can be integrated to build a truly useful AI product that goes beyond simple Q&A.

---

## Project Purpose

The core goal of this project is to create a private, offline-friendly assistant that can:
- Understand complex documents
- Retrieve relevant chunks of information
- Answer natural language questions with contextual accuracy
- Reason using an agentic framework when needed

All of this is accomplished without relying on APIs or external services, making it suitable for individuals or organizations with strict data privacy requirements.

---

## Features

- Accepts multiple document types: PDF, CSV, DOCX, XLSX, JSON
- Automatically parses, chunks, and indexes documents using embeddings
- Uses FAISS for fast similarity-based retrieval of document chunks
- Employs a local Mistral LLM (via Ollama) for generation
- Wraps the retrieval pipeline in an agent-based reasoning system for smarter decision-making
- Modular backend architecture to support future scaling and feature additions

---

## Tech Stack

### Core Libraries and Tools

- **LangChain**: A flexible framework for building applications with LLMs. Used to manage chains, agents, prompts, and vector retrieval logic.
- **FAISS**: A fast vector similarity search library that enables efficient document chunk retrieval based on user queries.
- **HuggingFace Embeddings**: Converts document text into vector embeddings that can be stored in FAISS and retrieved for relevance during querying.
- **Ollama + Mistral**: Lightweight, local LLM that generates responses. Avoids dependency on cloud-based APIs.
- **Flask**: Backend API framework used to serve the assistant and connect endpoints.
- **PyMuPDF / docx / pandas / openpyxl / json**: Python libraries used to parse different file types.

---

## Why I Chose These Technologies

- **LangChain** was chosen for its support of agents, retrievers, prompt templates, and ease of chaining components.
- **FAISS** provides reliable, fast retrieval from large datasets, which is critical in a RAG pipeline.
- **Mistral via Ollama** allows for complete offline use without compromising generation quality.
- **AgentExecutor** in LangChain adds step-by-step reasoning and logic to query processing, ideal for multi-step document understanding.
- **File parsers (pymupdf, pandas, etc.)** were selected for their reliability and robustness across document formats.

---

## Retrieval-Augmented Generation (RAG) Explained

RAG improves traditional language models by injecting actual data from external sources (in this case, your documents) into the model’s prompt before generating a response. The flow is as follows:

1. Documents are parsed and broken into manageable "chunks" (text blocks).
2. Each chunk is converted into a numerical embedding (vector) using HuggingFace's embedding model.
3. These vectors are stored in FAISS for fast similarity-based retrieval.
4. When a user asks a question:
    - The system searches the vectorstore for the most relevant document chunks.
    - These chunks, along with the user’s query, are passed to the LLM.
    - The model then generates an answer grounded in real document content, not just its training data.

This process enables the assistant to answer questions it has never seen before — as long as the answer exists within the provided documents.

---

## Agents and LangChain's AgentExecutor

Instead of hardcoding logic or chaining fixed steps, the system uses LangChain’s `AgentExecutor` to dynamically determine:
- Whether a tool (like the retriever) needs to be called
- Which documents or content are relevant
- How to combine retrieved knowledge with reasoning

This gives the assistant a decision-making capability — it doesn’t just respond, it thinks through the problem and selects the best path.

---

## Project Structure

The codebase is modular and production-ready, designed for clarity, scalability, and ease of debugging.

