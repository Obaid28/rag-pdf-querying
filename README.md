## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed for semantic querying of PDF documents using vector embeddings. It integrates state-of-the-art NLP and vector search libraries including Hugging Face Transformers, LangChain, and FAISS to enablefactual, context-aware responses from large language models (LLMs).

Key features include:
- Semantic chunking of PDFs into manageable text segments with metadata.
- Vector embedding of chunks for efficient similarity search.
- Metadata-aware retrieval to improve interpretability and relevance.
- Prompt engineering to guide LLM responses and reduce hallucinations.
- Scalable deployment with a FastAPI backend and a user-friendly Streamlit frontend.

-----------------------------------------------------------------------------------------------------------

## Motivation

Large language models often generate impressive but factually inconsistent or hallucinated answers. This project addresses these challenges by augmenting generation with retrieval over semantically relevant PDF content, grounding LLM responses in real document context to improve factual consistency and reduce hallucinations.

-----------------------------------------------------------------------------------------------------------

## Architecture

    PDF[PDF Document] --> Chunker[Semantic Chunking]
    Chunker --> Embeddings[Vector Embeddings (HuggingFace)]
    Embeddings --> FAISS[FAISS Vector Store]
    Query --> Retriever[FAISS Similarity Search]
    Retriever --> LLM[Hugging Face Text Generation]
    LLM --> Response[Generated Answer]
    Response --> API[FastAPI Endpoint]
    API --> Frontend[Streamlit UI]

-----------------------------------------------------------------------------------------------------------

## Features

✅ PDF uploading via API  
✅ Semantic chunking with 20-character overlap  
✅ FAISS-based vector store  
✅ Metadata-aware retrieval (page index, chunk index)  
✅ GPT2-XL for generation  
✅ Prompt templating for factual accuracy  
✅ JSON output with answer and supporting chunks

-----------------------------------------------------------------------------------------------------------

## Usage

Upload PDF: Use the Streamlit UI to upload a PDF document.

Build index: The backend will chunk, embed, and index the document automatically.

Query: Enter your question. The system retrieves relevant chunks and generates a grounded answer.

View sources: Retrieved chunks and metadata are displayed to verify response provenance.

-----------------------------------------------------------------------------------------------------------

## How to Run

```bash
pip install -r requirements.
uvicorn app:app --host 127.0.0.1 --port 8000

Then use a tool like Postman or cURL:

curl -X POST http://127.0.0.1:8000/query/ \
  -F "pdf=@your_file.pdf" \
  -F "question=What is the main topic of this paper?"

In another terminal, run the Streamlit app:

streamlit run streamlit_app.py
