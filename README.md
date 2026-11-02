# Local Advanced RAG System

A self-contained, enterprise-grade, privacy-focused Retrieval-Augmented Generation (RAG) platform using Python, local embeddings, persistent ChromaDB, and local Ollama LLMs.

---

## Architecture Overview

```
                      ┌────────────────────────┐
                      │   Streamlit Web UI     │
                      │  (Chat / Ingestion /   │
                      │   Settings / Eval)     │
                      └───────────┬────────────┘
                                  │ REST / SSE
                                  ▼
                      ┌────────────────────────┐
                      │    FastAPI Backend     │
                      │ (app/api/routes/*.py)  │
                      └───────────┬────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Ingestion Engine │    │ Retrieval Engine │    │ Generation Engine│
│ ──────────────── │    │ ──────────────── │    │ ──────────────── │
│ • PyMuPDF (PDF)  │    │ • ChromaDB Vector│    │ • Ollama Local   │
│ • docx, pptx     │    │ • BM25 Lexical   │    │ • Grounded System│
│ • openpyxl, csv  │    │ • RRF Fusion     │    │   Prompt         │
│ • Structure-aware│    │ • Cross-Encoder  │    │ • Citation Verify│
│   Chunking       │    │   Reranking      │    │ • Abstention Rule│
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## Features

1. **100% Local & Privacy-Preserving**: Zero cloud API dependencies. Embeddings and LLM inference execute completely offline via Ollama or local Sentence Transformers models.
2. **Multi-Format Ingestion**: Ingests PDF, DOCX, PPTX, XLSX, CSV, TXT, Markdown, and HTML with file validation and metadata provenance.
3. **Structure-Aware Chunking**: Preserves section boundaries, headings, slide numbers, sheet names, and page numbers.
4. **Hybrid Retrieval**: Combines dense semantic vector retrieval (ChromaDB) with sparse keyword retrieval (BM25) using Reciprocal Rank Fusion (RRF).
5. **Cross-Encoder Reranking**: Re-scores candidate chunks with Sentence Transformers cross-encoders for optimal context quality.
6. **Strict Hallucination Prevention & Abstention**: Rejects ungrounded statements and triggers deterministic abstention whenever context is insufficient.
7. **Traceable Citations**: Produces verified source attributions formatted as `[Source: document.pdf, Page: X]` mapped directly to original chunks.
8. **Comprehensive Evaluation**: Metrics for Precision@K, Recall@K, MRR, Hit Rate, answer faithfulness, and abstention behavior.

---

## Quickstart

### 1. Prerequisites
- Python 3.11+ (Python 3.13 supported)
- [Ollama](https://ollama.ai/) installed and running locally:
  ```bash
  ollama serve
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### 2. Environment Setup
The virtual environment is configured at `.venv`:
```powershell
# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements (if not already installed)
pip install -r requirements.txt
```

### 3. Launching Application
- **Backend REST API**:
  ```powershell
  .venv\Scripts\python.exe run_backend.py
  ```
  API Docs are accessible at: `http://localhost:8000/docs`

- **Streamlit Interactive UI**:
  ```powershell
  .venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
  ```
  Web interface will open at: `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health, Ollama status, vector store counts |
| `POST` | `/documents/upload` | Upload and index multi-format documents |
| `GET` | `/documents` | List all indexed documents |
| `GET` | `/documents/{document_id}` | Retrieve document metadata |
| `DELETE` | `/documents/{document_id}` | Delete document, vectors, and BM25 index |
| `POST` | `/documents/{document_id}/reindex` | Re-index existing document |
| `POST` | `/chat/query` | Synchronous RAG query with citation validation |
| `POST` | `/chat/stream` | Server-Sent Events (SSE) streaming chat |
| `GET` | `/chat/sessions` | List conversation sessions |
| `GET` | `/chat/history/{session_id}` | Retrieve chat history for session |
| `DELETE` | `/chat/history/{session_id}` | Delete conversation session |
| `POST` | `/retrieval/search` | Test standalone hybrid/semantic/keyword retrieval |

---

## Testing & Quality Assurance

Run the comprehensive automated test suite (48 tests covering all modules):
```powershell
.venv\Scripts\pytest.exe -v
```
