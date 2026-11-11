# Local Advanced RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](#prerequisites)

A self-contained, enterprise-grade, privacy-focused Retrieval-Augmented Generation (RAG) platform built with Python, local embeddings, persistent ChromaDB, hybrid BM25 lexical search, cross-encoder reranking, and local Ollama LLMs.

Includes a **website-embeddable chatbot widget** powered by native Shadow DOM that can be dropped into any external website with a single `<script>` tag or `<iframe>`.

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                     Client Layer                        │
                        │ ┌──────────────────────┐     ┌────────────────────────┐ │
                        │ │ Responsive Web App   │     │  Embeddable Chatbot    │ │
                        │ │ (HTML / CSS / JS SPA)│     │  (Shadow DOM / Iframe /│ │
                        │ │  (Chat / Ingestion / │     │   React / Demo Studio) │ │
                        │ │   Widget / Settings) │     │                        │ │
                        │ └──────────┬───────────┘     └───────────┬────────────┘ │
                        └────────────┼─────────────────────────────┼──────────────┘
                                     │ REST / SSE                  │ REST / SSE
                                     ▼                             ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │              FastAPI Backend (Port 8000)                │
                        │          (Health, Documents, Chat, Widget, Settings)    │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
         ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
         ▼                                           ▼                                           ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│     Ingestion & Processing      │ │        Retrieval Engine         │ │        Generation Engine        │
│ ─────────────────────────────── │ │ ─────────────────────────────── │ │ ─────────────────────────────── │
│ • Multi-format Loaders (PDF,    │ │ • Dense Semantic: ChromaDB      │ │ • Local Ollama LLM Client       │
│   DOCX, PPTX, XLSX, CSV, MD)    │ │ • Sparse Lexical: BM25 (Okapi)  │ │ • Grounded System Prompting     │
│ • Multimodal Vision: Moondream  │ │ • Reciprocal Rank Fusion (RRF)  │ │ • Hallucination Validator       │
│   diagram/chart transcription   │ │ • Cross-Encoder Reranker        │ │ • Deterministic Abstention      │
│ • Contextual Retrieval: Prepends│ │ • SQLite Relational Metadata    │ │ • Citation Extractor & Location │
│   situated document preambles   │ │   Storage (documents & sessions)│ │   Auto-Correction Engine        │
└─────────────────────────────────┘ └─────────────────────────────────┘ └─────────────────────────────────┘
```

---

## Key Features

1. **100% Local & Privacy-Preserving**: Zero cloud API dependencies and zero data telemetry. Embeddings, vision models, and LLM inferences execute completely offline via Ollama or local Sentence Transformers models.
2. **Multi-Format Document Ingestion**: Ingests PDF, DOCX, PPTX, XLSX, CSV, TXT, Markdown, and HTML with file validation, SHA-256 hash de-duplication, and metadata provenance.
3. **Multimodal Vision Ingestion (Diagrams & Charts)**: Automatically transcribes diagrams, flowcharts, architectures, and charts from PDFs, DOCX, and PPTX files into rich contextual Markdown descriptions using local Ollama vision models (`moondream` or `llava`).
4. **Contextual Retrieval (Anthropic-Style)**: Eliminates "amnesiac chunk retrieval misses" by prepending situated document title, section heading, and page anchors to chunks prior to embedding and BM25 indexing. Supports both `structural` (instant, zero-latency) and `llm` (dynamic local synthesis) preamble modes.
5. **Structure-Aware & Parent-Child Chunking**: Preserves section hierarchies, slide boundaries, table structures, and page provenance.
6. **Hybrid Dense + Sparse Retrieval**: Combines dense semantic vector retrieval (ChromaDB cosine similarity) with sparse keyword retrieval (BM25) using Reciprocal Rank Fusion (RRF).
7. **Cross-Encoder Reranking**: Re-scores candidate chunks with Sentence Transformers cross-encoders (`bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) for optimal context quality.
8. **Strict Hallucination Prevention & Abstention**: Rejects ungrounded statements, checks numerical grounding against source context, and triggers deterministic abstention whenever retrieval context is insufficient.
9. **Traceable Citations & Location Auto-Correction**: Produces verified source attributions formatted as `[Source: document.pdf, Page: X]` mapped directly to original chunks, autocorrecting hallucinated page numbers based on actual chunk metadata.
10. **Website-Embeddable Chatbot Widget**:
    - **Native Shadow DOM Encapsulation**: Complete CSS and DOM isolation — host site styles (Bootstrap, Tailwind, WordPress) never break the chatbot, and chatbot CSS never bleeds into the host page.
    - **Zero Dependencies**: Lightweight, self-contained vanilla JavaScript (`/static/chat-widget.js`) with no external CDN or library requirements.
    - **Real-Time Token Streaming**: Server-Sent Events (SSE) streaming with typing indicators and fallback to synchronous execution.
    - **Verified Sources Accordion**: Displays interactive citation badges and expandable ground-truth source cards.
11. **Interactive Demo Studio & Showcase**:
    - Live customizable demo page at `http://127.0.0.1:8000/widget/demo` featuring a simulated enterprise webpage with the chatbot widget active.
    - Live customizer for brand colors, titles, bubble placement (bottom-right vs bottom-left), greetings, and suggested question pills.
    - 1-click code generator for 1-line `<script>` tag, responsive `<iframe>` embed, and React / Next.js components.
12. **Comprehensive Evaluation Framework**: Metrics for Precision@K, Recall@K, MRR, Hit Rate, answer faithfulness, and abstention behavior.

---

## Quickstart

### 1. Prerequisites
- Python 3.11+ (Python 3.13 fully supported)
- [Ollama](https://ollama.ai/) installed and running locally:
  ```bash
  ollama serve
  ollama pull llama3.2
  ollama pull nomic-embed-text
  # Optional: for multimodal diagram/chart transcription
  ollama pull moondream
  ```

### 2. Environment Setup
The virtual environment is configured at `.venv`:
```powershell
# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements (if not already installed)
pip install -r requirements.txt
```

### 3. Launching Applications

- **Backend REST API**:
  ```powershell
  .venv\Scripts\python.exe run_backend.py
  ```
  - API Documentation (Swagger): `http://127.0.0.1:8000/docs`
  - Interactive Widget Demo Studio: `http://127.0.0.1:8000/widget/demo`
  - Direct Iframe Chat: `http://127.0.0.1:8000/widget/embed`

- **Responsive Web Application (HTML5 / CSS3 / JavaScript SPA)**:
  ```powershell
  .venv\Scripts\python.exe run_frontend.py
  ```
  - Web console opens at: `http://localhost:8501` (or directly via FastAPI on `http://127.0.0.1:8000/`)
  - Includes **Dashboard**, **Document Management**, **Grounded Chat**, **Retrieval Lab**, **Embeddable Chatbot**, **Settings**, and **Evaluation**.
  - 100% offline-first, theme-adaptive (light/dark mode), zero external CDNs.

---

## Embedding the Chatbot in External Websites

### Method 1: One-Line Script Tag (Floating Widget)
Add this snippet before the closing `</body>` tag of any HTML page (WordPress, Webflow, Squarespace, Wix, Shopify, or static HTML):

```html
<script 
  src="http://127.0.0.1:8000/static/chat-widget.js" 
  data-api-url="http://127.0.0.1:8000"
  data-title="Document Assistant"
  data-subtitle="Offline Grounded Knowledge"
  data-primary-color="#0284c7"
  data-position="bottom-right"
  data-greeting="Hello! 👋 Ask me anything about your uploaded documents."
  data-suggestions="Summarize key documents|What are the main findings?|List technical specs"
  defer>
</script>
```

#### Supported Configuration Attributes:
| Attribute | Description | Default |
|---|---|---|
| `data-api-url` | Backend base URL | `http://127.0.0.1:8000` |
| `data-title` | Chatbot header title | `Document Assistant` |
| `data-subtitle` | Subtitle / Tagline | `Privacy-Preserving Local RAG` |
| `data-primary-color` | Brand theme color (HEX) | `#0284c7` |
| `data-position` | Bubble location (`bottom-right` or `bottom-left`) | `bottom-right` |
| `data-greeting` | Initial welcome message | Built-in greeting |
| `data-suggestions` | Pipe-separated quick prompt pills | Default sample prompts |
| `data-theme` | Visual theme (`dark` or `light`) | `dark` |

### Method 2: Responsive Iframe Embed
For CMS builders or sandboxed environments that restrict custom JavaScript:

```html
<iframe 
  src="http://127.0.0.1:8000/widget/embed?api=http://127.0.0.1:8000&title=Document+Assistant&color=%230284c7"
  width="100%" 
  height="600" 
  frameborder="0" 
  style="border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); max-width: 420px;"
  title="AI Document Assistant">
</iframe>
```

### Method 3: React / Next.js Component
```jsx
import { useEffect } from "react";

export default function DocumentChatWidget() {
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "http://127.0.0.1:8000/static/chat-widget.js";
    script.setAttribute("data-api-url", "http://127.0.0.1:8000");
    script.setAttribute("data-title", "Document Assistant");
    script.setAttribute("data-primary-color", "#0284c7");
    script.setAttribute("data-position", "bottom-right");
    script.async = true;
    document.body.appendChild(script);

    return () => {
      const root = document.getElementById("rag-chat-widget-root");
      if (root) root.remove();
      delete window.__RAG_CHAT_WIDGET_LOADED__;
    };
  }, []);

  return null;
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health, Ollama connectivity, vector store count, active models |
| `POST` | `/documents/upload` | Upload, validate, parse, chunk, and index multi-format files |
| `GET` | `/documents` | List all indexed documents in the relational repository |
| `GET` | `/documents/{document_id}` | Retrieve document metadata |
| `DELETE` | `/documents/{document_id}` | Delete document from SQLite, ChromaDB, BM25, and disk |
| `POST` | `/documents/{document_id}/reindex` | Re-index existing document from disk |
| `POST` | `/chat/query` | Synchronous RAG query with grounded citation validation |
| `POST` | `/chat/stream` | Server-Sent Events (SSE) streaming chat tokens and source cards |
| `GET` | `/chat/sessions` | List active conversation sessions |
| `GET` | `/chat/history/{session_id}` | Retrieve message turn history for a session |
| `DELETE` | `/chat/history/{session_id}` | Delete a conversation session and history |
| `POST` | `/retrieval/search` | Standalone hybrid, semantic, or keyword retrieval probe |
| `GET` | `/settings` | Retrieve active runtime configurations and installed Ollama models |
| `POST` | `/settings` | Update runtime settings (models, thresholds, vision, contextual retrieval) |
| `GET` | `/widget/chat-widget.js` | Deliver standalone embeddable chatbot JavaScript |
| `GET` | `/widget/embed` | Deliver responsive full-page HTML chat interface for iframes |
| `GET` | `/widget/demo` | Interactive widget customizer and simulated website showcase |
| `GET` | `/widget/info` | Return widget configuration metadata and embed code snippets |

---

## Project Structure

```
RAG_advanced/
├── app/
│   ├── api/
│   │   ├── dependencies.py          # FastAPI dependency injection & singletons
│   │   ├── routes/
│   │   │   ├── chat.py              # Chat query, SSE streaming & session endpoints
│   │   │   ├── documents.py         # Document upload, list, delete, re-index
│   │   │   ├── health.py            # System health and Ollama status
│   │   │   ├── retrieval.py         # Standalone retrieval probing endpoint
│   │   │   ├── settings.py          # Runtime configuration update endpoints
│   │   │   └── widget.py            # Embeddable widget & demo delivery endpoints
│   │   └── schemas.py               # Pydantic v2 request and response schemas
│   ├── core/
│   │   ├── config.py                # Pydantic BaseSettings (.env management)
│   │   ├── exceptions.py            # Domain-specific custom exceptions
│   │   └── logging.py               # Thread-safe structured logging
│   ├── database/
│   │   ├── database.py              # SQLAlchemy engine & session factory
│   │   ├── models.py                # DocumentRecord and ChatMessageRecord tables
│   │   └── repository.py            # SQLite CRUD repositories
│   ├── embeddings/
│   │   └── embedding_service.py     # SentenceTransformers & Ollama embeddings + cache
│   ├── evaluation/
│   │   ├── evaluator.py             # RAG metrics (Precision@K, Faithfulness, Abstention)
│   │   └── test_cases.py            # Automated benchmark evaluation dataset
│   ├── generation/
│   │   ├── answer_validator.py      # Numerical grounding, sufficiency & abstention
│   │   ├── citation_manager.py      # Citation extraction, validation & autocorrection
│   │   ├── ollama_client.py         # Ollama HTTP client with token streaming
│   │   ├── prompt_builder.py        # Grounded system prompt & history formatting
│   │   └── rag_chain.py             # End-to-end grounded RAG answering chain
│   ├── ingestion/
│   │   ├── chunking.py              # Structure-aware, paragraph & sentence chunking
│   │   ├── contextual_enricher.py   # Anthropic-style situated chunk preambles
│   │   ├── document_loader.py       # Main ingestion engine orchestrator
│   │   ├── loaders.py               # Multi-format parsers (PDF, DOCX, PPTX, XLSX, etc.)
│   │   ├── text_cleaner.py          # Unicode normalization & whitespace sanitization
│   │   └── vision_processor.py      # Local Ollama vision diagram/chart extraction
│   ├── memory/
│   │   └── conversation_manager.py  # Session history buffer & sliding context
│   ├── retrieval/
│   │   ├── hybrid_search.py         # Reciprocal Rank Fusion (RRF) dense+sparse engine
│   │   ├── keyword_search.py        # Okapi BM25 sparse lexical engine
│   │   ├── query_rewriter.py        # Conversational query disambiguation
│   │   ├── reranker.py              # Cross-Encoder neural reranker
│   │   ├── retrieval_pipeline.py    # Pipeline coordinating query, search & reranking
│   │   └── semantic_search.py       # Dense vector search against ChromaDB
│   ├── static/
│   │   ├── index.html               # Responsive Single-Page Application (HTML5)
│   │   ├── css/
│   │   │   └── app.css              # Theme-adaptive, high-contrast, responsive CSS3
│   │   ├── js/
│   │   │   └── app.js               # Reactive SPA client & offline Markdown parser
│   │   ├── chat-widget.js           # Zero-dependency Shadow DOM embeddable widget
│   │   ├── demo.html                # Interactive Customizer Studio & Showcase
│   │   └── embed.html               # Responsive standalone chat page for iframes
│   └── vectorstore/
│       └── chroma_store.py          # Persistent ChromaDB client & vector CRUD
├── tests/                           # Complete automated pytest test suite (88 tests)
│   ├── test_abstention.py
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_citations.py
│   ├── test_contextual_retrieval.py
│   ├── test_database.py
│   ├── test_document_upload.py
│   ├── test_embeddings.py
│   ├── test_evaluation.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_vectorstore.py
│   ├── test_vision.py
│   └── test_widget.py
├── run_backend.py                   # FastAPI backend launcher
├── run_frontend.py                  # Responsive Web UI server launcher (port 8501)
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## Testing & Quality Assurance

Run the comprehensive automated test suite (76 tests covering all modules):
```powershell
.venv\Scripts\pytest.exe -v
```

### Test Coverage Highlights:
- **Abstention & Grounding**: Validates deterministic abstention on unsupported questions, missing context, and fabricated figures.
- **Contextual Retrieval**: Tests structural and local LLM-assisted situated chunk preambles and BM25 document-level keyword matching.
- **Multimodal Vision**: Tests image filtering, solid-color mask rejection, CMYK-to-RGB normalization, and vision prompt transcription.
- **Embeddable Chatbot**: Tests `/widget/chat-widget.js`, `/widget/embed`, `/widget/demo`, and `/widget/info` endpoints.
- **Document Ingestion**: Validates PDF, DOCX, PPTX, XLSX, CSV, Markdown, and HTML extraction pipelines.
- **Citations**: Validates regex extraction flexibility, hallucinated page number autocorrection, and ungrounded tag stripping.
- **Vector Store & Hybrid Search**: Validates ChromaDB CRUD, cosine distance thresholds, BM25 indexing, and RRF rank fusion.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

