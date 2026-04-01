# DocMind — AI Document Q&A (RAG App)

> Upload any PDF. Ask questions in plain English. Get answers grounded in your document — with exact page citations.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-CPU-FF6B35?style=flat)
![Deployed](https://img.shields.io/badge/Deployed-Railway-0B0D0E?style=flat&logo=railway&logoColor=white)

---

## What is this?

DocMind is a full-stack **Retrieval-Augmented Generation (RAG)** application. It lets you have a conversation with any PDF document.

Instead of the LLM answering from its training data, every answer is retrieved directly from your uploaded document — with source page references so you can verify the information yourself.

**No hallucinations. No guessing. Just your document, intelligently searchable.**

---

## Live Demo

🌐 **[Try it live →](https://web-production-bfff40.up.railway.app)**

---

## Features

- **PDF upload** with drag-and-drop support
- **Semantic search** using vector embeddings — finds relevant content even when the exact words don't match
- **Source citations** — every answer shows which page it came from
- **Multi-document support** — upload multiple PDFs, query across all of them
- **Hallucination prevention** — custom prompt engineering forces the LLM to only use document context
- **Interactive web UI** — no frontend framework, pure HTML/CSS/JS
- **Auto-generated API docs** at `/docs` (Swagger UI)

---

## Tech Stack

| Layer         | Technology                      | Purpose                                    |
| ------------- | ------------------------------- | ------------------------------------------ |
| API           | FastAPI + Uvicorn               | REST endpoints, file upload, CORS          |
| Orchestration | LangChain LCEL                  | RAG pipeline (modern, non-deprecated)      |
| Embeddings    | OpenAI `text-embedding-3-small` | Semantic vector generation                 |
| Vector store  | FAISS (CPU)                     | Fast similarity search                     |
| LLM           | GPT-4o (`temperature=0`)        | Answer generation                          |
| PDF parsing   | PyPDF                           | Text extraction from PDFs                  |
| Frontend      | Vanilla HTML/CSS/JS             | Drag-and-drop UI, chat interface           |
| Deployment    | Railway                         | Cloud hosting with env variable management |

---

## How It Works

### Ingestion (when you upload a PDF)

```
PDF file
  → PyPDF extracts text page by page
  → RecursiveCharacterTextSplitter cuts into 1000-char chunks (200-char overlap)
  → OpenAI Embeddings converts each chunk into a 1536-dim vector
  → FAISS indexes all vectors and saves to disk
```

### Retrieval (when you ask a question)

```
Your question
  → Embedded with the same model
  → FAISS finds the 4 most similar chunks
  → Chunks + question passed to GPT-4o with a grounding prompt
  → Answer returned with source page metadata
```

---

## Project Structure

```
rag-app/
├── main.py              # FastAPI app — /upload, /ask, /health endpoints
├── ingest.py            # PDF → chunks → embeddings → FAISS index
├── retriever.py         # LCEL chain: FAISS search → GPT-4o → answer
├── static/
│   └── index.html       # Full web UI (drag-drop upload, chat interface)
├── faiss_index/         # Auto-created on first upload (gitignored)
│   ├── index.faiss      # Binary vector index
│   └── index.pkl        # Chunk text + metadata
├── .env                 # Your API key (never committed)
├── .env.example         # Template for other developers
├── .gitignore
├── requirements.txt
├── Procfile             # Railway deployment config
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key — [get one here](https://platform.openai.com/api-keys)
- An OpenAI account with billing enabled

### 1. Clone the repo

```bash
git clone https://github.com/Parvpatelv/docmind-rag.git
cd docmind-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

> Use single quotes if your key contains special characters:
> `echo 'OPENAI_API_KEY=sk-proj-...' > .env`

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

### 6. Open the app

Visit **http://localhost:8000** in your browser.

Or use the API directly:

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
     -F "file=@your-document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
```

---

## API Reference

### `GET /health`

Check if the server is running and whether an index exists.

**Response:**

```json
{ "status": "ok", "index_exists": true }
```

---

### `POST /upload`

Upload a PDF and index it for querying.

**Request:** `multipart/form-data` with a `file` field (PDF only)

**Response:**

```json
{
  "filename": "document.pdf",
  "chunks_indexed": 47,
  "message": "Indexed 47 chunks from 'document.pdf'."
}
```

---

### `POST /ask`

Ask a question about the uploaded documents.

**Request:**

```json
{ "question": "What are the key findings?" }
```

**Response:**

```json
{
  "answer": "The key findings are...",
  "sources": [
    {
      "page": 3,
      "source": "document.pdf",
      "snippet": "The study found that..."
    }
  ]
}
```

---

**Interactive API docs:** http://localhost:8000/docs

---

## Configuration

All configuration is done via environment variables in `.env`:

| Variable         | Required | Description         |
| ---------------- | -------- | ------------------- |
| `OPENAI_API_KEY` | Yes      | Your OpenAI API key |

### Swap the model

To use a cheaper or local model, edit these files:

**`ingest.py` and `retriever.py`** — change the embedding model:

```python
# Cheaper OpenAI option
model="text-embedding-ada-002"

# Free local option (requires sentence-transformers)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

**`retriever.py`** — change the LLM:

```python
# Cheaper
model="gpt-3.5-turbo"

# Free local (requires Ollama running locally)
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
```

---

## Deployment

This app is deployed on **Railway**.

### Deploy your own instance

1. Fork this repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your forked repo
4. Add environment variable: `OPENAI_API_KEY` = your key
5. Railway auto-deploys. Your public URL appears in the dashboard.

The `Procfile` handles the startup command:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Common Issues

| Error                     | Cause                     | Fix                                        |
| ------------------------- | ------------------------- | ------------------------------------------ |
| `401 invalid_api_key`     | Wrong or corrupted key    | Re-copy key, use single quotes in `.env`   |
| `429 insufficient_quota`  | No OpenAI billing         | Add payment at platform.openai.com/billing |
| `ModuleNotFoundError`     | venv not active           | Run `source venv/bin/activate`             |
| `400 No PDF uploaded yet` | Asking before uploading   | Upload a PDF first via `/upload`           |
| Empty answers             | Scanned PDF (image-based) | Use OCR tool first to extract text         |

---

## What I Learned Building This

- **RAG is a pipeline, not a black box** — every stage (chunking, embedding, retrieval, prompting) can be tuned independently
- **LangChain moves fast** — three import paths broke mid-build due to library restructuring; learned to debug Python packaging issues in real time
- **Prompt engineering prevents hallucination** — without a grounding prompt, GPT-4o answers from training data; two lines of prompt engineering fixed it completely
- **Deployment is its own skill** — getting off localhost with secrets managed, correct host/port config, and a stable public URL is a different challenge from writing the code

---

## License

MIT License — feel free to use, modify, and build on this.

---

## Author

**Parv Patel**

- GitHub: [@Parvpatelv](https://github.com/Parvpatelv)
- LinkedIn: [your LinkedIn URL here]

_Currently open to AI engineering and backend development roles._
