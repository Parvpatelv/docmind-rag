import os, shutil, tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ingest import ingest_pdf, FAISS_INDEX_PATH
from retriever import build_qa_chain, answer_question

qa_chain = None
# Simple in-memory history: list of {"role": "user"/"assistant", "content": str}
chat_history = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    if Path(FAISS_INDEX_PATH).exists():
        print("Loading QA chain...")
        qa_chain = build_qa_chain()
    else:
        print("No index yet — upload a PDF first.")
    yield

app = FastAPI(title="RAG Q&A API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    @app.get("/")
    async def serve_frontend():
        return FileResponse("static/index.html")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list

class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str

@app.get("/health")
async def health():
    return {"status": "ok", "index_exists": Path(FAISS_INDEX_PATH).exists()}

@app.post("/clear")
async def clear_history():
    """Clear conversation history — called when user starts a new chat."""
    global chat_history
    chat_history = []
    return {"message": "Conversation history cleared."}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain, chat_history
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        chunks = ingest_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Ingestion error: {e}")
    finally:
        os.unlink(tmp_path)
    qa_chain = build_qa_chain()
    chat_history = []  # reset history when new doc is uploaded
    return UploadResponse(
        filename=file.filename,
        chunks_indexed=chunks,
        message=f"Indexed {chunks} chunks from '{file.filename}'."
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask(body: QuestionRequest):
    global chat_history
    if qa_chain is None:
        raise HTTPException(400, "No PDF uploaded yet.")
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        # Pass history in — retriever will use it for context
        result = answer_question(qa_chain, body.question, chat_history)

        # Append this turn to history AFTER getting the answer
        chat_history.append({"role": "user", "content": body.question})
        chat_history.append({"role": "assistant", "content": result["answer"]})

        # Keep history to last 10 turns (5 Q&A pairs) to avoid token overflow
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

    except Exception as e:
        raise HTTPException(500, f"QA error: {e}")
    return AnswerResponse(**result)