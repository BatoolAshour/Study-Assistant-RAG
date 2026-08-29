import os
import shutil

from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY from .env if present

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingest import ingest_pdf
from src.query import answer_question, summarize_document

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Study Assistant RAG API")

# Allow the frontend (served from a different origin/port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4
    history: list[HistoryTurn] = []
    source: str | None = None  # restrict to one ingested PDF; None = search all


class SummarizeRequest(BaseModel):
    source: str | None = None  # None = summarize everything ingested (usually not what you want)


class QuestionResponse(BaseModel):
    answer: str


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = ingest_pdf(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {"filename": file.filename, "chunks_stored": num_chunks}


@app.post("/query", response_model=QuestionResponse)
async def query(req: QuestionRequest):
    """Ask a question about the ingested documents."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        history = [h.dict() for h in req.history]
        answer = answer_question(
            req.question, top_k=req.top_k, verbose=False, history=history, source=req.source
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return {"answer": answer}


@app.post("/summarize", response_model=QuestionResponse)
async def summarize(req: SummarizeRequest):
    """Summarize one ingested document. Pass 'source' as the exact filename used at ingest time."""
    try:
        answer = summarize_document(source=req.source, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarize failed: {e}")

    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve the simple frontend — must be mounted last, after all API routes above
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")