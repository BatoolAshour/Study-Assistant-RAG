# Study Assistant RAG

Ask questions about your own PDFs and get grounded answers with page citations — no hallucinated facts.

## How it works
PDF → chunked → embedded locally → stored in Chroma → your question retrieves the top matching chunks → Groq LLM answers using only that context.

## Setup
```bash
pip install -r requirements.txt
```
Create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```
(free key: [console.groq.com/keys](https://console.groq.com/keys))

## Usage

**CLI:**
```bash
python main.py --pdf pdfs/notes.pdf --question "What's the difference between X and Y?"
python main.py --chat   # interactive mode, after ingesting
```

**Web app:**
```bash
uvicorn api:app
```
Open http://localhost:8000

## Structure
```
main.py       CLI entry point
api.py        FastAPI server
frontend/     Browser UI
src/ingest.py PDF → chunks → embeddings → Chroma
src/query.py  question → retrieve → Groq → answer
```

## Config
Edit in `src/query.py` / `src/ingest.py`:
- `GROQ_MODEL` — LLM used (default: `openai/gpt-oss-20b`)
- `TOP_K` — chunks retrieved per question (default: 5)
- chunk size/overlap in `ingest.py`
