from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import os
import re
from groq import Groq

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5           # chunks actually sent to the LLM, after reranking
FETCH_K = 20        # chunks pulled from Chroma before reranking narrows them down
RERANK_MIN_SCORE = -3.0  # cross-encoder chunks scoring below this are dropped as irrelevant
GROQ_MODEL = "openai/gpt-oss-20b"  # fast, good instruction-following, available on your account

_model = None  # embedding model loaded once, reused across calls
_reranker = None  # cross-encoder, loaded once, reused across calls
_groq_client = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing or empty. Check your .env file is in the "
                "project root and formatted as: GROQ_API_KEY=gsk_yourkeyhere (no quotes, no spaces)."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def retrieve(question, top_k=TOP_K, fetch_k=FETCH_K, source=None):
    model = get_model()
    query_embedding = model.encode([question]).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        where={"source": source} if source else None,
    )

    candidates = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        candidates.append({"text": text, "source": meta["source"], "page": meta["page"]})

    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    chunks = [c for score, c in ranked if score >= RERANK_MIN_SCORE][:top_k]
    return chunks


def build_prompt(question, chunks, history=None):
    context = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )

    history_block = ""
    if history:
        turns = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
        history_block = f"""Previous conversation (for context on what "it"/"them"/"those" refer to):
{turns}

"""

    return f"""Answer the question using ONLY the context below. If the context doesn't contain
the answer, say so — do not use outside knowledge. Cite the page number(s) you used.
Write any math or formulas in plain text (e.g. "f(x) -> y"), not LaTeX — do not use \\( \\) or $ symbols.

{history_block}Context:
{context}

Question: {question}

Answer:"""


def clean_latex(text):
    """Strip common LaTeX delimiters/markup the model sometimes uses despite
    instructions not to, so output always renders as plain text in the UI."""
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)       # \( ... \)  -> ...
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.DOTALL)  # \[ ... \] -> ...
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)  # $$ ... $$ -> ...
    text = re.sub(r"(?<!\$)\$(.*?)\$(?!\$)", r"\1", text)          # $ ... $   -> ...
    text = text.replace(r"\rightarrow", "->")
    text = text.replace(r"\to", "->")
    text = text.replace(r"\times", "x")
    text = text.replace(r"\cdot", "*")
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    return text


def ask_groq(prompt, max_tokens=1500):
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return clean_latex(response.choices[0].message.content)


def get_all_chunks(source=None):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    where = {"source": source} if source else None
    results = collection.get(where=where)

    chunks = []
    for text, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})
    chunks.sort(key=lambda c: c["page"])
    return chunks


def summarize_document(source=None, verbose=True, batch_size=15):
    chunks = get_all_chunks(source=source)
    if not chunks:
        return "No document found. Ingest a PDF first."

    # Split into batches so each request stays within token limits
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    if verbose:
        print(f"\nSummarizing {len(chunks)} chunks in {len(batches)} batch(es) ...")

    # Map: summarize each batch
    batch_summaries = []
    for i, batch in enumerate(batches):
        context = "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in batch)
        prompt = f"""Summarize the key points from this excerpt of a document.
Be concise but keep specific facts, numbers, and findings. Note page numbers where relevant.

Excerpt:
{context}

Summary:"""
        if verbose:
            print(f"  Summarizing batch {i + 1}/{len(batches)} ...")
        batch_summaries.append(ask_groq(prompt, max_tokens=500))

    # Reduce: combine batch summaries into one final summary
    combined = "\n\n".join(f"[Part {i + 1}]\n{s}" for i, s in enumerate(batch_summaries))
    final_prompt = f"""The following are summaries of consecutive sections of a document.
Combine them into one clear, well-organized summary covering the main points,
methodology (if any), key findings, and conclusions. Remove repetition between parts.

Section summaries:
{combined}

Final summary:"""

    if verbose:
        print("  Combining into final summary ...")

    return ask_groq(final_prompt, max_tokens=1200)


def answer_question(question, top_k=TOP_K, verbose=True, history=None, source=None):
    retrieval_query = question
    if history:
        retrieval_query = f"{history[-1]['question']} {question}"

    chunks = retrieve(retrieval_query, top_k=top_k, source=source)

    if verbose:
        print(f"\nTop {len(chunks)} chunks retrieved:")
        for c in chunks:
            print(f"  - page {c['page']}: {c['text'][:80]}...")

    prompt = build_prompt(question, chunks, history=history)

    if verbose:
        print(f"\nAsking Groq ({GROQ_MODEL}) ...\n")

    return ask_groq(prompt)