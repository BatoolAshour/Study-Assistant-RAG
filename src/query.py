"""
src/query.py
Retrieves relevant chunks for a question and asks Groq (free, fast API) to answer.
Not meant to be run directly — called from main.py.

Requires:
    GROQ_API_KEY environment variable (free key: https://console.groq.com/keys)
"""

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
GROQ_MODEL = "openai/gpt-oss-20b"  # free, very fast. Try "llama-3.3-70b-versatile" for better quality

_model = None  # embedding model loaded once, reused across calls
_groq_client = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq()  # reads GROQ_API_KEY from env
    return _groq_client


def retrieve(question, top_k=TOP_K):
    model = get_model()
    query_embedding = model.encode([question]).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})
    return chunks


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    return f"""Answer the question using ONLY the context below. If the context doesn't contain
the answer, say so — do not use outside knowledge. Cite the page number(s) you used.

Context:
{context}

Question: {question}

Answer:"""


def ask_groq(prompt):
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


def answer_question(question, top_k=TOP_K, verbose=True):
    """Full pipeline: retrieve -> build prompt -> ask Groq. Returns the answer string."""
    chunks = retrieve(question, top_k=top_k)

    if verbose:
        print(f"\nTop {len(chunks)} chunks retrieved:")
        for c in chunks:
            print(f"  - page {c['page']}: {c['text'][:80]}...")

    prompt = build_prompt(question, chunks)

    if verbose:
        print(f"\nAsking Groq ({GROQ_MODEL}) ...\n")

    return ask_groq(prompt)