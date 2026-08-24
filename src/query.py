"""
Retrieves relevant chunks for a question and asks Ollama to answer using them.
Not meant to be run directly — called from main.py.
"""

import chromadb
import requests
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"  


def retrieve(question, top_k=TOP_K):
    model = SentenceTransformer(EMBED_MODEL)
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


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_question(question, top_k=TOP_K, verbose=True):
    """Full pipeline: retrieve -> build prompt -> ask Ollama. Returns the answer string."""
    chunks = retrieve(question, top_k=top_k)

    if verbose:
        print(f"\nTop {len(chunks)} chunks retrieved:")
        for c in chunks:
            print(f"  - page {c['page']}: {c['text'][:80]}...")

    prompt = build_prompt(question, chunks)

    if verbose:
        print(f"\nAsking Ollama ({OLLAMA_MODEL}) ...\n")

    return ask_ollama(prompt)