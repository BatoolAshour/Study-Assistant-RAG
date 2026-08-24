""" Loads a PDF, chunks it, embeds it, and stores it in Chroma.
Not meant to be run directly — called from main.py.
"""

import os

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_pdf_pages(path):
    """Return list of (page_number, text) tuples."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((i + 1, text))
    return pages
 
 
def chunk_pages(pages, chunk_size=500, chunk_overlap=50):
    """Split each page's text into overlapping chunks, keeping page number as metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "page": page_num})
    return chunks
 
 
def ingest_pdf(pdf_path):
    """Full pipeline: load -> chunk -> embed -> store. Returns number of chunks stored."""
    doc_name = os.path.basename(pdf_path)
 
    print(f"Loading {pdf_path} ...")
    pages = load_pdf_pages(pdf_path)
    print(f"  {len(pages)} pages with text extracted")
 
    print("Chunking ...")
    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks created")
 
    print(f"Loading embedding model ({EMBED_MODEL}) ...")
    model = SentenceTransformer(EMBED_MODEL)
 
    print("Embedding chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
 
    print("Storing in Chroma ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
 
    ids = [f"{doc_name}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name, "page": c["page"]} for c in chunks]
 
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
 
    print(f"Done. {len(chunks)} chunks stored in '{CHROMA_PATH}'.")
    return len(chunks)

