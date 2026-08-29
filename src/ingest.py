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
 
 
def chunk_pages(pages, chunk_size=1000, chunk_overlap=150):
    """Join all page text into one continuous document (so chunks can span page
    breaks instead of being cut off at them), split into overlapping chunks,
    then map each chunk back to the page it starts on via character offsets."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Build one big string plus a list of (start_offset, page_num) breakpoints
    # so we can recover which page a given chunk started on.
    full_text = ""
    offsets = []  # (start_char_offset, page_num)
    for page_num, text in pages:
        offsets.append((len(full_text), page_num))
        full_text += text + "\n"

    pieces = splitter.split_text(full_text)

    chunks = []
    search_pos = 0
    for piece in pieces:
        # Find where this piece starts in full_text (search forward from the
        # last match to handle repeated text correctly and stay efficient).
        idx = full_text.find(piece, max(search_pos - chunk_overlap, 0))
        if idx == -1:
            idx = search_pos
        search_pos = idx

        # Page whose offset is the last one <= idx
        page_num = offsets[0][1]
        for start_offset, p in offsets:
            if start_offset <= idx:
                page_num = p
            else:
                break

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
    # hnsw:space="cosine" matters for MiniLM/BGE-style embeddings — Chroma's
    # default is squared L2, which ranks results differently (and worse) for
    # normalized sentence embeddings than cosine similarity does.
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
 
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