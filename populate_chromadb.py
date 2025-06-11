# app/populate_chromadb.py

from app.retriever import fetch_and_chunk_articles
from app.chromadb_utils import upsert_chunks

if __name__ == "__main__":
    chunks = fetch_and_chunk_articles()
    print(f"Fetched {len(chunks)} chunks")
    upsert_chunks(chunks)
    print("✅ Successfully updated ChromaDB.")
