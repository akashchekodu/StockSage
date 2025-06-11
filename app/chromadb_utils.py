# app/chromadb_utils.py

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid

# ✅ Initialize ChromaDB persistent client (new API)
client = PersistentClient(path="./chroma_db")

# ✅ Create or load collection
collection = client.get_or_create_collection(
    name="news_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ✅ Load embedder
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

def upsert_chunks(chunks: List[Dict], batch_size: int = 5000):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        documents = [chunk["chunk"] for chunk in batch]
        metadatas = [{"title": c["title"], "source": c["source"], "link": c["link"]} for c in batch]
        ids = [str(uuid.uuid4()) for _ in batch]
        embeddings = embedder.encode(documents).tolist()

        collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )

def search_similar_chunks(query: str, top_k: int = 10) -> List[Dict]:
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append({
            "chunk": doc,
            "title": meta["title"],
            "source": meta["source"],
            "link": meta["link"]
        })
    return output
