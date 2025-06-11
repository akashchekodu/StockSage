from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict, Tuple
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os
import re
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from app.llm import call_mistral
from app.chromadb_utils import upsert_chunks, search_similar_chunks  # ✅ ChromaDB added

load_dotenv()

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')


def connect_db():
    return psycopg2.connect(os.getenv("DB_CONNECTION_STRING"))


def rephrase_query(query: str, num_variants: int = 3) -> List[str]:
    prompt = f"""You are a helpful assistant.

    Given the user question below, rephrase it into {num_variants} diverse but equivalent versions that preserve the original meaning. Avoid repeating exact words.

    Question: "{query}"

    Rephrased versions:
    1."""
        
    response = call_mistral(prompt)
    text = response.strip()

    variants = re.findall(r"\d+\.\s*(.+)", text)
    
    return variants[:num_variants]


def semantic_chunk_text_smart(text: str, similarity_threshold: float = 0.75, window: int = 5) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return []

    embeddings = embedder.encode(sentences, convert_to_tensor=True).cuda()

    chunks = []
    current_chunk = [sentences[0]]
    current_embeds = [embeddings[0]]

    for i in range(1, len(sentences)):
        past_embeds = torch.stack(current_embeds[-window:])
        sim_scores = F.cosine_similarity(embeddings[i].unsqueeze(0), past_embeds, dim=1)
        max_sim = sim_scores.max().item()

        if max_sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_embeds = [embeddings[i]]
        else:
            current_chunk.append(sentences[i])
            current_embeds.append(embeddings[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def fetch_and_chunk_articles(limit: int = 800, index_to_chromadb: bool = False) -> List[Dict]:
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT id, title, article_body, link, source
        FROM news
        WHERE created_at >= NOW() - INTERVAL '2 days'
        ORDER BY created_at DESC
        LIMIT 500;
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"Searching {len(rows)} documents")

    chunks = []
    for row in rows:
        id, title, body, link, source = row
        if not body:
            continue

        body_chunks = semantic_chunk_text_smart(body)
        for i, chunk in enumerate(body_chunks):
            final_chunk = f"{title}. {chunk}" if i == 0 else chunk
            chunks.append({
                "id": id,
                "title": title,
                "chunk": final_chunk,
                "link": link,
                "source": source
            })

    if index_to_chromadb:
        upsert_chunks(chunks)  # ✅ Add to ChromaDB if flag is True

    return chunks


def filter_with_bm25(query: str, chunks: List[Dict], top_n: int = 100, title_weight: int = 3, source_weight: int = 1) -> List[Dict]:
    corpus = [
        (" ".join([c['title']] * title_weight)) +
        (" ".join([c['source']] * source_weight)) +
        " " + c['chunk']
        for c in chunks
    ]
    tokenized_corpus = [doc.lower().split() or [""] for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(bm25_scores)[::-1][:top_n]
    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks


def group_top_chunks_per_article(ranked_chunks: List[tuple], top_n_per_article: int = 3) -> List[Dict]:
    from collections import defaultdict

    grouped = defaultdict(list)
    for chunk, score in ranked_chunks:
        key = (chunk['title'], chunk['link'])
        grouped[key].append((chunk, score))

    combined_articles = []
    for (title, link), group in grouped.items():
        group.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [g[0]['chunk'] for g in group[:top_n_per_article]]
        combined_chunk = " ".join(top_chunks)
        combined_articles.append({
            "title": group[0][0]['title'],
            "link": group[0][0]['link'],
            "source": group[0][0]['source'],
            "chunk": combined_chunk
        })

    return combined_articles


def rank_with_semantic_similarity(query: str, chunks: List[Dict]) -> List[tuple]:
    query_emb = embedder.encode(query, convert_to_tensor=True).cuda()
    chunk_texts = [f"{c['title']} ({c['source']} - {c['link']}):\n{c['chunk']}" for c in chunks]
    chunk_embs = embedder.encode(chunk_texts, convert_to_tensor=True).cuda()

    sims = F.cosine_similarity(chunk_embs, query_emb.unsqueeze(0), dim=1)
    ranked = list(zip(chunks, sims.tolist()))
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def rerank_with_cross_encoder(query: str, ranked_chunks: List[tuple], top_k: int = None) -> List[Tuple[Dict, float]]:
    cross_inputs = [
        (query, f"{chunk['title']} ({chunk['source']} - {chunk['link']}):\n{chunk['chunk']}")
        for chunk, _ in ranked_chunks
    ]
    cross_scores = cross_encoder.predict(cross_inputs)

    reranked = sorted(
        zip(ranked_chunks, cross_scores),
        key=lambda x: x[1],  # sort by cross score
        reverse=True
    )

    return [(chunk, score) for ((chunk, _), score) in reranked[:top_k]] if top_k else [(chunk, score) for ((chunk, _), score) in reranked]



def hybrid_top_chunks(query: str, chunks: List[Dict], top_n_each: int = 50, title_weight: int = 3, source_weight: int = 1) -> List[Dict]:
    rephrased_queries = [query] + rephrase_query(query, num_variants=2)

    all_top_chunks = []
    seen_keys = set()

    chunk_texts = [f"{c['title']} ({c['source']} - {c['link']}):\n{c['chunk']}" for c in chunks]
    chunk_embs = embedder.encode(chunk_texts, convert_to_tensor=True).cuda()

    for q in rephrased_queries:
        bm25_top_chunks = filter_with_bm25(q, chunks, top_n=top_n_each, title_weight=title_weight, source_weight=source_weight)

        query_emb = embedder.encode(q, convert_to_tensor=True).cuda()
        sims = F.cosine_similarity(chunk_embs, query_emb.unsqueeze(0), dim=1)
        dense_top_idx = torch.topk(sims, top_n_each).indices.tolist()
        dense_top_chunks = [chunks[i] for i in dense_top_idx]

        combined = bm25_top_chunks + dense_top_chunks
        for c in combined:
            key = (c['title'], c['link'], c['chunk'])
            if key not in seen_keys:
                seen_keys.add(key)
                all_top_chunks.append(c)

    return all_top_chunks


def get_relevant_articles(query: str, top_k: int = 5, min_score: float = 0.4) -> List[Dict]:
    top_chunks = search_similar_chunks(query, top_k * 3)  # Get more to allow filtering

    if not top_chunks:
        print("⚠️ No results from ChromaDB.")
        return []

    semantically_ranked = rank_with_semantic_similarity(query, top_chunks)
    grouped_articles = group_top_chunks_per_article(semantically_ranked, top_n_per_article=3)

    dummy_scores = [(article, 0.0) for article in grouped_articles]
    reranked_with_scores = rerank_with_cross_encoder(query, dummy_scores)

    # ✅ Filter by min_score
    filtered = [(chunk, score) for chunk, score in reranked_with_scores if score >= min_score]

    # ✅ Limit to top_k if enough remain
    final_results = [chunk for chunk, score in filtered[:top_k]]

    # print(f"\n✅ Returning {len(final_results)} result(s) with score ≥ {min_score}:")
    # for i, doc in enumerate(final_results):
    #     print(f"{i+1}. {doc['title']} [{doc['source']}]")

    return final_results
