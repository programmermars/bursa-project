"""Retrieval: vector (semantic), BM25 (keyword) and hybrid via Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import Index, tokenize
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def _vector(index: Index, query: str, k: int, company: str | None) -> list[tuple[str, float]]:
    qv = index.embedder.embed([query])[0].tolist()
    res = index.collection.query(
        query_embeddings=[qv], n_results=min(k, len(index.chunks)),
        where={"company": company} if company else None,
    )
    ids, dists = res["ids"][0], res["distances"][0]
    return [(i, 1.0 - d) for i, d in zip(ids, dists)]  # cosine distance -> similarity


def _bm25(index: Index, query: str, k: int, company: str | None) -> list[tuple[str, float]]:
    scores = index.bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        c = index.chunks[i]
        if company and c.company != company:
            continue
        if scores[i] <= 0:
            break
        out.append((c.chunk_id, float(scores[i])))
        if len(out) == k:
            break
    return out


def rrf(rankings: list[list[tuple[str, float]]], k: int, c: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = sum 1 / (c + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]


def retrieve(index: Index, query: str, k: int = 5, mode: str = "hybrid", company: str | None = None) -> list[Hit]:
    mode = mode.lower()
    if mode == "vector":
        ranked = _vector(index, query, k, company)
    elif mode == "bm25":
        ranked = _bm25(index, query, k, company)
    elif mode == "hybrid":
        pool = max(k * 4, 20)
        ranked = rrf([_vector(index, query, pool, company), _bm25(index, query, pool, company)], k)
    else:
        raise ValueError(f"Unknown retrieval mode '{mode}' (use vector, bm25 or hybrid)")
    return [Hit(index.by_id[cid], score, r) for r, (cid, score) in enumerate(ranked, start=1)]
