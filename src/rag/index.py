"""Build and load the search index: Chroma (vectors, HNSW) + BM25 (keywords)."""
from __future__ import annotations

import json
import pickle
import re
import shutil
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient
from rank_bm25 import BM25Okapi

from .embeddings import make_embedder
from .ingest import Chunk, load_folder

COLLECTION = "reports"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def build_index(settings, raw_dir: Path | None = None, index_dir: Path | None = None, log=print) -> int:
    raw_dir = raw_dir or settings.raw_dir
    index_dir = index_dir or settings.index_dir
    chunks = load_folder(raw_dir, settings.chunk_size, settings.chunk_overlap)
    log(f"Loaded {len(chunks)} chunks from {len({c.source for c in chunks})} PDF(s)")

    if index_dir.exists():
        SharedSystemClient.clear_system_cache()  # drop cached clients so a rebuild in the same process starts clean
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)

    texts = [c.text for c in chunks]
    embedder = make_embedder(settings)
    embedder.fit(texts)
    log(f"Embedding with {embedder.name} ...")
    vectors = embedder.embed(texts)
    embedder.save(index_dir)

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        col.add(
            ids=[c.chunk_id for c in part],
            embeddings=vectors[i:i + 500].tolist(),
            documents=[c.text for c in part],
            metadatas=[{"company": c.company, "year": c.year, "source": c.source, "page": c.page} for c in part],
        )

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(BM25Okapi([tokenize(t) for t in texts]), f)
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f)
    with open(index_dir / "meta.json", "w") as f:
        json.dump({"embedder": embedder.name, "backend": settings.embed_backend, "chunks": len(chunks),
                   "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}, f, indent=2)
    log(f"Index saved to {index_dir}")
    return len(chunks)


class Index:
    def __init__(self, settings, index_dir: Path | None = None):
        self.dir = index_dir or settings.index_dir
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"No index at {self.dir}. Run `python -m rag.cli ingest` first.")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if self.meta["backend"] != settings.embed_backend:
            raise ValueError(
                f"Index was built with EMBED_BACKEND={self.meta['backend']} but settings say "
                f"{settings.embed_backend}. Rebuild the index or change the setting."
            )
        self.chunks = [Chunk(**c) for c in json.loads((self.dir / "chunks.json").read_text(encoding="utf-8"))]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        with open(self.dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.collection = chromadb.PersistentClient(path=str(self.dir / "chroma")).get_collection(COLLECTION)
        self.embedder = make_embedder(settings, self.dir)

    @property
    def companies(self) -> list[str]:
        return sorted({c.company for c in self.chunks})
