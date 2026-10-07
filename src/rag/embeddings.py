"""Embedding backends. All run locally and free.

- ollama : `nomic-embed-text` served by Ollama (default; uses the GPU when present)
- hf     : sentence-transformers model, e.g. BAAI/bge-small-en-v1.5 (downloads once)
- tfidf  : scikit-learn TF-IDF + SVD, no download at all (tests / CI / offline fallback)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
import requests


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def save(self, folder: Path) -> None: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class OllamaEmbedder:
    def __init__(self, model: str, url: str):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"ollama:{model}"

    def fit(self, texts):  # nothing to train
        pass

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch):
            try:
                r = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts[i:i + batch]}, timeout=300)
                r.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(
                    f"Could not reach Ollama at {self.url}. Start it with `ollama serve` and run "
                    f"`ollama pull {self.model}`, or set EMBED_BACKEND=tfidf. ({e})"
                ) from e
            vectors.extend(r.json()["embeddings"])
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def save(self, folder):
        pass


class HFEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model)
        self.name = f"hf:{model}"

    def fit(self, texts):
        pass

    def embed(self, texts):
        return _normalise(np.asarray(self.model.encode(texts, batch_size=32, show_progress_bar=False), dtype=np.float32))

    def save(self, folder):
        pass


class TfidfEmbedder:
    """Latent semantic analysis: TF-IDF followed by truncated SVD. Fitted on the corpus."""

    name = "tfidf-lsa"

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.vectorizer = None
        self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
        x = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42).fit(x)

    def embed(self, texts):
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF embedder is not fitted; build the index first.")
        return _normalise(self.svd.transform(self.vectorizer.transform(texts)).astype(np.float32))

    def save(self, folder: Path):
        with open(folder / "tfidf.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.svd), f)

    @classmethod
    def load(cls, folder: Path) -> "TfidfEmbedder":
        obj = cls()
        with open(folder / "tfidf.pkl", "rb") as f:
            obj.vectorizer, obj.svd = pickle.load(f)
        return obj


def make_embedder(settings, index_dir: Path | None = None) -> Embedder:
    backend = settings.embed_backend.lower()
    if backend == "ollama":
        return OllamaEmbedder(settings.ollama_embed_model, settings.ollama_url)
    if backend == "hf":
        return HFEmbedder(settings.hf_embed_model)
    if backend == "tfidf":
        if index_dir is not None and (index_dir / "tfidf.pkl").exists():
            return TfidfEmbedder.load(index_dir)
        return TfidfEmbedder()
    raise ValueError(f"Unknown EMBED_BACKEND '{settings.embed_backend}' (use ollama, hf or tfidf)")
