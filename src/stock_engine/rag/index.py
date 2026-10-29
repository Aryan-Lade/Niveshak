"""
Time-filtered vector indexing and dense retrieval using ChromaDB and BGE embeddings.

CRITICAL LEAKAGE GUARANTEE:
retrieve_dense(query, tickers, as_of, k) strictly enforces:
    published_at <= as_of
No document or news chunk published after query time as_of can EVER be returned.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class DenseEmbedder:
    """
    Sentence embedding provider with fallback for testing environments.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        if self.model_name in ["mock", "mock_fast", "test"]:
            self._model = None
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load {self.model_name} ({e}). Using deterministic pseudo-embedder.")
            self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of strings into normalized embedding vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)

        # Fallback: deterministic character-hash pseudo-embeddings for fast tests
        vecs = []
        for text in texts:
            np.random.seed(abs(hash(text)) % (2**31))
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


def chunk_document(
    doc_id: str,
    title: str,
    summary: str,
    tickers: List[str],
    published_at: int,
    source: str,
    max_words: int = 80,
    overlap_words: int = 15,
) -> List[Dict[str, Any]]:
    """
    Split title + summary into chunks of 300-500 tokens (approx 75-100 words).
    Short items stay as a single chunk.
    """
    full_text = f"{title.strip()}. {summary.strip()}".strip()
    words = full_text.split()

    if len(words) <= max_words:
        return [{
            "chunk_id": f"{doc_id}_c0",
            "doc_id": doc_id,
            "text": full_text,
            "tickers": json.dumps(tickers),
            "published_at": int(published_at),
            "source": source,
        }]

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0
    step = max(max_words - overlap_words, 1)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_words]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": f"{doc_id}_c{chunk_idx}",
            "doc_id": doc_id,
            "text": chunk_text,
            "tickers": json.dumps(tickers),
            "published_at": int(published_at),
            "source": source,
        })
        chunk_idx += 1
        if i + max_words >= len(words):
            break

    return chunks


class ChromaVectorIndex:
    """
    Persistent ChromaDB vector index with time-filtered dense retrieval.
    """

    def __init__(
        self,
        persist_dir: Path = Path("data_store/chroma"),
        collection_name: str = "financial_news",
        embedder: Optional[DenseEmbedder] = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder or DenseEmbedder()

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Idempotently index document chunks into ChromaDB.
        Deterministic chunk_id prevents duplicates.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "doc_id": str(c["doc_id"]),
            "tickers": str(c["tickers"]),
            "published_at": int(c["published_at"]),
            "source": str(c["source"]),
        } for c in chunks]

        embeddings = self.embedder.encode(documents).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def retrieve_dense(
        self,
        query: str,
        tickers: Optional[List[str]] = None,
        as_of: Optional[int] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Dense semantic retrieval with strict point-in-time filtering.

        CRITICAL LEAKAGE RULE:
        Returns ONLY chunks where published_at <= as_of.
        """
        if self.collection.count() == 0:
            return []

        query_emb = self.embedder.encode([query]).tolist()

        # Query a generous candidate pool from Chroma
        n_candidates = min(max(k * 4, 20), self.collection.count())
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0] if "distances" in results else [0.0] * len(ids)

        filtered_results: List[Dict[str, Any]] = []

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            pub_at = int(meta.get("published_at", 0))

            # STRICT TIME FILTER: Drop if published after as_of
            if as_of is not None and pub_at > as_of:
                continue

            # TICKER FILTER: If tickers provided, must match at least one
            if tickers:
                doc_tickers = []
                try:
                    doc_tickers = json.loads(meta.get("tickers", "[]"))
                except Exception:
                    pass
                if not any(t in doc_tickers for t in tickers):
                    continue

            score = round(1.0 - float(dist), 4)
            filtered_results.append({
                "chunk_id": cid,
                "doc_id": meta.get("doc_id"),
                "text": doc,
                "score": score,
                "published_at": pub_at,
                "source": meta.get("source"),
                "tickers": meta.get("tickers"),
            })

            if len(filtered_results) >= k:
                break

        return filtered_results
