"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.

Components:
1. Candidate filtering: published_at <= as_of AND ticker in tickers.
2. BM25 keyword retrieval: rank_bm25 over filtered candidate corpus.
3. Dense semantic retrieval: ChromaDB vector search with cosine similarity.
4. Reciprocal Rank Fusion (RRF): fuses dense + BM25 rankings with constant k = 60.
5. Cross-Encoder reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 reranks top 20 candidates.
6. Returns top 5 Evidence objects.

NON-NEGOTIABLE LEAKAGE RULE:
The as_of time filter strictly applies to BOTH BM25 and dense retrievers.
No document published after as_of can ever enter candidate generation or reranking.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Retrieved and reranked financial news evidence chunk."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    published_at: int
    source: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "published_at": self.published_at,
            "source": self.source,
            "url": self.url,
        }


class CrossEncoderReranker:
    """
    Reranks (query, document) pairs using a transformer cross-encoder.
    Falls back to token-overlap scoring in test/offline environments.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        if self.model_name in ["mock", "mock_fast", "test"]:
            self._model = None
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder {self.model_name} ({e}). Using lexical fallback.")
            self._model = None

    def predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Compute reranking scores for list of (query, text) pairs."""
        if not pairs:
            return np.empty(0, dtype=np.float32)

        if self._model is not None:
            scores = self._model.predict(pairs)
            return np.array(scores, dtype=np.float32)

        # Fast fallback reranker: lexical jaccard overlap + keyword coverage
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            if not q_words or not d_words:
                scores.append(0.0)
            else:
                overlap = len(q_words & d_words)
                jaccard = overlap / len(q_words | d_words)
                scores.append(float(jaccard * 10.0))
        return np.array(scores, dtype=np.float32)


class HybridRetriever:
    """
    Hybrid Retriever combining BM25 keyword matching and dense vector search
    with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
    """

    def __init__(
        self,
        db_path: Path = Path("data_store/news.db"),
        chroma_dir: Path = Path("data_store/chroma"),
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        vector_index: Optional[Any] = None,
    ) -> None:
        self.db_path = db_path
        self.chroma_dir = chroma_dir
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self._vector_index = vector_index

    @property
    def vector_index(self) -> Any:
        if self._vector_index is None:
            from stock_engine.rag.index import ChromaVectorIndex
            self._vector_index = ChromaVectorIndex(persist_dir=self.chroma_dir)
        return self._vector_index

    def _fetch_candidate_chunks_sqlite(
        self,
        tickers: Optional[List[str]] = None,
        as_of: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch time-filtered and ticker-filtered candidates from SQLite news store.
        Strictly enforces published_at <= as_of.
        """
        if not self.db_path.exists():
            return []

        query = """
            SELECT DISTINCT a.id, a.title, a.summary, a.source, a.url, a.published_at,
                   COALESCE(group_concat(t.ticker, ','), '') as tickers
            FROM news_articles a
            LEFT JOIN news_ticker t ON a.id = t.doc_id
            WHERE 1=1
        """
        params: List[Any] = []

        # STRICT TIME FILTER
        if as_of is not None:
            query += " AND a.published_at <= ?"
            params.append(as_of)

        query += " GROUP BY a.id"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        chunks: List[Dict[str, Any]] = []
        for row in rows:
            doc_id, title, summary, source, url, pub_at, ticker_str = row
            doc_tickers = [t.strip() for t in ticker_str.split(",") if t.strip()]

            # Ticker filter
            if tickers and not any(t in doc_tickers for t in tickers):
                continue

            text = f"{title}. {summary}".strip()
            chunks.append({
                "chunk_id": f"{doc_id}_c0",
                "doc_id": doc_id,
                "text": text,
                "published_at": pub_at,
                "source": source,
                "url": url,
                "tickers": doc_tickers,
            })

        return chunks

    def retrieve_bm25(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        BM25 keyword retrieval over the pre-filtered candidate chunks.
        """
        if not candidates:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # Fallback simple keyword frequency
            scored = []
            q_tokens = query.lower().split()
            for c in candidates:
                txt = c["text"].lower()
                score = sum(txt.count(tok) for tok in q_tokens)
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in scored[:top_k]]

        tokenized_corpus = [c["text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                item = dict(candidates[idx])
                item["bm25_score"] = float(scores[idx])
                results.append(item)
        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
            score(d) = sum_{retriever} 1.0 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            chunk_map[cid] = item

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = item

        # Sort by combined RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused = []
        for cid in sorted_cids[:top_n]:
            item = dict(chunk_map[cid])
            item["rrf_score"] = round(rrf_scores[cid], 5)
            fused.append(item)
        return fused

    def retrieve(
        self,
        query: str,
        tickers: Optional[List[str]] = None,
        as_of: Optional[int] = None,
        k: int = 5,
        use_reranker: bool = True,
    ) -> List[Evidence]:
        """
        Execute full hybrid pipeline:
        1. Dense retrieval (as_of time filter applied)
        2. Candidate SQLite fetch (as_of time filter applied)
        3. BM25 retrieval over candidate set
        4. Reciprocal Rank Fusion (k=60) -> Top 20
        5. Cross-Encoder reranking -> Top 5 Evidence objects
        """
        # 1. Dense retrieval
        dense_hits = self.vector_index.retrieve_dense(
            query=query,
            tickers=tickers,
            as_of=as_of,
            k=20,
        )

        # 2. SQLite candidates for BM25
        candidates = self._fetch_candidate_chunks_sqlite(tickers=tickers, as_of=as_of)

        # 3. BM25 retrieval
        bm25_hits = self.retrieve_bm25(query=query, candidates=candidates, top_k=20)

        # 4. RRF Fusion
        fused_candidates = self.reciprocal_rank_fusion(dense_hits, bm25_hits, k=60, top_n=20)

        if not fused_candidates:
            return []

        # 5. Cross-Encoder Reranking
        if use_reranker and len(fused_candidates) > 1:
            pairs = [(query, c["text"]) for c in fused_candidates]
            ce_scores = self.reranker.predict(pairs)
            for item, score in zip(fused_candidates, ce_scores):
                item["final_score"] = float(score)
            fused_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        else:
            for item in fused_candidates:
                item["final_score"] = item.get("rrf_score", 0.0)

        # 6. Format top k Evidence objects
        evidence_list: List[Evidence] = []
        for c in fused_candidates[:k]:
            evidence_list.append(
                Evidence(
                    chunk_id=c.get("chunk_id", ""),
                    doc_id=c.get("doc_id", ""),
                    text=c.get("text", ""),
                    score=round(float(c.get("final_score", 0.0)), 4),
                    published_at=int(c.get("published_at", 0)),
                    source=c.get("source", "Unknown"),
                    url=c.get("url", ""),
                )
            )

        return evidence_list

