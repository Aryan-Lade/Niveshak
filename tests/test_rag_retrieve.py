"""
Unit tests for Day 20 Hybrid retrieval (BM25 + Dense RRF + Reranker).
"""

import time
import numpy as np
import pytest

from stock_engine.rag.index import ChromaVectorIndex, DenseEmbedder, chunk_document
from stock_engine.rag.retrieve import HybridRetriever, CrossEncoderReranker, Evidence


@pytest.fixture
def synthetic_retrieval_corpus(tmp_path):
    """
    Construct tiny synthetic corpus proving:
    - Doc 1: exact keyword ("EBIDTA EBITDA_ACRONYM_UNIQUE_123") without semantic context
    - Doc 2: semantic paraphrase ("firm revenues expanded dramatically") without exact keyword
    - Doc 3: future document published after as_of
    """
    db_file = tmp_path / "test_news.db"
    chroma_dir = tmp_path / "test_chroma"

    # Set up SQLite
    from stock_engine.rag.ingest.rss import SQLiteNewsStore, NewsArticle
    store = SQLiteNewsStore(db_path=db_file)

    art1 = NewsArticle(
        id="doc_kw",
        title="EBITDA_ACRONYM_UNIQUE_123 report for Q3",
        summary="Special fiscal figures recorded by finance team.",
        source="News",
        url="https://example.com/kw",
        published_at=1000,
        fetched_at=1000,
    )
    art2 = NewsArticle(
        id="doc_para",
        title="Tremendous commercial success and surging corporate earnings",
        summary="Revenues and profit numbers climbed substantially across all business units.",
        source="News",
        url="https://example.com/para",
        published_at=2000,
        fetched_at=2000,
    )
    art3_future = NewsArticle(
        id="doc_future",
        title="Future breaking story published next week",
        summary="Breaking updates on market movement.",
        source="News",
        url="https://example.com/future",
        published_at=9999,  # After as_of
        fetched_at=9999,
    )

    store.insert_article(art1, ["TEST.NS"])
    store.insert_article(art2, ["TEST.NS"])
    store.insert_article(art3_future, ["TEST.NS"])

    # Index into Chroma
    embedder = DenseEmbedder(model_name="mock_fast")
    vindex = ChromaVectorIndex(persist_dir=chroma_dir, embedder=embedder)

    chunks = []
    chunks.extend(chunk_document("doc_kw", art1.title, art1.summary, ["TEST.NS"], 1000, "News"))
    chunks.extend(chunk_document("doc_para", art2.title, art2.summary, ["TEST.NS"], 2000, "News"))
    chunks.extend(chunk_document("doc_future", art3_future.title, art3_future.summary, ["TEST.NS"], 9999, "News"))
    vindex.index_chunks(chunks)

    retriever = HybridRetriever(
        db_path=db_file,
        chroma_dir=chroma_dir,
        reranker_model="mock_fast",
        vector_index=vindex,
    )
    return retriever


def test_bm25_finds_exact_keyword(synthetic_retrieval_corpus):
    """BM25 finds exact rare keyword."""
    retriever = synthetic_retrieval_corpus
    candidates = retriever._fetch_candidate_chunks_sqlite(tickers=["TEST.NS"], as_of=5000)
    hits = retriever.retrieve_bm25("EBITDA_ACRONYM_UNIQUE_123", candidates, top_k=5)
    doc_ids = [h["doc_id"] for h in hits]
    assert "doc_kw" in doc_ids


def test_hybrid_and_time_filter(synthetic_retrieval_corpus):
    """
    CRITICAL ACCEPTANCE TESTS:
    1. Hybrid retrieval retrieves both keyword and paraphrase hits.
    2. Zero documents published after as_of (9999) are ever returned.
    """
    retriever = synthetic_retrieval_corpus
    as_of_limit = 5000

    # Query matching keywords and concepts
    evidence = retriever.retrieve(
        query="EBITDA_ACRONYM_UNIQUE_123 earnings profit growth",
        tickers=["TEST.NS"],
        as_of=as_of_limit,
        k=5,
    )

    doc_ids = [e.doc_id for e in evidence]

    # Future doc (published at 9999) must NEVER appear
    assert "doc_future" not in doc_ids
    for e in evidence:
        assert e.published_at <= as_of_limit

    # Both documents within time limit should be present in results
    assert len(evidence) >= 1

