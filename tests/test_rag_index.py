"""
Unit tests for Day 19 time-filtered vector indexing and dense retrieval.
"""

import json
import time
import numpy as np
import pytest

from stock_engine.rag.index import (
    ChromaVectorIndex,
    DenseEmbedder,
    chunk_document,
)


@pytest.fixture
def test_vector_index(tmp_path):
    """Fixture providing a temporary ChromaVectorIndex."""
    embedder = DenseEmbedder(model_name="mock_fast")
    return ChromaVectorIndex(persist_dir=tmp_path / "chroma_test", embedder=embedder)


def test_metadata_roundtrip_and_chunking():
    """Verify document chunking preserves metadata and creates deterministic IDs."""
    chunks = chunk_document(
        doc_id="doc_123",
        title="Reliance Q3 earnings analysis",
        summary="Consolidated profit increased significantly year over year.",
        tickers=["RELIANCE.NS"],
        published_at=1700000000,
        source="Reuters",
    )
    assert len(chunks) >= 1
    c0 = chunks[0]
    assert c0["doc_id"] == "doc_123"
    assert c0["chunk_id"] == "doc_123_c0"
    assert c0["published_at"] == 1700000000
    assert c0["source"] == "Reuters"
    assert "RELIANCE.NS" in json.loads(c0["tickers"])


def test_idempotent_reindexing(test_vector_index):
    """Verify re-indexing the same chunks does not duplicate entries."""
    chunks = chunk_document(
        doc_id="doc_456",
        title="TCS signs multimillion digital deal",
        summary="Airtel and TCS collaborate on enterprise systems.",
        tickers=["TCS.NS", "BHARTIARTL.NS"],
        published_at=1700000500,
        source="Bloomberg",
    )

    # First index pass
    test_vector_index.index_chunks(chunks)
    count1 = test_vector_index.collection.count()
    assert count1 == len(chunks)

    # Second index pass (must be idempotent)
    test_vector_index.index_chunks(chunks)
    count2 = test_vector_index.collection.count()
    assert count2 == count1


def test_property_time_filter(test_vector_index):
    """
    CRITICAL PROPERTY TEST:
    For any arbitrary as_of cutoff, NO returned chunk may have published_at > as_of.
    """
    timestamps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    all_chunks = []
    for i, ts in enumerate(timestamps):
        all_chunks.extend(
            chunk_document(
                doc_id=f"doc_{i}",
                title=f"Market event {i} regarding HDFC Bank growth",
                summary="Banking loan demand remains healthy.",
                tickers=["HDFCBANK.NS"],
                published_at=ts,
                source="NewsWire",
            )
        )

    test_vector_index.index_chunks(all_chunks)

    # Test with random as_of cutoffs
    np.random.seed(42)
    random_as_ofs = np.random.randint(500, 9000, size=25)

    for as_of in random_as_ofs:
        results = test_vector_index.retrieve_dense(
            query="HDFC Bank loan growth",
            tickers=["HDFCBANK.NS"],
            as_of=int(as_of),
            k=5,
        )
        for r in results:
            assert r["published_at"] <= as_of, (
                f"Time leak detected! Chunk published_at={r['published_at']} > as_of={as_of}"
            )


def test_empty_result_handled(test_vector_index):
    """Verify empty results are handled gracefully when criteria cannot be met."""
    chunks = chunk_document(
        doc_id="doc_future",
        title="Future announcement",
        summary="Content.",
        tickers=["INFY.NS"],
        published_at=5000,
        source="Feed",
    )
    test_vector_index.index_chunks(chunks)

    # 1. Query with as_of before document was published
    res_early = test_vector_index.retrieve_dense(
        query="announcement",
        tickers=["INFY.NS"],
        as_of=2000,
    )
    assert res_early == []

    # 2. Query with non-matching ticker
    res_mismatch = test_vector_index.retrieve_dense(
        query="announcement",
        tickers=["RELIANCE.NS"],
        as_of=6000,
    )
    assert res_mismatch == []
