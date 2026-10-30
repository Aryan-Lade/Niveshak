"""
Unit tests for Day 21 structured LLM news extraction, retry loops, and FinBERT baseline.
"""

import json
import pytest

from stock_engine.rag.extract import (
    FinBERTBaseline,
    NewsExtractionResult,
    NewsExtractor,
)
from stock_engine.rag.llm import MockDeterministicLLMClient
from stock_engine.rag.retrieve import Evidence


def test_schema_validation_and_retry_logic(tmp_path):
    """
    Test retry logic:
    - First response: malformed JSON
    - Second response: valid structured JSON
    """
    invalid_resp = "This is not JSON: {ticker: invalid...}"
    valid_resp = json.dumps({
        "ticker": "RELIANCE.NS",
        "sentiment": 0.65,
        "event_type": "earnings",
        "surprise": "positive",
        "confidence": 0.90,
        "evidence_ids": ["art_101"],
        "summary": "Reliance Q3 net profit rose 11% YoY driven by retail performance.",
    })

    mock_client = MockDeterministicLLMClient(canned_responses=[invalid_resp, valid_resp])
    extractor = NewsExtractor(llm_client=mock_client)

    evidence = [
        Evidence(
            chunk_id="art_101_c0",
            doc_id="art_101",
            text="Reliance profit grew 11%.",
            score=0.9,
            published_at=1000,
            source="News",
            url="https://example.com/1",
        )
    ]

    result = extractor.extract("RELIANCE.NS", as_of=2000, evidence_list=evidence)

    assert result.ticker == "RELIANCE.NS"
    assert result.sentiment == 0.65
    assert result.event_type == "earnings"
    assert result.surprise == "positive"
    assert result.evidence_ids == ["art_101"]
    assert mock_client.call_count == 2  # Proves retry was invoked


def test_invalid_evidence_ids_rejected():
    """Verify fabricated evidence IDs not in the evidence list are rejected."""
    hallucinated_resp = json.dumps({
        "ticker": "TCS.NS",
        "sentiment": 0.4,
        "event_type": "product",
        "surprise": "positive",
        "confidence": 0.7,
        "evidence_ids": ["fake_doc_999", "legit_doc_01"],
        "summary": "TCS partners with cloud vendor.",
    })

    mock_client = MockDeterministicLLMClient(canned_responses=[hallucinated_resp])
    extractor = NewsExtractor(llm_client=mock_client)

    evidence = [
        Evidence(
            chunk_id="legit_doc_01_c0",
            doc_id="legit_doc_01",
            text="TCS cloud partner announcement.",
            score=0.8,
            published_at=1000,
            source="News",
            url="https://example.com/2",
        )
    ]

    result = extractor.extract("TCS.NS", as_of=2000, evidence_list=evidence)

    # fake_doc_999 must NOT be present in output
    assert "fake_doc_999" not in result.evidence_ids
    assert "legit_doc_01" in result.evidence_ids


def test_cache_hit_avoids_redundant_llm_calls():
    """Verify repeated extraction of same prompt/evidence hits cache without calling LLM."""
    mock_client = MockDeterministicLLMClient()
    extractor = NewsExtractor(llm_client=mock_client)

    evidence = [
        Evidence(
            chunk_id="c1",
            doc_id="d1",
            text="HDFC Bank growth announcement.",
            score=0.85,
            published_at=1000,
            source="News",
            url="https://example.com/3",
        )
    ]

    # First call
    _ = extractor.extract("HDFCBANK.NS", as_of=2000, evidence_list=evidence)
    assert mock_client.call_count == 1

    # Second call with identical input must hit cache
    _ = extractor.extract("HDFCBANK.NS", as_of=2000, evidence_list=evidence)
    assert mock_client.call_count == 1  # No additional LLM call


def test_no_order_output():
    """Verify that NewsExtractionResult contains strictly factual attributes and no order commands."""
    res = NewsExtractionResult(
        ticker="INFY.NS",
        sentiment=0.5,
        event_type="earnings",
        surprise="positive",
        confidence=0.8,
        evidence_ids=["doc_1"],
        summary="Infosys reports strong guidance.",
    )
    # Ensure there is no buy/sell command field
    schema_fields = NewsExtractionResult.model_fields.keys()
    assert "order" not in schema_fields
    assert "action" not in schema_fields
    assert "quantity" not in schema_fields

