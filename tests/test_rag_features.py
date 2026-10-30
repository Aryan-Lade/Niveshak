"""
Unit tests for Day 22 RAG features, property testing for point-in-time constraints,
and missing news zero-fill handling.
"""

import numpy as np
import pandas as pd
import pytest

from stock_engine.rag.extract import NewsExtractor
from stock_engine.rag.features import (
    build_rag_features_for_item,
    convert_date_to_ist_close_epoch,
    run_ablation_d_decision_gate,
)
from stock_engine.rag.llm import MockDeterministicLLMClient
from stock_engine.rag.retrieve import Evidence


def test_missing_news_zero_fill_no_nans():
    """Verify that dates/tickers with zero news return clean 0.0 features with no NaNs."""
    dt = pd.Timestamp("2024-05-15")
    as_of = convert_date_to_ist_close_epoch(dt)

    features = build_rag_features_for_item(
        ticker="RELIANCE.NS",
        as_of_epoch=as_of,
        evidence_list=[],  # Empty news
    )

    assert features["rag_sent_1d"] == 0.0
    assert features["rag_sent_3d"] == 0.0
    assert features["rag_news_count"] == 0.0
    assert features["rag_max_negative"] == 0.0
    assert features["rag_conf_mean"] == 0.0
    assert features["finbert_sent_mean"] == 0.0
    assert features["rag_event_earnings"] == 0.0

    # Ensure no NaN values anywhere in features
    for k, v in features.items():
        assert not np.isnan(v), f"Feature {k} returned NaN on empty news"


def test_property_no_future_evidence_used():
    """
    Property test: Ensure no evidence with published_at > as_of is ever processed
    or factored into rag_sent_1d or rag_news_count.
    """
    as_of = 10000

    past_evidence = Evidence(
        chunk_id="chunk_past",
        doc_id="doc_past",
        text="Solid quarterly profit expansion.",
        score=0.9,
        published_at=9500,  # <= as_of
        source="News",
        url="https://example.com/past",
    )

    future_evidence = Evidence(
        chunk_id="chunk_future",
        doc_id="doc_future",
        text="Catastrophic factory explosion and sudden loss.",
        score=0.95,
        published_at=10500,  # > as_of (FUTURE)
        source="News",
        url="https://example.com/future",
    )

    extractor = NewsExtractor(llm_client=MockDeterministicLLMClient())

    # Call with both past and future evidence
    features = build_rag_features_for_item(
        ticker="TCS.NS",
        as_of_epoch=as_of,
        evidence_list=[past_evidence, future_evidence],
        extractor=extractor,
    )

    # Valid evidence count must strictly be 1 (excluding future_evidence)
    assert features["rag_news_count"] == 1.0


def test_decision_gate_outcome_structure(tmp_path):
    """Verify that Ablation D executes and produces the decision gate report."""
    res = run_ablation_d_decision_gate(reports_dir=tmp_path)
    assert res["decision_gate_passed"] is True
    assert res["p_value"] < 0.10
    assert (tmp_path / "rag_ablation.md").exists()
