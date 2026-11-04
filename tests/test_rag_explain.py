"""
Unit tests for Day 26 Explain-trade RAG Assistant and Guardrails.

Checks:
1. Fabricated [doc_id] citation is caught and triggers retry / fallback.
2. Investment advice language is strictly detected and rejected.
3. Fallback template produces deterministic, factual explanation.
4. No-evidence condition is handled honestly without hallucinating.
5. Verification of disclaimer presence on every response.
"""

import pytest

from stock_engine.rag.explain import TradeContext, TradeExplainer
from stock_engine.rag.llm import MockDeterministicLLMClient
from stock_engine.rag.retrieve import Evidence


@pytest.fixture
def sample_trade_context():
    ev = Evidence(
        chunk_id="art_100_c0",
        doc_id="art_100",
        text="TCS signs multi-million dollar cloud modernisation contract in Europe.",
        score=0.92,
        published_at=1700000000,
        source="News",
        url="https://example.com/tcs",
    )
    return TradeContext(
        ticker="TCS.NS",
        date="2024-05-15",
        calibrated_prob=0.61,
        regime="bull",
        top_shap_features={"f_ret_20": 0.052, "f_vol_20": -0.015},
        risk_reasons=["Sized 25 shares within 1% risk limit"],
        evidence=[ev],
    )


def test_fabricated_citation_is_caught_and_falls_back(sample_trade_context):
    """If the LLM invents a doc_id that does not exist in evidence, fallback is invoked."""
    hallucinated_text = """
MODEL VIEW: Model probability is 61%.
EVIDENCE FROM NEWS: Growth cited in [fake_doc_999].
RISKS: Tech spend slowdown.
WHAT WOULD CHANGE THIS VIEW: Trend reversal.
"""
    mock_llm = MockDeterministicLLMClient(canned_responses=[hallucinated_text, hallucinated_text])
    explainer = TradeExplainer(llm_client=mock_llm)

    result = explainer.explain(sample_trade_context)

    # Must NOT contain fake_doc_999 in cited IDs
    assert "fake_doc_999" not in result.cited_doc_ids
    assert "Educational research engine. Not financial advice." in result.disclaimer


def test_investment_advice_language_is_rejected(sample_trade_context):
    """If the LLM emits 'you should buy' or 'strong buy', guardrail rejects it."""
    advice_text = """
MODEL VIEW: High momentum.
EVIDENCE FROM NEWS: Expansion in [art_100]. Therefore you should buy this stock now for guaranteed return.
RISKS: None.
WHAT WOULD CHANGE THIS VIEW: Nothing.
"""
    mock_llm = MockDeterministicLLMClient(canned_responses=[advice_text, advice_text])
    explainer = TradeExplainer(llm_client=mock_llm)

    result = explainer.explain(sample_trade_context)

    # Must fall back to safe template
    assert "you should buy" not in result.evidence_from_news.lower()
    assert "guaranteed return" not in result.evidence_from_news.lower()
    assert "Educational research engine" in result.disclaimer


def test_no_evidence_case_handled_honestly():
    """When no news is available, the assistant explicitly states so."""
    ctx_no_news = TradeContext(
        ticker="HDFCBANK.NS",
        date="2024-05-15",
        calibrated_prob=0.55,
        regime="sideways",
        top_shap_features={"f_rsi_14": 0.012},
        risk_reasons=["Standard sizing"],
        evidence=[],
    )

    explainer = TradeExplainer()
    fallback = explainer.build_fallback_explanation(ctx_no_news)

    assert "No relevant news evidence was available" in fallback.evidence_from_news
    assert fallback.cited_doc_ids == []
    assert "Not financial advice" in fallback.disclaimer


def test_valid_grounded_explanation_succeeds(sample_trade_context):
    """When the LLM outputs valid factual citations, it is verified and returned."""
    valid_text = """
MODEL VIEW: Calibrated probability 61% based on momentum features.
EVIDENCE FROM NEWS: Strong enterprise contract expansion noted in [art_100].
RISKS: Key concerns include macro technology spending delays.
WHAT WOULD CHANGE THIS VIEW: Breakdown below moving average support.
"""
    mock_llm = MockDeterministicLLMClient(canned_responses=[valid_text])
    explainer = TradeExplainer(llm_client=mock_llm)

    result = explainer.explain(sample_trade_context)

    assert "art_100" in result.cited_doc_ids
    assert "Not financial advice" in result.disclaimer

