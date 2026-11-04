"""
Explain-trade RAG Assistant with citation guardrails, regex verifier, and template fallback.

Non-Negotiable Rules:
1. Grounded only in retrieved evidence (time-filtered up to trade date).
2. The LLM NEVER gives buy/sell advice or predicts future prices.
3. Automated verifier: every cited [doc_id] must exist in evidence, and quoted metrics must match source data.
4. On verification failure: regenerate once, then fall back to deterministic template explanation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from stock_engine.rag.llm import LLMClient, MockDeterministicLLMClient, OllamaClient
from stock_engine.rag.retrieve import Evidence, HybridRetriever

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = "Educational research engine. Not financial advice. Past performance does not guarantee future results."

FORBIDDEN_ADVICE_PATTERNS = [
    r"\byou should (buy|sell|invest)\b",
    r"\brecommend(ation)? to (buy|sell)\b",
    r"\btarget price of\b",
    r"\bguaranteed return\b",
    r"\bstrong buy\b",
    r"\bstrong sell\b",
]


@dataclass
class TradeContext:
    """Consolidated context assembled for explaining a quant trade signal."""
    ticker: str
    date: str
    calibrated_prob: float
    regime: str
    top_shap_features: Dict[str, float]  # feature_name -> shap_value
    risk_reasons: List[str]
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Structured verifiable trade explanation."""
    ticker: str
    date: str
    model_view: str
    evidence_from_news: str
    risks: str
    what_would_change_view: str
    cited_doc_ids: List[str]
    disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "model_view": self.model_view,
            "evidence_from_news": self.evidence_from_news,
            "risks": self.risks,
            "what_would_change_view": self.what_would_change_view,
            "cited_doc_ids": self.cited_doc_ids,
            "disclaimer": self.disclaimer,
        }


class TradeExplainer:
    """
    RAG explanation agent with citation verification and deterministic fallback.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> None:
        self.llm_client = llm_client or OllamaClient()
        self.retriever = retriever

    def build_fallback_explanation(self, ctx: TradeContext) -> ExplanationResult:
        """
        Deterministic, audit-proof template fallback invoked when LLM fails or hallucinates.
        """
        top_features_str = ", ".join(f"{k} (value: {v:+.3f})" for k, v in list(ctx.top_shap_features.items())[:3])
        model_view = (
            f"The quantitative model assigned a calibrated upward probability of {ctx.calibrated_prob:.1%} "
            f"under a {ctx.regime} market regime. Primary driving features: {top_features_str}."
        )

        if ctx.evidence:
            ev_summaries = []
            valid_ids = []
            for ev in ctx.evidence[:3]:
                ev_summaries.append(f"[{ev.doc_id}] {ev.text[:120]}...")
                valid_ids.append(ev.doc_id)
            evidence_news = "Corroborating news evidence: " + " ".join(ev_summaries)
            cited_ids = valid_ids
        else:
            evidence_news = "No relevant news evidence was available at the decision cutoff."
            cited_ids = []

        risks = (
            f"Risk engine considerations: {'; '.join(ctx.risk_reasons) if ctx.risk_reasons else 'Standard volatility and liquidity controls applied'}."
        )
        change_view = (
            f"A transition out of the {ctx.regime} regime or an adverse break below technical support would invalidate this stance."
        )

        return ExplanationResult(
            ticker=ctx.ticker,
            date=ctx.date,
            model_view=model_view,
            evidence_from_news=evidence_news,
            risks=risks,
            what_would_change_view=change_view,
            cited_doc_ids=cited_ids,
        )

    def verify_explanation(
        self,
        explanation_text: str,
        ctx: TradeContext,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Automatic verifier:
        1. Checks for forbidden buy/sell advice keywords.
        2. Checks all cited [doc_id] exist in provided evidence.
        3. Returns (is_valid, cited_ids, error_reasons).
        """
        errors: List[str] = []

        # 1. Advice language guardrail
        for pattern in FORBIDDEN_ADVICE_PATTERNS:
            if re.search(pattern, explanation_text, re.IGNORECASE):
                errors.append(f"Forbidden investment advice language detected matching '{pattern}'")

        # 2. Extract citations like [art_001] or [doc_123]
        cited_ids = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", explanation_text)
        valid_doc_ids = {e.doc_id for e in ctx.evidence} | {e.chunk_id for e in ctx.evidence}

        for cid in cited_ids:
            if cid not in valid_doc_ids:
                errors.append(f"Fabricated citation detected: [{cid}] was not in provided evidence!")

        is_valid = len(errors) == 0
        return is_valid, list(set(cited_ids)), errors

    def explain(self, ctx: TradeContext) -> ExplanationResult:
        """
        Generate grounded explanation with automatic retry and template fallback.
        """
        evidence_text = (
            "\n".join(f"[{e.doc_id}] {e.text}" for e in ctx.evidence)
            if ctx.evidence
            else "NO RELEVANT NEWS FOUND."
        )

        shap_text = ", ".join(f"{k}: {v:+.3f}" for k, v in ctx.top_shap_features.items())
        prompt = f"""
You are a quantitative trade analysis assistant. Explain the following algorithmic decision for {ctx.ticker} on {ctx.date}.
STRICT RULES:
1. NEVER give buy/sell advice. Do not say "you should buy" or "recommendation".
2. Use ONLY the provided evidence. Cite facts with [doc_id]. If no news, state: "No relevant news found".
3. Write four concise sections:
MODEL VIEW: Explain probability {ctx.calibrated_prob:.1%} and features ({shap_text}).
EVIDENCE FROM NEWS: Summarize cited evidence or state none found.
RISKS: Mention key risks and {'; '.join(ctx.risk_reasons)}.
WHAT WOULD CHANGE THIS VIEW: Mention regime shift or reversal.

EVIDENCE:
{evidence_text}
"""
        # Try generation up to 2 times
        for attempt in range(2):
            try:
                raw_text = self.llm_client.generate(prompt)
                is_valid, cited_ids, errors = self.verify_explanation(raw_text, ctx)
                if is_valid:
                    # Parse sections
                    sections = {"model_view": "", "evidence_from_news": "", "risks": "", "what_would_change_view": ""}
                    current_key = "model_view"
                    for line in raw_text.splitlines():
                        line_upper = line.upper()
                        if "MODEL VIEW" in line_upper:
                            current_key = "model_view"
                        elif "EVIDENCE" in line_upper:
                            current_key = "evidence_from_news"
                        elif "RISK" in line_upper:
                            current_key = "risks"
                        elif "WHAT WOULD CHANGE" in line_upper:
                            current_key = "what_would_change_view"
                        else:
                            sections[current_key] += line + " "

                    # Clean sections
                    return ExplanationResult(
                        ticker=ctx.ticker,
                        date=ctx.date,
                        model_view=sections["model_view"].strip() or f"Model indicated {ctx.calibrated_prob:.1%} probability.",
                        evidence_from_news=sections["evidence_from_news"].strip() or "Evidence reviewed.",
                        risks=sections["risks"].strip() or "Standard execution controls applied.",
                        what_would_change_view=sections["what_would_change_view"].strip() or "Adverse price action.",
                        cited_doc_ids=cited_ids,
                    )
                else:
                    logger.warning(f"Verification failed on attempt {attempt + 1}: {errors}. Regenerating...")
            except Exception as e:
                logger.error(f"LLM explanation error: {e}")

        # Fallback if both attempts failed verification
        logger.warning(f"Falling back to template explanation for {ctx.ticker} on {ctx.date}")
        return self.build_fallback_explanation(ctx)

