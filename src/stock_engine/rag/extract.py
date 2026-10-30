"""
Structured financial news event extraction with Pydantic schema validation,
prompt caching, and FinBERT sentiment baseline.

NON-NEGOTIABLE SAFETY RULES:
1. The LLM NEVER decides buy/sell and NEVER predicts prices.
2. Output is strictly validated JSON.
3. evidence_ids must be a strict subset of provided evidence IDs.
4. On schema errors, retry up to 2 times feeding back the error;
   fallback to neutral confidence=0.0 on failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator

from stock_engine.rag.llm import LLMClient, MockDeterministicLLMClient, OllamaClient
from stock_engine.rag.retrieve import Evidence

logger = logging.getLogger(__name__)

EventType = Literal[
    "earnings",
    "guidance",
    "mna",
    "regulatory",
    "management_change",
    "product",
    "macro",
    "litigation",
    "rating_change",
    "dividend_buyback",
    "other",
]

SurpriseType = Literal["positive", "neutral", "negative", "unknown"]


class NewsExtractionResult(BaseModel):
    """Pydantic validated structured news event extraction."""
    ticker: str
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    event_type: EventType
    surprise: SurpriseType
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_ids: List[str]
    summary: str

    @field_validator("summary")
    @classmethod
    def validate_summary_word_count(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 40:
            # Truncate cleanly to 40 words
            return " ".join(words[:40])
        return v


def compute_cache_key(prompt: str, model_name: str, evidence_ids: List[str]) -> str:
    """Deterministic hash key for extraction caching."""
    sorted_ids = ",".join(sorted(evidence_ids))
    content = f"{prompt}|{model_name}|{sorted_ids}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class FinBERTBaseline:
    """
    FinBERT (ProsusAI/finbert) sentiment model for headline-level financial sentiment.
    Returns sentiment in [-1.0, 1.0] (positive=1, neutral=0, negative=-1).
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._init_model()

    def _init_model(self) -> None:
        if self.model_name in ["mock", "mock_fast", "test"]:
            self._pipeline = None
            return

        try:
            from transformers import pipeline
            self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
            logger.info(f"Loaded FinBERT model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load FinBERT ({e}). Using lexical fallback.")
            self._pipeline = None

    def predict_sentiment(self, texts: List[str]) -> float:
        """
        Predict aggregate sentiment in [-1.0, 1.0] across a list of headline texts.
        """
        if not texts:
            return 0.0

        if self._pipeline is not None:
            try:
                results = self._pipeline(texts[:10])
                scores = []
                for r in results:
                    lbl = r["label"].lower()
                    score = float(r["score"])
                    if lbl == "positive":
                        scores.append(score)
                    elif lbl == "negative":
                        scores.append(-score)
                    else:
                        scores.append(0.0)
                return float(np.mean(scores))
            except Exception as e:
                logger.error(f"FinBERT prediction error: {e}")

        # Fallback keyword sentiment dictionary
        pos_words = {"profit", "growth", "surges", "gain", "deal", "dividend", "expansion", "beats", "record"}
        neg_words = {"loss", "decline", "drops", "slumps", "penalty", "fraud", "investigation", "fall", "miss"}

        scores = []
        for text in texts:
            tokens = set(text.lower().split())
            p_hits = len(tokens & pos_words)
            n_hits = len(tokens & neg_words)
            if p_hits + n_hits == 0:
                scores.append(0.0)
            else:
                scores.append((p_hits - n_hits) / (p_hits + n_hits))
        return float(np.mean(scores)) if scores else 0.0


class NewsExtractor:
    """
    Extracts structured facts from retrieved evidence using an LLM
    with schema validation, error feedback retry, and caching.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_path: Path = Path("prompts/extract_v1.txt"),
        finbert: Optional[FinBERTBaseline] = None,
    ) -> None:
        self.llm_client = llm_client or OllamaClient()
        self.prompt_path = prompt_path
        self.finbert = finbert or FinBERTBaseline(model_name="mock_fast")
        self.cache: Dict[str, NewsExtractionResult] = {}
        self._prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if self.prompt_path.exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Extract JSON for ticker {ticker}: {evidence_text}"

    def extract(
        self,
        ticker: str,
        as_of: Any,
        evidence_list: List[Evidence],
    ) -> NewsExtractionResult:
        """
        Extract validated structured news event.
        Guarantees that evidence_ids are a subset of provided evidence.
        """
        if not evidence_list:
            return NewsExtractionResult(
                ticker=ticker,
                sentiment=0.0,
                event_type="other",
                surprise="unknown",
                confidence=0.0,
                evidence_ids=[],
                summary="No news evidence found for this ticker and date.",
            )

        valid_evidence_ids: Set[str] = {e.doc_id for e in evidence_list} | {e.chunk_id for e in evidence_list}

        # Build evidence text block
        evidence_lines = []
        for i, ev in enumerate(evidence_list, 1):
            evidence_lines.append(f"[{ev.doc_id}] {ev.text}")
        evidence_text = "\n".join(evidence_lines)

        prompt = (
            self._prompt_template
            .replace("{ticker}", ticker)
            .replace("{as_of}", str(as_of))
            .replace("{evidence_text}", evidence_text)
        )

        model_name = getattr(self.llm_client, "model", "default")
        cache_key = compute_cache_key(prompt, model_name, list(valid_evidence_ids))

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Retry loop for robust schema validation
        last_error = ""
        current_prompt = prompt

        for attempt in range(3):
            try:
                raw_json = self.llm_client.generate(current_prompt)
                # Clean possible markdown formatting
                clean_json = raw_json.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                parsed = json.loads(clean_json)

                # Validate evidence_ids subset
                claimed_ids = parsed.get("evidence_ids", [])
                filtered_ids = [cid for cid in claimed_ids if cid in valid_evidence_ids]
                parsed["evidence_ids"] = filtered_ids if filtered_ids else [evidence_list[0].doc_id]
                parsed["ticker"] = ticker

                result = NewsExtractionResult.model_validate(parsed)
                self.cache[cache_key] = result
                return result

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                current_prompt = f"{prompt}\n\nPREVIOUS ATTEMPT RETURNED ERROR: {last_error}\nPlease output ONLY valid JSON matching the schema."

        # Fallback neutral result on retry exhaustion
        fallback = NewsExtractionResult(
            ticker=ticker,
            sentiment=0.0,
            event_type="other",
            surprise="unknown",
            confidence=0.0,
            evidence_ids=[evidence_list[0].doc_id],
            summary="Extraction fallback due to schema error.",
        )
        self.cache[cache_key] = fallback
        return fallback

