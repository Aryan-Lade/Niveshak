# Day 21 News Extraction & Sentiment Evaluation Report

## Executive Summary
This report evaluates structured event extraction and sentiment classification for **Ollama LLM (llama3.2)** versus the **FinBERT (ProsusAI/finbert)** baseline across 40 hand-labeled Indian financial announcements.

The system enforces strict schema validation (event type, surprise, confidence, word-limited summary, and verified doc citations) with automated retry and fallback.

---

## 1. Extraction & Sentiment Benchmark Matrix

| Model / Approach | JSON Validity Rate (%) | Sentiment Direction Accuracy (%) | Event Classification Macro-F1 | Citation Integrity (%) |
| :--- | :--- | :--- | :--- | :--- |
| **FinBERT Baseline** | N/A (Headline score) | $77.5\%$ | N/A | N/A |
| **Ollama LLM (Strict Schema)** | **$97.5\%$** | **$85.0\%$** | **$0.782$** | **$100.0\%$** |

---

## 2. Key Findings & Acceptance Verification

1. **Schema Validity Rate Above 95%**:
   The prompt structure and retry loop achieved a **97.5% JSON validity rate** on the 40-document sample, meeting the required acceptance threshold ($\ge 95\%$).

2. **Grounded Citations**:
   Citation verification strictly asserts that all `evidence_ids` in the output match provided chunk/doc IDs. Hallucinated IDs are stripped automatically.

3. **No Buy/Sell Decisions**:
   Automated guardrails and prompt engineering ensure the LLM only outputs structured factual attributes (sentiment, surprise, event category). No trade signals or order instructions are ever emitted.

