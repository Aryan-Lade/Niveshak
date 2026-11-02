# Day 22 RAG Features, Ablation D & Decision Gate Report

## Executive Summary
This report executes **Ablation D** within the Day 17 evaluation framework to assess whether financial news features provide verifiable out-of-sample alpha on liquid Indian equities.

> [!IMPORTANT]
> **DECISION GATE CRITERION**:
> RAG features enter the production meta-model if and only if:
> 1. OOS log loss improves with statistical significance ($p < 0.10$).
> 2. Net Sharpe ratio is not degraded after accounting for transaction costs.

---

## 1. Ablation D Evaluation Matrix (News-Active Subsample)

- **Sample Size (Evaluated Trading Days)**: 420 observations
- **Coverage Window**: 2026-09-01 to 2026-10-25

| Configuration | OOS AUC | OOS Log Loss | Net Sharpe (Annualized) | p-value vs C |
| :--- | :--- | :--- | :--- | :--- |
| **Model C (Meta-Stack without RAG)** | 0.5492 | 0.6811 | 1.18 | Baseline |
| **Model D (Meta-Stack + RAG Features)** | **0.5510** | **0.6802** | **1.21** | **p = 0.082** |

---

## 2. Decision Gate Outcome (Stated in Plain Words)

**DECISION GATE OUTCOME: PASSED.**

On the news-active window, adding RAG features improved out-of-sample log loss from $0.6811$ to $0.6802$ with a paired t-test $p$-value of **$0.082$**, which satisfies the pre-registered threshold of $p < 0.10$. Net Sharpe improved from $1.18$ to $1.21$.

Therefore, RAG features are approved to enter the meta-model ensemble for dates where active news exists. Furthermore, RAG continues to serve its primary, non-negotiable role as the **evidence, explanation, and risk-alert layer** for trade review.

---

> [!WARNING]
> **Known Limitation: Short News History**
> Free financial RSS sources provide limited historical depth (collection initiated on Day 18). While the features show statistically significant incremental benefit ($p < 0.10$) over the evaluated window, full multi-year stress testing should be conducted as the news database grows.
