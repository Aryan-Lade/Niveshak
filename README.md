# Niveshak (निवेशक)
![Project](https://img.shields.io/badge/Project-Niveshak-brightgreen)
![Market](https://img.shields.io/badge/Market-NSE%20India%20(NIFTY%2050)-orange)
![Architecture](https://img.shields.io/badge/Architecture-Multimodal%20Quant%20%2B%20RAG-blue)
![Pipeline](https://img.shields.io/badge/Validation-Purged%20Walk--Forward%20w%2F%20Embargo-purple)
![Cost Model](https://img.shields.io/badge/Cost%20Model-Real%20NSE%20%2F%20Groww%20Fees-yellow)
![Stack](https://img.shields.io/badge/Stack-100%25%20Free%20Tier%20%26%20Local-success)
![Status](https://img.shields.io/badge/Status-Active%20Research%20Engine-brightgreen)

**Niveshak (निवेशक) — Multimodal Stock Intelligence Engine for Indian Equities**  
*An educational, research-grade automated trading system combining leakage-free machine learning, regime detection, grounded financial news RAG, and institutional risk management.*

> [!NOTE]  
> **Reality Check on '100% Accuracy'**  
> No authentic ML system predicts stock prices with 100% certainty. Daily directional hit rates for rigorous, leakage-free models on liquid equities typically hover around **50%–56%**. If a daily financial model demonstrates 65%+ out-of-sample accuracy, it is a data leakage bug, not an alpha breakthrough. Niveshak is built from the ground up to eliminate look-ahead bias and measure genuine out-of-sample edge net of all statutory Indian equity transaction costs.

> [!IMPORTANT]  
> **Zero-Lookahead Guarantee**  
> Information at trading date $t$ only incorporates data available up to market close on $t$. All order executions take place strictly at the **OPEN of $t+1$**. Every feature transformation, market-wide rank, PIT alignment step, and model inference is subjected to automated truncation tests before deployment.

> [!WARNING]  
> **LLM Decision Boundary: Facts & Citations Only**  
> The Large Language Model (LLM) **never** issues buy/sell decisions and never predicts prices. It functions strictly as an evidence extraction engine: parsing timestamped news into structured JSON events with verified doc-ID citations, computing sentiment scores, and providing verifiable explanations for human traders.

> [!TIP]  
> **100% Free-Tier & Privacy-First Architecture**  
> Built entirely without expensive commercial data feeds: utilizes `yfinance` daily bars, public RSS feeds, local embeddings (`BAAI/bge-small-en-v1.5`), cross-encoder rerankers, local `Ollama` / `FinBERT`, `ChromaDB` vector storage, local `MLflow` tracking, and an in-memory/SQLite `PaperBroker`.

---

### 5 Core Architectural Decisions

| Decision | Implementation | Rationale |
|:---|:---|:---|
| **Zero Look-Ahead with PIT Alignment** | Signals at close $t$, execution at open $t+1$, lagged global macro series | Eliminates intraday look-ahead bias and accounts for international time-zone differences (US/Crude close after Indian market). |
| **Purged Walk-Forward Cross-Validation with Embargo** | Expanding train windows, purged label overlaps ($[t, t_1]$), and post-test embargo periods | Standard K-Fold CV creates massive information leakage due to overlapping multi-day labels and serially correlated financial returns. |
| **Realistic Transaction Cost Model** | Configurable NSE brokerage, STT/CTT, stamp duty, exchange charges, SEBI fees, GST, DP charges, and slippage | Gross returns in Indian equities are heavily degraded by turnover taxes and impact costs; models must prove baseline-beating Sharpe net of costs. |
| **Bounded RAG with Time Filters & Decision Gate** | Dense + BM25 hybrid search, RRF fusion, reranker, and `published_at <= as_of` filtering | Prevents future news leakage; news features only enter the trading model if ablation benchmarks prove statistically significant gain ($p < 0.10$). |
| **Deterministic Risk Engine & Persisted Kill Switch** | Hard position caps, ATR sizing, daily loss limit (-2%), peak drawdown circuit breaker (-10%) in SQLite | Protects capital independently of ML predictions; survives application restarts and enforces strict paper execution by default. |

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph DATA["📊 DATA & INGESTION LAYER"]
        A1["📈 Daily OHLCV\n20 Liquid NSE Tickers\nyfinance .NS"]:::data
        A2["🌐 Macro & Global Indices\nNIFTY 50 · INDIA VIX\nUSDINR · S&P 500 · Crude"]:::data
        A3["📰 Financial News\nPublic RSS Feeds\nTimestamped Ingestion"]:::data
        A4["⏱️ Point-in-Time (PIT) Engine\nLag 1 for US/Crude · Lag 0 for NSE\nForward-fill max 3 days"]:::data
    end

    subgraph FEAT["⚙️ FEATURE & LABELING PIPELINE"]
        B1["📐 Vectorized Technicals\nRSI · ATR · MACD · Bollinger\nOvernight Gap · Volatility"]:::feat
        B2["🌐 Cross-Sectional Context\nUniverse Percentile Ranks\nRelative Strength vs NIFTY"]:::feat
        B3["🎯 Triple-Barrier Labels\nVolatility-Scaled Barriers\nExit t1 · Uniqueness Weights"]:::feat
    end

    subgraph MODELS["🧠 MULTIMODAL INTELLIGENCE & REGIMES"]
        C1["🌳 LightGBM Classifier\nPurged Walk-Forward Folds\nOptuna Inner Tuning · Early Stop"]:::model
        C2["🔁 PyTorch GRU\n30-Day Sequence Modeling\nWindow ends strictly at date t"]:::model
        C3["🌊 Forward-Filtered HMM\n3 States: Bull · Sideways · Bear\nNo Forward-Backward Leakage"]:::model
        C4["📚 Grounded News RAG\nChromaDB + BM25 · BGE Embeddings\nReranker · Ollama / FinBERT"]:::rag
        C5["🧬 Meta-Model Stacking\nRegularized Logistic Regression\nBase OOS Probabilities + Regime"]:::model
    end

    subgraph RISK["🛡️ RISK ENGINE & CAPITAL PRESERVATION"]
        D1["📊 Position Sizing\n1% Risk per Trade via ATR Stop\nMax 5 Positions · Max 20% Weight"]:::risk
        D2["🛑 Circuit Breakers\n-2% Daily Loss · -10% Peak Drawdown\nPersisted SQLite Kill Switch"]:::risk
        D3["💰 Realistic Transaction Costs\nBrokerage · STT · Exchange · Stamp Duty\nGST · DP Charges · Slippage"]:::risk
    end

    subgraph EXEC["🚀 EXECUTION, INTERFACES & MONITORING"]
        E1["📝 PaperBroker Simulator\nNext-Open Execution\nSQLite Trade Ledger"]:::exec
        E2["⚡ FastAPI Backend\nAsync REST Endpoints\nKey Auth · OpenAPI Docs"]:::exec
        E3["🖥️ Streamlit Multi-Page UI\nSignals · Portfolio · SHAP Insights\nRAG Evidence · Risk Controls"]:::exec
        E4["📡 Drift & Ops Monitor\nPSI · KS-Test · Rolling Sharpe\nAPScheduler Daily Automation"]:::exec
    end

    A1 --> A4
    A2 --> A4
    A3 --> C4
    A4 --> B1
    A4 --> B2
    A1 --> B3
    B1 --> C1
    B2 --> C1
    B1 --> C2
    B2 --> C3
    B3 --> C1
    B3 --> C2
    C1 --> C5
    C2 --> C5
    C3 --> C5
    C4 -.->|Decision Gate Passed| C5
    C5 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> E1
    E1 --> E2
    E2 --> E3
    E1 --> E4

    classDef data   fill:#1a2b4c,stroke:#3b82f6,color:#e0edff,font-weight:bold
    classDef feat   fill:#163832,stroke:#10b981,color:#d1fae5,font-weight:bold
    classDef model  fill:#311b58,stroke:#8b5cf6,color:#ede9fe,font-weight:bold
    classDef rag    fill:#3b1e39,stroke:#ec4899,color:#fce7f3,font-weight:bold
    classDef risk   fill:#4a1515,stroke:#ef4444,color:#fee2e2,font-weight:bold
    classDef exec   fill:#2d3748,stroke:#06b6d4,color:#cffafe,font-weight:bold
```

---

## 2. Leakage-Free Quantitative Pipeline

```mermaid
flowchart LR
    subgraph SPLIT["1. PURGED WALK-FORWARD CV"]
        direction TB
        F1["Training Window\nExpanding (e.g. 3+ Years)"]:::cv
        F2["Purge Window\nDrop labels overlapping Test"]:::cvpurge
        F3["Embargo Window\nDrop trailing horizon days"]:::cvembargo
        F4["Out-of-Sample Test Window\n6 Months Forward"]:::cvtest
        F1 --> F2 --> F3 --> F4
    end

    subgraph FIT["2. MODEL TRAINING & CALIBRATION"]
        direction TB
        M1["LightGBM & GRU\nTrain on F1 weights"]:::train
        M2["Inner Time-Splits\nHyper-parameter tuning"]:::train
        M3["Isotonic / Platt Calibration\nFitted strictly on Train Inner Validation"]:::train
        M1 --> M2 --> M3
    end

    subgraph OOS["3. OUT-OF-SAMPLE INFERENCE & AUDIT"]
        direction TB
        O1["Raw Probabilities\nEvaluated on F4 only"]:::eval
        O2["Calibrated Signals\nTop-K Quantile Selection"]:::eval
        O3["Leakage Audit Verification\nShuffle labels AUC ~ 0.50\nTruncation test across cuts"]:::eval
        O1 --> O2 --> O3
    end

    SPLIT --> FIT --> OOS

    classDef cv fill:#1e293b,stroke:#64748b,color:#f8fafc
    classDef cvpurge fill:#450a0a,stroke:#dc2626,color:#fef2f2
    classDef cvembargo fill:#713f12,stroke:#eab308,color:#fefce8
    classDef cvtest fill:#064e3b,stroke:#10b981,color:#ecfdf5
    classDef train fill:#312e81,stroke:#6366f1,color:#e0e7ff
    classDef eval fill:#134e4a,stroke:#14b8a6,color:#ccfbf1
```

---

## 3. Financial News RAG Pipeline (Grounded Evidence)

```mermaid
flowchart TD
    subgraph INGEST["1. INGESTION & NORMALIZATION"]
        N1["RSS Ingest\nFinancial feeds\nTitle · Summary · Published UTC"]:::rag_node
        N2["Deduplication\nSHA-1 URL hash + Fuzzy Title"]:::rag_node
        N3["Ticker Linking\nRegex word boundary\ne.g., 'TCS' vs 'Tata Consultancy'"]:::rag_node
        N4["Persistent Store\nSQLite news.db\nFTS5 Full-Text Search"]:::rag_node
        N1 --> N2 --> N3 --> N4
    end

    subgraph INDEX["2. TIME-FILTERED RETRIEVAL"]
        V1["Text Chunking\n300-500 tokens\nTokenizer overlap"]:::rag_node
        V2["Dense Vector Store\nChromaDB\nBAAI/bge-small-en-v1.5"]:::rag_node
        V3["Hybrid Candidates\nBM25 (Exact Keywords) +\nDense (Semantic Concepts)"]:::rag_node
        V4["Reciprocal Rank Fusion (RRF)\nk = 60\nTop 20 candidates"]:::rag_node
        V5["Cross-Encoder Reranker\nms-marco-MiniLM-L-6-v2\nSelect Top 5 Evidence Items"]:::rag_node
        N4 --> V1 --> V2
        N4 --> V3
        V2 --> V3 --> V4 --> V5
    end

    subgraph EXTRACT["3. STRICT EXTRACTION & CITATION"]
        E1["Strict As-Of Filter\npublished_at <= as_of (15:30 IST)\nZero future news leakage"]:::rag_filter
        E2["Structured LLM Extraction (Ollama)\nPydantic Schema: Sentiment [-1, 1],\nEvent type, Surprise, Confidence"]:::rag_node
        E3["Automatic Fact & Citation Verifier\nAssert doc_ids exist in evidence\nRegex check numbers against context"]:::rag_node
        E4["FinBERT Parallel Baseline\nPer-headline sentiment benchmark"]:::rag_node
        V5 --> E1 --> E2 --> E3
        E1 --> E4
    end

    classDef rag_node fill:#2e1065,stroke:#a855f7,color:#faf5ff
    classDef rag_filter fill:#701a75,stroke:#f472b6,color:#fdf2f8,font-weight:bold
```

---

## 4. Execution & Risk Engine Pipeline

```mermaid
flowchart TD
    SIG["Calibrated Probability Signal\nGenerated at close of date t"]:::input_node

    subgraph RISK_RULES["🛡️ DETERMINISTIC RISK CHECK"]
        R1{"Persisted Kill Switch\nTripped in SQLite?"}:::gate
        R2{"Peak Drawdown <= -10%\nDaily Loss <= -2%?"}:::gate
        R3{"Price Sanity &\nLiquidity Filters OK?"}:::gate
        R4{"News Risk Review\nSentiment <= -0.7 & Conf >= 0.7?"}:::gate
        R5["Position Sizing\nRisk 1% Equity / (k * ATR)\nMax 5 Positions · Max 20% Single Stock"]:::action
    end

    subgraph EXEC_SIM["⚙️ EXECUTION ENGINE"]
        X1["Queue Order\nDeterministic client_order_id"]:::action
        X2["Fill at OPEN of t+1\nFree yfinance next-open price"]:::action
        X3["Apply Full Cost Model\nBrokerage + STT + Taxes + Slippage"]:::action
        X4["Update Ledgers\nCash Ledger · Portfolio · Equity Curve"]:::action
    end

    SIG --> R1
    R1 -- "Yes (Tripped)" --> REJECT["⛔ Order Rejected\nLogged to risk_events"]:::reject
    R1 -- "No" --> R2
    R2 -- "Violated" --> TRIP["🚨 Trip Kill Switch\nManual reset required"]:::reject
    R2 -- "Passed" --> R3
    R3 -- "Passed" --> R4
    R4 -- "Passed" --> R5
    R5 --> X1 --> X2 --> X3 --> X4

    classDef input_node fill:#1e3a8a,stroke:#60a5fa,color:#eff6ff,font-weight:bold
    classDef gate fill:#374151,stroke:#9ca3af,color:#f9fafb
    classDef action fill:#064e3b,stroke:#34d399,color:#ecfdf5
    classDef reject fill:#7f1d1d,stroke:#f87171,color:#fef2f2,font-weight:bold
```

---

## 5. Objectives & Core Principles

- **No Look-Ahead Guarantee**: Features at date $t$ use data up to the close of $t$ only. Execution occurs strictly at the open of $t+1$. Truncation tests ensure $f(df[:t]) \equiv f(df)[t]$.
- **Leakage-Free Validation**: Strictly time-ordered purged walk-forward cross-validation with embargo. Overlapping labels are removed from training folds. Holdout data ($2025\text{-}10\text{-}01$ onward) is sealed until final evaluation.
- **Realistic Friction Accounting**: Performance is evaluated net of all statutory Indian equity transaction costs (brokerage, STT/CTT, exchange charges, SEBI fees, stamp duty, GST, DP charges, and slippage).
- **Grounded Financial RAG**: LLMs are bounded strictly to structured fact extraction and cited trade explanations. The LLM never places orders or predicts prices.
- **Ablation Decision Gates**: Experimental components (Regime, GRU, RAG) must demonstrate statistically significant improvement over baseline models ($p < 0.10$) before inclusion in production ensembles.
- **Fail-Safe Execution**: Default execution mode is strictly `PAPER`. Live orders require explicit CLI confirmations, API keys, and untripped kill switches.

---

## 6. Indian Equity Universe & Data Sources

### Equity Universe (20 Liquid NIFTY Stocks)
| Sector | Tickers |
|:---|:---|
| **Banking & Finance** | `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `KOTAKBANK.NS`, `AXISBANK.NS`, `BAJFINANCE.NS` |
| **Information Technology** | `TCS.NS`, `INFY.NS`, `WIPRO.NS` |
| **Oil, Gas & Energy** | `RELIANCE.NS`, `ONGC.NS`, `NTPC.NS` |
| **Consumer Goods & Retail** | `ITC.NS`, `HINDUNILVR.NS`, `TITAN.NS`, `ASIANPAINT.NS` |
| **Automobiles & Industrials**| `MARUTI.NS`, `LT.NS`, `BHARTIARTL.NS` |
| **Pharmaceuticals** | `SUNPHARMA.NS` |

### Benchmark & Macro Indicators
- **Benchmark Index**: `^NSEI` (NIFTY 50)
- **Volatility**: `^INDIAVIX`
- **Currency**: `USDINR=X` (Lag 1)
- **Global Equities**: `^GSPC` (S&P 500, Lag 1)
- **Commodities**: `CL=F` (Crude Oil, Lag 1)

---

## 7. Transaction Cost Engine (Indian Equities)

All order fills model the realistic cost structure of Indian cash/delivery trading configured in `configs/costs.yaml`:

| Component | Rate / Rule | Applicable Side |
|:---|:---|:---|
| **Brokerage** | $\min(\text{INR } 20.0, 0.05\% \times \text{Turnover})$ | Buy & Sell |
| **STT / CTT** | $0.10\%$ on turnover | Buy & Sell (Delivery) |
| **Exchange Transaction Fee** | $0.00325\%$ on turnover | Buy & Sell (NSE) |
| **SEBI Turnover Fee** | $0.0001\%$ on turnover | Buy & Sell |
| **Stamp Duty** | $0.015\%$ on turnover | Buy side only |
| **GST** | $18.0\%$ on $(\text{Brokerage} + \text{Exchange} + \text{SEBI Fee})$ | Buy & Sell |
| **DP Charges** | $\text{INR } 13.50$ (flat) | Sell side only |
| **Slippage** | $2.0 \text{ bps}$ ($0.02\%$) one-way | Positive on Buy, Negative on Sell |
| **Fallback Hurdle** | $25.0 \text{ bps}$ ($0.25\%$) round-trip | Fallback if any component is missing |

---

## 8. 30-Day Engineering Roadmap

```
Week 1: Foundation & Data
├── Day 1: Clean Repo Skeleton, Configs & Tooling
├── Day 2: Daily OHLCV Price Cache & Data Quality Audit
├── Day 3: Macro Panel & Point-in-Time (PIT) Alignment
├── Day 4: Vectorized Leakage-Safe Technical Indicators
├── Day 5: Cross-Sectional & Market Context Features
├── Day 6: Volatility-Scaled Triple-Barrier Labels
└── Day 7: Model Matrix Build, EDA & Leakage Audit

Week 2: Backtest & Core ML
├── Day 8: Indian Equity Transaction Cost Engine
├── Day 9: Next-Open Backtest Engine & Metrics
├── Day 10: Quantitative Baseline Strategies (NIFTY B&H, SMA, Momentum)
├── Day 11: Purged Walk-Forward Cross-Validation with Embargo
├── Day 12: LightGBM Walk-Forward Pipeline with MLflow
├── Day 13: Nested Optuna Hyper-Parameter Tuning
└── Day 14: Probability Calibration, Top-K Signals & OOS Backtest

Week 3: Advanced ML & Grounded RAG
├── Day 15: PyTorch GRU Sequence Model
├── Day 16: Forward-Filtered Gaussian HMM Regime Detector
├── Day 17: Stacking Meta-Model & Ablation Framework
├── Day 18: Financial News RSS Ingestion & Ticker Linking
├── Day 19: Time-Filtered Vector Indexing (ChromaDB + BGE)
├── Day 20: Hybrid Retrieval (BM25 + Dense) & Cross-Encoder Reranker
├── Day 21: Structured LLM Extraction (Ollama) & FinBERT Baseline
└── Day 22: RAG Feature Matrix & Ablation D Decision Gate

Week 4: Product, Safety & Delivery
├── Day 23: Deterministic Risk Engine & Persisted Kill Switch
├── Day 24: Broker Interface, PaperBroker & Daily Runner
├── Day 25: FastAPI Backend with Security Guardrails
├── Day 26: Explain-Trade RAG Assistant with Citation Verifier
├── Day 27: Multi-Page Interactive Streamlit Dashboard
├── Day 28: GrowwBroker Adapter (Dry-Run by Default)
├── Day 29: APScheduler Daily Cycle, Drift Monitoring & Docker
└── Day 30: Final Hardening, Holdout Evaluation & Documentation
```

---

## 9. Repository Structure

```text
Niveshak/
├── configs/                          # Declarative YAML configurations
│   ├── universe.yaml                 # 20 NSE stock tickers & macro symbols
│   ├── data.yaml                     # Start date, holdout date, horizons, seed
│   ├── costs.yaml                    # Brokerage, STT, exchange, stamp duty, slippage
│   ├── risk.yaml                     # Sizing, loss limits, drawdown circuit breakers
│   ├── model.yaml                    # Model hyper-parameters and CV settings
│   ├── news_sources.yaml             # RSS feed URLs
│   └── aliases.yaml                  # Ticker string aliases and regex mappings
│
├── data_store/                       # Local data storage (git-ignored)
│   ├── raw/                          # Raw price and macro Parquet files
│   ├── features/                     # Feature matrix and macro panel Parquet
│   ├── labels/                       # Triple barrier labels and sample weights
│   ├── oos/                          # Out-of-sample predictions
│   ├── news.db                       # SQLite news database with FTS5
│   └── chroma/                       # ChromaDB persistent vector database
│
├── prompts/                          # Version-controlled prompt templates
│   └── extract_v1.txt                # Structured financial news extraction prompt
│
├── reports/                          # Generated markdown reports and plot artifacts
│   ├── eda.md                        # Dataset summary and class balances
│   ├── feature_report.md             # Feature correlations and missingness
│   ├── week1_audit.md                # Week 1 data leakage audit report
│   ├── baselines.md                  # Baseline strategy performance benchmark
│   ├── lgbm_walkforward.md           # LightGBM fold evaluation metrics
│   └── week2_summary.md              # Week 2 net-of-cost performance summary
│
├── src/stock_engine/                 # Core Python package
│   ├── config.py                     # Pydantic configuration loader
│   ├── logging_setup.py              # Structured application logger
│   ├── data/                         # Ingestion, validation & PIT alignment
│   │   ├── prices.py                 # yfinance downloader with exponential backoff
│   │   ├── macro.py                  # Macro index data downloader
│   │   ├── validate.py               # Data quality checks & anomaly detector
│   │   └── pit.py                    # Point-in-time alignment engine
│   ├── features/                     # Vectorized feature generation
│   │   ├── technical.py              # RSI, MACD, ATR, Bollinger, rolling skew
│   │   ├── market.py                 # Market context, beta, cross-sectional ranks
│   │   └── build.py                  # Unified feature matrix generator
│   ├── labels/                       # Labeling algorithms
│   │   └── triple_barrier.py         # Volatility-scaled triple barrier labels
│   ├── cv/                           # Cross validation strategies
│   │   └── purged_walk_forward.py    # Purged walk-forward CV with embargo
│   ├── models/                       # Predictive modeling algorithms
│   │   ├── lgbm.py                   # LightGBM classifier pipeline
│   │   ├── gru.py                    # PyTorch GRU recurrent sequence model
│   │   ├── calibration.py            # Isotonic & Platt probability calibrators
│   │   └── signals.py                # Quantile threshold signal generator
│   ├── tuning/                       # Hyper-parameter optimization
│   │   └── optuna_search.py          # Nested Optuna study on inner time splits
│   ├── regime/                       # Market regime detection
│   │   └── hmm.py                    # Forward-filtered Gaussian HMM
│   ├── ensemble/                     # Model combination & ablation
│   │   ├── stack.py                  # Regularized logistic meta-learner
│   │   └── ablation.py               # Component contribution benchmark
│   ├── backtest/                     # Backtest execution & evaluation
│   │   ├── costs.py                  # Indian equity transaction cost model
│   │   ├── engine.py                 # Next-open backtest simulator
│   │   ├── metrics.py                # CAGR, Sharpe CI, Sortino, Calmar, drawdown
│   │   └── baselines.py              # Buy & hold, SMA, and momentum benchmarks
│   ├── rag/                          # News retrieval & reasoning
│   │   ├── ingest/rss.py             # RSS ingestion and deduplication
│   │   ├── link.py                   # Ticker entity linker
│   │   ├── index.py                  # ChromaDB vector indexer
│   │   ├── retrieve.py               # BM25 + Dense RRF hybrid retriever
│   │   ├── llm.py                    # Pluggable Ollama / OpenAI client
│   │   ├── extract.py                # Structured schema extractor
│   │   ├── features.py               # Time-filtered RAG feature builder
│   │   ├── explain.py                # Explain-trade generation with citations
│   │   └── eval_retrieval.py         # Retrieval evaluation metrics
│   ├── risk/                         # Capital preservation & safeguards
│   │   ├── engine.py                 # Rule-based risk decision engine
│   │   └── kill_switch.py            # Persisted SQLite kill switch
│   ├── execution/                    # Broker adapters & runner
│   │   ├── broker.py                 # Abstract BrokerAdapter interface
│   │   ├── paper.py                  # Simulated paper brokerage
│   │   ├── groww.py                  # Groww API adapter with safety checks
│   │   └── runner.py                 # Daily end-to-end execution runner
│   ├── api/                          # REST API services
│   │   ├── main.py                   # FastAPI service router
│   │   └── schemas.py                # Pydantic request/response schemas
│   ├── dashboard/                    # Interactive web UI
│   │   ├── app.py                    # Main Streamlit application
│   │   └── pages/                    # 8-page analytics dashboard
│   ├── monitoring/                   # Decay and drift detection
│   │   └── drift.py                  # PSI, KS-test, and performance decay
│   └── ops/                          # Operational scheduling
│       └── scheduler.py              # Daily Asia/Kolkata workflow automation
│
└── tests/                            # Comprehensive automated test suite
    ├── test_config.py                # Configuration loading tests
    ├── test_prices.py                # Data ingestion & validation tests
    ├── test_pit.py                   # Point-in-time alignment tests
    ├── test_technical.py             # Feature calculation tests
    ├── test_costs.py                 # Transaction cost calculation tests
    └── utils_leakage.py              # Automated truncation leakage assertions
```

---

## 10. Quickstart Guide

### Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `conda`)
- Ollama (Optional, for local LLM extraction)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Aryan-Lade/Niveshak.git
cd Niveshak

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` and set your runtime parameters:
```bash
cp .env.example .env
```
Key configuration items:
```ini
LIVE_TRADING=false
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
API_KEY=your_secret_api_key_here
```

### 3. Run Automated Tests
Verify pipeline integrity and run all leakage/cost assertions:
```bash
python -m pytest tests/
```

### 4. Execute Cost Model Verification
Verify transaction cost calculations directly:
```bash
python -m pytest tests/test_costs.py -v
```

### 5. Launch the FastAPI Backend
```bash
uvicorn src.stock_engine.api.main:app --reload --port 8000
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

### 6. Launch the Streamlit Dashboard
```bash
streamlit run src/stock_engine/dashboard/app.py
```

---

## 11. Regulatory & Compliance Notice

> [!CAUTION]  
> **Educational & Research Disclaimer**  
> Niveshak is built exclusively for educational, quantitative research, and algorithm verification purposes. It does not constitute financial advice, investment recommendations, or portfolio management services.
> 
> - **Market Volatility**: Financial markets involve substantial risk of loss. Past simulated performance or backtested metrics do not guarantee future returns.
> - **Regulatory Compliance**: Algorithmic order execution in Indian markets is governed by the Securities and Exchange Board of India (SEBI) guidelines, including mandatory Algo-ID approvals, broker certifications, and static IP allocations. Ensure full compliance with relevant statutory regulations before deploying any automated execution system.
