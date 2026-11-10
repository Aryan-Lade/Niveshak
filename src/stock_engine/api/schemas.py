"""
Pydantic Request and Response Schemas for Niveshak FastAPI Backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    mode: str = "PAPER"
    timestamp: str


class SignalItem(BaseModel):
    ticker: str
    date: str
    score: float
    calibrated_prob: float
    regime: Optional[str] = None
    rag_sentiment: Optional[float] = None
    risk_action: Optional[str] = None


class SignalsResponse(BaseModel):
    as_of: str
    count: int
    signals: List[SignalItem]


class PositionItem(BaseModel):
    ticker: str
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float


class PositionsResponse(BaseModel):
    count: int
    positions: List[PositionItem]
    total_market_value: float


class TradeItem(BaseModel):
    fill_id: str
    client_order_id: str
    ticker: str
    side: str
    quantity: int
    price: float
    cost: float
    fill_time: str


class TradesResponse(BaseModel):
    count: int
    trades: List[TradeItem]


class EquitySnapshotItem(BaseModel):
    date: str
    cash: float
    positions_value: float
    total_equity: float


class EquityResponse(BaseModel):
    count: int
    current_equity: float
    current_cash: float
    snapshots: List[EquitySnapshotItem]


class BacktestSummaryResponse(BaseModel):
    strategy_name: str
    cagr: float
    sharpe: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    max_drawdown: float
    annual_turnover: float
    win_rate: float
    profit_factor: float
    best_baseline: str
    best_baseline_sharpe: float
    beats_baseline: bool
    notes: str


class RiskStatusResponse(BaseModel):
    is_tripped: bool
    reason: Optional[str] = None
    tripped_at: Optional[str] = None
    reset_at: Optional[str] = None
    max_positions: int
    max_weight_per_stock: float
    daily_loss_limit_pct: float
    drawdown_circuit_breaker_pct: float
    min_avg_daily_traded_value_inr: float
    price_sanity_band_pct: float


class KillSwitchRequest(BaseModel):
    reason: str = Field(..., description="Mandatory reason for manual halt")


class ResetRequest(BaseModel):
    confirm: bool = Field(..., description="Explicit confirmation required to reset kill switch")


class PaperRunRequest(BaseModel):
    as_of: Optional[str] = Field(None, description="Trading date YYYY-MM-DD. Defaults to today.")


class PaperRunResponse(BaseModel):
    date: str
    status: str
    fills_count: int
    orders_queued_count: int
    total_equity: float
    cash: float


class NewsItem(BaseModel):
    doc_id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: int
    tickers: List[str] = Field(default_factory=list)


class NewsRecentResponse(BaseModel):
    ticker: Optional[str] = None
    count: int
    articles: List[NewsItem]


class ExplainRequest(BaseModel):
    trade_id: Optional[str] = None
    ticker: Optional[str] = None
    date: Optional[str] = None


class ExplainResponse(BaseModel):
    ticker: str
    date: str
    model_view: str
    evidence_from_news: str
    risks: str
    what_would_change_view: str
    cited_doc_ids: List[str]
    disclaimer: str


class RebalanceOrderItem(BaseModel):
    ticker: str
    action: str
    current_shares: int
    target_shares: int
    delta_shares: int
    estimated_price: float
    trade_value: float
    total_cost_inr: float


class RebalancePlanRequest(BaseModel):
    target_weights: Dict[str, float]
    market_prices: Optional[Dict[str, float]] = None


class RebalancePlanResponse(BaseModel):
    nav: float
    cash_available: float
    cash_after_rebalance: float
    orders: List[RebalanceOrderItem]
    total_estimated_costs_inr: float
    turnover_inr: float


class CopilotOverlayResponse(BaseModel):
    ticker: str
    timestamp: str
    direction: str
    confidence_pct: float
    current_price: float
    entry_price: float
    target_1: float
    target_2: float
    stop_loss: float
    risk_reward_ratio: float
    pattern_detected: str
    regime: str
    signals: Dict[str, Any]
    broker_compatibility: List[str]


