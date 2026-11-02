"""
Institutional Deterministic Risk Engine.

Pure functions, state dataclasses, and SQLite event logging.
Independent from machine learning models to guarantee capital preservation.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from stock_engine.risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/risk.yaml")
DEFAULT_DB_PATH = Path("data_store/trading.db")


@dataclass
class PortfolioState:
    """Current snapshot of portfolio capital and holdings."""
    total_equity: float
    cash: float
    start_of_day_equity: float
    peak_equity: float
    open_positions: Dict[str, float] = field(default_factory=dict)  # ticker -> current position value (INR)

    @property
    def daily_pnl_pct(self) -> float:
        if self.start_of_day_equity <= 0:
            return 0.0
        return (self.total_equity - self.start_of_day_equity) / self.start_of_day_equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.total_equity - self.peak_equity) / self.peak_equity


@dataclass
class OrderRequest:
    """Prospective buy or sell order submitted to the risk engine."""
    ticker: str
    side: str  # "BUY" or "SELL"
    intended_price: float
    last_close: float
    atr: float
    avg_daily_traded_value: float
    vix_zscore: Optional[float] = None
    rag_sentiment: Optional[float] = None
    rag_confidence: Optional[float] = None


@dataclass
class RiskDecision:
    """Output of risk evaluation for a given order request."""
    approved: bool
    approved_qty: int
    reasons: List[str] = field(default_factory=list)
    risk_action: str = "EXECUTE"  # "EXECUTE", "REJECT", "REVIEW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "approved_qty": self.approved_qty,
            "reasons": self.reasons,
            "risk_action": self.risk_action,
        }


# =========================================================================
# Pure Risk Rule Functions
# =========================================================================

def check_liquidity_filter(
    avg_daily_volume_value: float,
    min_required: float,
) -> Tuple[bool, Optional[str]]:
    """Verify market liquidity meets institutional minimum turnover."""
    if avg_daily_volume_value < min_required:
        return False, f"Average daily traded value INR {avg_daily_volume_value:,.0f} < required INR {min_required:,.0f}"
    return True, None


def check_price_sanity(
    intended_price: float,
    last_close: float,
    band_pct: float,
) -> Tuple[bool, Optional[str]]:
    """Reject orders if intended price gaps excessively from previous close."""
    if last_close <= 0:
        return False, "Invalid non-positive previous close price"

    pct_diff = abs(intended_price - last_close) / last_close
    if pct_diff > band_pct:
        return False, f"Price {intended_price:.2f} deviates {pct_diff:.2%} from last close {last_close:.2f} (band: {band_pct:.1%})"
    return True, None


def check_vix_regime(
    vix_zscore: Optional[float],
    max_zscore: float,
) -> Tuple[bool, Optional[str]]:
    """Optional macro volatility filter: skip entries during extreme VIX spikes."""
    if vix_zscore is not None and vix_zscore > max_zscore:
        return False, f"VIX z-score {vix_zscore:.2f} exceeds extreme panic threshold {max_zscore:.2f}"
    return True, None


def check_daily_loss_limit(
    daily_pnl_pct: float,
    limit_pct: float,
) -> Tuple[bool, Optional[str]]:
    """Daily circuit breaker: no new entries if daily drawdown exceeds threshold."""
    if daily_pnl_pct <= limit_pct:
        return False, f"Daily PnL {daily_pnl_pct:.2%} breached daily loss limit {limit_pct:.2%}"
    return True, None


def check_drawdown_circuit_breaker(
    drawdown_pct: float,
    breaker_pct: float,
) -> Tuple[bool, Optional[str]]:
    """Portfolio circuit breaker: breach trips kill switch."""
    if drawdown_pct <= breaker_pct:
        return False, f"Portfolio drawdown {drawdown_pct:.2%} breached max drawdown limit {breaker_pct:.2%}"
    return True, None


def calculate_position_size(
    equity: float,
    cash: float,
    price: float,
    atr: float,
    risk_per_trade_pct: float,
    atr_multiplier: float,
    max_weight_per_stock: float,
) -> Tuple[int, List[str]]:
    """
    Position sizing formula:
    risk_amount = equity * risk_per_trade_pct
    stop_distance = atr_multiplier * atr
    raw_qty = risk_amount / stop_distance
    Bounded by: max_weight_per_stock, available cash.
    """
    reasons: List[str] = []
    if price <= 0 or atr <= 0 or equity <= 0:
        return 0, ["Non-positive price, ATR, or equity"]

    stop_distance = max(atr_multiplier * atr, 0.01 * price)
    risk_amount = equity * risk_per_trade_pct
    raw_shares = int(risk_amount // stop_distance)

    # Max weight constraint
    max_val = equity * max_weight_per_stock
    weight_shares = int(max_val // price)

    # Cash constraint (long-only, no leverage)
    cash_shares = int(cash // price)

    approved_qty = min(raw_shares, weight_shares, cash_shares)

    if approved_qty <= 0:
        if cash_shares <= 0:
            reasons.append(f"Insufficient cash (INR {cash:,.2f}) for 1 share of INR {price:.2f}")
        elif raw_shares <= 0:
            reasons.append("Sizing calculation yielded 0 shares under ATR risk constraints")
        return 0, reasons

    reasons.append(
        f"Sized {approved_qty} shares (risk cap: {raw_shares}, weight cap: {weight_shares}, cash cap: {cash_shares})"
    )
    return approved_qty, reasons


def check_news_risk_flag(
    ticker: str,
    rag_sentiment: Optional[float],
    rag_confidence: Optional[float],
    sentiment_thresh: float,
    confidence_thresh: float,
    auto_exit: bool,
) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluate news risk for a held ticker.
    Produces REVIEW flag (or AUTO_EXIT if configured).
    """
    if rag_sentiment is None or rag_confidence is None:
        return False, "EXECUTE", None

    if rag_sentiment <= sentiment_thresh and rag_confidence >= confidence_thresh:
        action = "AUTO_EXIT" if auto_exit else "REVIEW"
        reason = (
            f"Adverse news detected on {ticker}: sentiment {rag_sentiment:.2f} <= {sentiment_thresh:.2f} "
            f"with confidence {rag_confidence:.2f} >= {confidence_thresh:.2f}"
        )
        return True, action, reason

    return False, "EXECUTE", None


# =========================================================================
# Risk Engine Class
# =========================================================================

class RiskEngine:
    """
    Deterministic risk orchestrator with YAML configurations and SQLite audit trail.
    """

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        db_path: Path = DEFAULT_DB_PATH,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)
        self.config = self._load_config()
        self.kill_switch = kill_switch or KillSwitch(db_path=self.db_path)
        self._init_db()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                return raw.get("risk", {}) if raw else {}
        return {}

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    approved_qty INTEGER NOT NULL,
                    risk_action TEXT NOT NULL,
                    reasons TEXT NOT NULL
                )
            """)
            conn.commit()

    def _log_event(
        self,
        ticker: str,
        side: str,
        decision: RiskDecision,
    ) -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO risk_events (timestamp, ticker, side, approved, approved_qty, risk_action, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    ticker,
                    side,
                    int(decision.approved),
                    decision.approved_qty,
                    decision.risk_action,
                    "; ".join(decision.reasons),
                ),
            )
            conn.commit()

    def evaluate_order(
        self,
        order: OrderRequest,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """
        Evaluate prospective trade against all deterministic risk rules.
        """
        reasons: List[str] = []

        # 1. Kill switch check
        if self.kill_switch.is_tripped():
            ks_status = self.kill_switch.status()
            reason = f"Kill switch is active: {ks_status.get('reason', 'Trading halted')}"
            dec = RiskDecision(approved=False, approved_qty=0, reasons=[reason], risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 2. Drawdown circuit breaker
        dd_limit = float(self.config.get("drawdown_circuit_breaker_pct", -0.10))
        ok_dd, dd_reason = check_drawdown_circuit_breaker(portfolio.drawdown_pct, dd_limit)
        if not ok_dd:
            self.kill_switch.trip(dd_reason or "Drawdown circuit breaker breached")
            dec = RiskDecision(approved=False, approved_qty=0, reasons=[dd_reason], risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        # Sells bypass new-entry constraints
        if order.side.upper() == "SELL":
            dec = RiskDecision(approved=True, approved_qty=0, reasons=["Sell order approved by risk engine"])
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 3. Daily loss limit check
        daily_loss_limit = float(self.config.get("daily_loss_limit_pct", -0.02))
        ok_daily, daily_reason = check_daily_loss_limit(portfolio.daily_pnl_pct, daily_loss_limit)
        if not ok_daily:
            dec = RiskDecision(approved=False, approved_qty=0, reasons=[daily_reason], risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 4. Max simultaneous open positions
        max_positions = int(self.config.get("max_positions", 5))
        if order.ticker not in portfolio.open_positions and len(portfolio.open_positions) >= max_positions:
            dec = RiskDecision(
                approved=False,
                approved_qty=0,
                reasons=[f"Max positions reached: {len(portfolio.open_positions)}/{max_positions}"],
                risk_action="REJECT",
            )
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 5. Price sanity band
        sanity_band = float(self.config.get("price_sanity_band_pct", 0.05))
        ok_price, price_reason = check_price_sanity(order.intended_price, order.last_close, sanity_band)
        if not ok_price:
            dec = RiskDecision(approved=False, approved_qty=0, reasons=[price_reason], risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 6. Liquidity filter
        min_liquidity = float(self.config.get("min_avg_daily_traded_value_inr", 50000000.0))
        ok_liq, liq_reason = check_liquidity_filter(order.avg_daily_traded_value, min_liquidity)
        if not ok_liq:
            dec = RiskDecision(approved=False, approved_qty=0, reasons=[liq_reason], risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 7. VIX regime filter
        max_vix_z = float(self.config.get("max_vix_zscore", 2.5))
        ok_vix, vix_reason = check_vix_regime(order.vix_zscore, max_vix_z)
        if not ok_vix:
            dec = RiskDecision(approved=False, approved_qty=0, reasons=[vix_reason], risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        # 8. Position sizing under ATR, weight, and cash constraints
        risk_pct = float(self.config.get("risk_per_trade_pct", 0.01))
        atr_mult = float(self.config.get("atr_multiplier", 2.0))
        max_weight = float(self.config.get("max_weight_per_stock", 0.20))

        qty, size_reasons = calculate_position_size(
            equity=portfolio.total_equity,
            cash=portfolio.cash,
            price=order.intended_price,
            atr=order.atr,
            risk_per_trade_pct=risk_pct,
            atr_multiplier=atr_mult,
            max_weight_per_stock=max_weight,
        )

        reasons.extend(size_reasons)
        if qty <= 0:
            dec = RiskDecision(approved=False, approved_qty=0, reasons=reasons, risk_action="REJECT")
            self._log_event(order.ticker, order.side, dec)
            return dec

        dec = RiskDecision(approved=True, approved_qty=qty, reasons=reasons, risk_action="EXECUTE")
        self._log_event(order.ticker, order.side, dec)
        return dec

    def review_held_positions(
        self,
        ticker: str,
        rag_sentiment: Optional[float],
        rag_confidence: Optional[float],
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check for severe news risks on already held positions.
        """
        sent_thresh = float(self.config.get("news_risk_sentiment_threshold", -0.70))
        conf_thresh = float(self.config.get("news_risk_confidence_threshold", 0.70))
        auto_exit = bool(self.config.get("auto_exit_on_news_risk", False))

        return check_news_risk_flag(
            ticker=ticker,
            rag_sentiment=rag_sentiment,
            rag_confidence=rag_confidence,
            sentiment_thresh=sent_thresh,
            confidence_thresh=conf_thresh,
            auto_exit=auto_exit,
        )

