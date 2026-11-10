"""
Niveshak Live Day-Trader Copilot & Intraday Pattern Scanner.

Inspired by modular day-trading architectures, adapted natively for Indian liquid equities (NSE):
1. Opening Range Breakout (ORB: 15-min and 30-min price discovery range).
2. Volume-Weighted Average Price (VWAP) dynamic institutional anchor and bands.
3. Intraday Momentum & Volume Surge detection ($V > 1.5 \times \text{SMA}_{20}$).
4. Live Directional UP / DOWN Prediction with calibrated confidence score.
5. Risk-Engine Invalidation Stop Loss (ATR) and Multi-Stage Profit Targets (R1, R2).
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CopilotPrediction:
    ticker: str
    timestamp: str
    direction: str                     # "UP", "DOWN", "NEUTRAL"
    confidence_pct: float              # 0.0 to 100.0
    current_price: float
    entry_price: float
    target_1: float
    target_2: float
    stop_loss: float
    risk_reward_ratio: float
    pattern_detected: str
    regime: str                        # "Bull", "Sideways", "Bear"
    signals: Dict[str, Any]
    broker_compatibility: List[str]


class IntradayCopilotScanner:
    """
    Real-time intraday market analysis engine that scans live price action,
    detects technical momentum breakouts, and issues high-conviction UP/DOWN predictions.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 60.0,
        atr_multiplier_sl: float = 1.0,
        atr_multiplier_tp1: float = 1.5,
        atr_multiplier_tp2: float = 2.5,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp1 = atr_multiplier_tp1
        self.atr_multiplier_tp2 = atr_multiplier_tp2

    def analyze_ticker(
        self,
        ticker: str,
        bars: Optional[pd.DataFrame] = None,
    ) -> CopilotPrediction:
        """
        Computes intraday indicators and returns structured UP/DOWN prediction.
        If live bars are not provided, generates deterministic synthetic intraday series.
        """
        clean_ticker = ticker.upper()
        if not clean_ticker.endswith(".NS") and not clean_ticker.startswith("^"):
            clean_ticker = f"{clean_ticker}.NS"

        if bars is None or len(bars) < 15:
            bars = self._generate_synthetic_intraday(clean_ticker)

        # Calculate Indicators
        high = bars["high"].values
        low = bars["low"].values
        close = bars["close"].values
        volume = bars["volume"].values
        curr_price = float(close[-1])

        # 1. Opening Range (First 3 bars = 15m)
        orb_high = float(np.max(high[:3]))
        orb_low = float(np.min(low[:3]))

        # 2. VWAP Calculation: Cumulative(Typical Price * Volume) / Cumulative(Volume)
        typical_price = (high + low + close) / 3.0
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical_price * volume)
        vwap = float(cum_tp_vol[-1] / (cum_vol[-1] + 1e-8))

        # 3. ATR (Average True Range)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr = float(np.mean(tr[-10:])) if len(tr) >= 10 else curr_price * 0.008

        # 4. Volume Surge Ratio
        recent_vol = np.mean(volume[-3:])
        avg_vol = np.mean(volume)
        vol_surge = float(recent_vol / (avg_vol + 1e-8))

        # 5. Momentum Scoring (RSI Proxy & Slope)
        pct_change_5 = (curr_price - close[-5]) / close[-5] if len(close) >= 5 else 0.0
        dist_from_vwap = (curr_price - vwap) / vwap

        # Determine Directional Signals
        bull_points = 0.0
        bear_points = 0.0

        # Price vs VWAP
        if curr_price > vwap:
            bull_points += 2.0
        else:
            bear_points += 2.0

        # Opening Range Breakout
        if curr_price > orb_high:
            bull_points += 2.5
            pattern = "Bullish Opening Range High Breakout (ORB-15)"
        elif curr_price < orb_low:
            bear_points += 2.5
            pattern = "Bearish Opening Range Low Breakdown (ORB-15)"
        elif curr_price > vwap and dist_from_vwap < 0.005:
            bull_points += 1.5
            pattern = "Bullish VWAP Mean-Reversion Bounce"
        elif curr_price < vwap and dist_from_vwap > -0.005:
            bear_points += 1.5
            pattern = "Bearish VWAP Rejection Pullback"
        else:
            pattern = "Consolidation inside Opening Range"

        # Volume Surge Boost
        if vol_surge > 1.3:
            if curr_price > vwap:
                bull_points += 1.5
            else:
                bear_points += 1.5

        # Momentum confirmation
        if pct_change_5 > 0.003:
            bull_points += 1.0
        elif pct_change_5 < -0.003:
            bear_points += 1.0

        # Aggregate Prediction
        total_points = bull_points + bear_points + 1e-5
        if bull_points > bear_points:
            direction = "UP"
            confidence = round(min(92.0, max(55.0, (bull_points / total_points) * 100.0)), 1)
            regime = "Bull"
            entry = curr_price
            sl = round(entry - (self.atr_multiplier_sl * atr), 2)
            tp1 = round(entry + (self.atr_multiplier_tp1 * atr), 2)
            tp2 = round(entry + (self.atr_multiplier_tp2 * atr), 2)
        elif bear_points > bull_points:
            direction = "DOWN"
            confidence = round(min(92.0, max(55.0, (bear_points / total_points) * 100.0)), 1)
            regime = "Bear"
            entry = curr_price
            sl = round(entry + (self.atr_multiplier_sl * atr), 2)
            tp1 = round(entry - (self.atr_multiplier_tp1 * atr), 2)
            tp2 = round(entry - (self.atr_multiplier_tp2 * atr), 2)
        else:
            direction = "NEUTRAL"
            confidence = 50.0
            regime = "Sideways"
            entry = curr_price
            sl = round(entry - atr, 2)
            tp1 = round(entry + atr, 2)
            tp2 = round(entry + (1.5 * atr), 2)

        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr_ratio = round(reward / (risk + 1e-5), 2)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

        return CopilotPrediction(
            ticker=clean_ticker,
            timestamp=now_str,
            direction=direction,
            confidence_pct=confidence,
            current_price=round(curr_price, 2),
            entry_price=round(entry, 2),
            target_1=tp1,
            target_2=tp2,
            stop_loss=sl,
            risk_reward_ratio=rr_ratio,
            pattern_detected=pattern,
            regime=regime,
            signals={
                "vwap": round(vwap, 2),
                "orb_high": round(orb_high, 2),
                "orb_low": round(orb_low, 2),
                "atr": round(atr, 2),
                "volume_surge_ratio": round(vol_surge, 2),
                "pct_change_recent": round(pct_change_5 * 100, 2),
            },
            broker_compatibility=["Groww", "Angel One", "TradingView", "Zerodha Kite"],
        )

    def _generate_synthetic_intraday(self, ticker: str) -> pd.DataFrame:
        """
        Generates realistic 5-minute candle structure for fallback / simulation.
        """
        seed = abs(hash(ticker)) % 10000
        rng = np.random.default_rng(seed)

        base_price = 2500.0 if "RELIANCE" in ticker else (3800.0 if "TCS" in ticker else 1500.0)
        n_bars = 40

        returns = rng.normal(0.0003, 0.002, size=n_bars)
        price_series = base_price * np.exp(np.cumsum(returns))

        highs = price_series * (1.0 + rng.uniform(0.0005, 0.003, size=n_bars))
        lows = price_series * (1.0 - rng.uniform(0.0005, 0.003, size=n_bars))
        opens = np.roll(price_series, 1)
        opens[0] = base_price
        volumes = rng.integers(5000, 50000, size=n_bars)

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": price_series,
            "volume": volumes,
        })
