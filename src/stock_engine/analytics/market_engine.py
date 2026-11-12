"""
Market snapshot engine for the Niveshak terminal.

What this module does
---------------------
It turns one trading session of per-minute data for the NSE F&O universe into
everything the web terminal shows: breadth, sector moves, open-interest
build-up classes, a ranked list of setups and price levels for each stock.

The analytics are plain, documented formulas (see ``snapshot`` and
``_levels``). They describe what price, volume and open interest did; they do
not predict anything and are not investment advice.

Where the data comes from
-------------------------
``simulate_day`` generates a deterministic *simulated* session (one fixed
random seed per calendar date) so the terminal works without a paid feed.
Every payload carries ``meta.data_mode == "simulated"`` and the UI shows it.
To go live, replace ``simulate_day`` with a function that returns the same
``SessionData`` arrays from a broker / vendor feed - nothing else changes.
"""

from __future__ import annotations

import datetime as dt
import math
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from stock_engine.analytics.universe import BY_SYMBOL, UNIVERSE

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), name="IST")
SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
SESSION_MINUTES = 375
BAR_MINUTES = 5
BARS_PER_SESSION = SESSION_MINUTES // BAR_MINUTES

# A move smaller than this (in %) counts as unchanged for breadth.
FLAT_BAND_PCT = 0.05
# Stocks below these thresholds are tagged "choppy" and left out of the boards.
CHOPPY_EFFICIENCY = 0.08
CHOPPY_MOVE_PCT = 0.20
# Rankings need a little session data before they mean anything.
SETTLE_MINUTES = 15

SCORE_WEIGHTS = {"momentum": 0.40, "trend": 0.25, "open_interest": 0.20, "volume": 0.15}

BUILDUP_LABELS = {
    "long_buildup": "Long build-up",
    "short_covering": "Short covering",
    "short_buildup": "Short build-up",
    "long_unwinding": "Long unwinding",
    "flat": "Flat",
}

HISTOGRAM_EDGES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
HISTOGRAM_LABELS = [
    "below -3%", "-3 to -2%", "-2 to -1%", "-1 to -0.5%", "-0.5 to 0%",
    "0 to 0.5%", "0.5 to 1%", "1 to 2%", "2 to 3%", "above 3%",
]


# --------------------------------------------------------------------------
# Session clock
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionClock:
    """Which session to show and how many minutes of it exist right now."""

    session_date: dt.date
    state: str          # "open" | "closed" | "pre_open"
    live_minute: int    # 1..375 minutes of the session available


def resolve_session(now: Optional[dt.datetime] = None) -> SessionClock:
    """
    Map a wall-clock time to the session the terminal should display.

    NSE cash/F&O hours are 09:15-15:30 IST, Monday to Friday. Outside those
    hours the previous completed session is shown in full. Exchange holidays
    are not modelled here; plug in the holiday calendar with the live feed.
    """
    now = (now or dt.datetime.now(tz=IST)).astimezone(IST)
    today = now.date()
    weekday = today.weekday() < 5
    clock = now.time()

    if weekday and SESSION_OPEN <= clock < SESSION_CLOSE:
        elapsed = (now.hour * 60 + now.minute) - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
        return SessionClock(today, "open", max(1, min(SESSION_MINUTES, elapsed)))
    if weekday and clock >= SESSION_CLOSE:
        return SessionClock(today, "closed", SESSION_MINUTES)

    previous = today - dt.timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= dt.timedelta(days=1)
    return SessionClock(previous, "pre_open" if weekday else "closed", SESSION_MINUTES)


def minute_to_label(minute: int) -> str:
    """Session minute (0 = 09:15) to an 'HH:MM' IST label."""
    total = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute + int(minute)
    return f"{total // 60:02d}:{total % 60:02d}"


def label_to_minute(label: str) -> int:
    """'HH:MM' IST to session minute. Raises ValueError on anything else."""
    try:
        hours, minutes = label.strip().split(":")
        total = int(hours) * 60 + int(minutes)
    except Exception as exc:  # noqa: BLE001 - any parse problem is the same error to the caller
        raise ValueError(f"Expected a time like '11:30', got {label!r}") from exc
    if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
        raise ValueError(f"Expected a time like '11:30', got {label!r}")
    return total - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)


# --------------------------------------------------------------------------
# Session data (simulated feed)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionData:
    """Per-minute arrays for one session. Shape (n_stocks, 375) unless noted."""

    session_date: dt.date
    prev_close: np.ndarray        # (n,)
    prev_high: np.ndarray         # (n,)
    prev_low: np.ndarray          # (n,)
    daily_range_pct: np.ndarray   # (n,) the stock's typical daily move, %
    avg_turnover_cr: np.ndarray   # (n,) usual full-day turnover, Rs crore
    open_pct: np.ndarray          # (n,) opening gap vs previous close, %
    pct: np.ndarray               # % change vs previous close at each minute close
    oi_pct: np.ndarray            # % change in futures open interest since previous close
    minute_rel_volume: np.ndarray  # minute volume / the stock's usual average minute volume
    volume_ratio: np.ndarray      # cumulative volume / usual cumulative volume by that time
    wick_up_pct: np.ndarray       # (n, 75) extra high per 5-minute bar, %
    wick_down_pct: np.ndarray     # (n, 75) extra low per 5-minute bar, %


def _static_traits() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traits that belong to the stock, not the day: range, beta, turnover."""
    rng = np.random.default_rng(1729)
    n = len(UNIVERSE)
    daily_range = rng.uniform(1.1, 2.6, n)
    beta = rng.uniform(0.7, 1.35, n)
    avg_turnover = np.exp(rng.normal(math.log(350.0), 0.8, n))
    return daily_range, beta, avg_turnover


@lru_cache(maxsize=6)
def simulate_day(session_date: dt.date) -> SessionData:
    """
    Deterministic simulated session for ``session_date``.

    Each stock's minute return is ``beta x market + sector + own``, where the
    stock's own drift is concentrated in a window of the day (so there is a
    visible impulse for order-block detection). Open interest drifts with the
    size of the move and is positive about 70% of the time, which produces a
    realistic mix of the four build-up classes.
    """
    n = len(UNIVERSE)
    t_count = SESSION_MINUTES
    rng = np.random.default_rng(session_date.toordinal() * 7919 + 20240101)
    daily_range, beta, avg_turnover = _static_traits()

    sector_names = sorted({i.sector for i in UNIVERSE})
    sector_of = np.array([sector_names.index(i.sector) for i in UNIVERSE])

    # --- the day's story ---------------------------------------------------
    market_drift = rng.normal(0.0, 0.55)
    sector_drift = rng.normal(0.0, 0.75, len(sector_names))
    own_drift = rng.standard_t(4, n) * 0.55 * (daily_range / 1.8)
    movers = rng.choice(n, size=max(6, n // 14), replace=False)
    own_drift[movers] += rng.choice([-1.0, 1.0], movers.size) * rng.uniform(1.5, 4.0, movers.size)
    gap = 0.35 * market_drift + rng.normal(0.0, 0.35, n)

    minutes = np.arange(t_count)
    centre = rng.uniform(20, 300, n)[:, None]
    width = rng.uniform(40, 160, n)[:, None]
    shape = np.exp(-(((minutes[None, :] - centre) / width) ** 2)) + 0.3
    shape /= shape.sum(axis=1, keepdims=True)

    # --- minute returns ----------------------------------------------------
    market_r = market_drift / t_count + rng.normal(0.0, 0.55 / math.sqrt(t_count), t_count)
    sector_r = sector_drift[:, None] / t_count + rng.normal(
        0.0, 0.45 / math.sqrt(t_count), (len(sector_names), t_count)
    )
    own_sd = (daily_range * 0.75 / math.sqrt(t_count))[:, None]
    own_r = own_drift[:, None] * shape + rng.normal(0.0, 1.0, (n, t_count)) * own_sd
    returns = beta[:, None] * market_r[None, :] + sector_r[sector_of] + own_r
    pct = gap[:, None] + np.cumsum(returns, axis=1)

    # --- reference prices --------------------------------------------------
    ref = np.array([i.ref_price for i in UNIVERSE])
    phase = np.array([(zlib.crc32(i.symbol.encode()) % 628) / 100.0 for i in UNIVERSE])
    prev_close = ref * np.exp(0.05 * np.sin(session_date.toordinal() / 14.0 + phase))
    prev_span = daily_range * rng.uniform(0.8, 1.5, n)
    upper_share = rng.uniform(0.2, 0.8, n)
    prev_high = prev_close * (1 + prev_span * upper_share / 100.0)
    prev_low = prev_close * (1 - prev_span * (1 - upper_share) / 100.0)

    # --- open interest -----------------------------------------------------
    trend_strength = (own_drift + sector_drift[sector_of] + beta * market_drift) / daily_range
    fresh_positions = np.where(rng.random(n) < 0.70, 1.0, -1.0)
    oi_drift = fresh_positions * (1.2 + 2.2 * np.abs(trend_strength)) * rng.uniform(0.5, 1.4, n)
    oi_drift += rng.normal(0.0, 0.6, n)
    oi_r = oi_drift[:, None] * shape + rng.normal(0.0, 1.0, (n, t_count)) * (0.9 / math.sqrt(t_count))
    oi_pct = np.cumsum(oi_r, axis=1)

    # --- volume ------------------------------------------------------------
    profile = 1 + 1.9 * np.exp(-minutes / 22.0) + 1.3 * np.exp(-(t_count - 1 - minutes) / 28.0)
    profile /= profile.mean()
    day_rel = np.exp(rng.normal(0.0, 0.28, n)) * (1 + 0.35 * np.abs(trend_strength))
    minute_rel = day_rel[:, None] * profile[None, :] * np.exp(rng.normal(0.0, 0.35, (n, t_count)))
    volume_ratio = np.cumsum(minute_rel, axis=1) / np.cumsum(profile)[None, :]

    bar_sd = own_sd * math.sqrt(BAR_MINUTES)
    wick_up = np.abs(rng.normal(0.0, 1.0, (n, BARS_PER_SESSION))) * bar_sd * 0.45
    wick_down = np.abs(rng.normal(0.0, 1.0, (n, BARS_PER_SESSION))) * bar_sd * 0.45

    return SessionData(
        session_date=session_date,
        prev_close=prev_close,
        prev_high=prev_high,
        prev_low=prev_low,
        daily_range_pct=daily_range,
        avg_turnover_cr=avg_turnover,
        open_pct=gap,
        pct=pct,
        oi_pct=oi_pct,
        minute_rel_volume=minute_rel,
        volume_ratio=volume_ratio,
        wick_up_pct=wick_up,
        wick_down_pct=wick_down,
    )


# --------------------------------------------------------------------------
# Pure analytics helpers (unit-tested on their own)
# --------------------------------------------------------------------------

def classify_buildup(price_change_pct: float, oi_change_pct: float) -> str:
    """
    The standard price x open-interest read used across Indian F&O desks.

        price up,   OI up   -> long build-up   (new longs being added)
        price up,   OI down -> short covering  (shorts closing)
        price down, OI up   -> short build-up  (new shorts being added)
        price down, OI down -> long unwinding  (longs closing)
    """
    if abs(price_change_pct) < FLAT_BAND_PCT:
        return "flat"
    if price_change_pct > 0:
        return "long_buildup" if oi_change_pct >= 0 else "short_covering"
    return "short_buildup" if oi_change_pct >= 0 else "long_unwinding"


def efficiency_ratio(path: np.ndarray) -> np.ndarray:
    """
    Kaufman efficiency ratio along the last axis: |net move| / sum of |steps|.
    1.0 is a straight line, values near 0 are back-and-forth chop.
    """
    path = np.asarray(path, dtype=float)
    steps = np.abs(np.diff(path, axis=-1)).sum(axis=-1)
    net = np.abs(path[..., -1] - path[..., 0])
    return np.where(steps > 1e-9, net / np.maximum(steps, 1e-9), 0.0)


def pivot_levels(high: float, low: float, close: float) -> Dict[str, float]:
    """Classic floor pivots from the previous session's high, low and close."""
    pivot = (high + low + close) / 3.0
    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "s1": 2 * pivot - high,
        "r2": pivot + (high - low),
        "s2": pivot - (high - low),
    }


def find_order_block(candles: List[Dict[str, Any]], direction: str) -> Optional[Dict[str, Any]]:
    """
    Locate the order block for ``direction`` ("bullish" or "bearish").

    Definition used here: take the strongest three-bar displacement in the
    given direction, then walk back (up to six bars) to the last candle that
    closed the other way. That candle's high-low range is the zone - the last
    place the other side was in control before the move began.
    """
    if len(candles) < 5:
        return None
    sign = 1.0 if direction == "bullish" else -1.0
    opens = np.array([c["open"] for c in candles])
    closes = np.array([c["close"] for c in candles])

    best_start, best_move = None, 0.0
    for start in range(1, len(candles) - 2):
        move = sign * (closes[start + 2] - opens[start])
        if move > best_move:
            best_start, best_move = start, move
    if best_start is None:
        return None

    origin = best_start - 1
    for j in range(best_start - 1, max(-1, best_start - 7), -1):
        if sign * (closes[j] - opens[j]) < 0:
            origin = j
            break
    bar = candles[origin]
    return {
        "low": bar["low"],
        "high": bar["high"],
        "formed_at": bar["time"],
        "impulse_from": candles[best_start]["time"],
        "impulse_pct": round(float(best_move / opens[best_start] * 100.0), 2),
    }


def tick_round(price: float) -> float:
    """Round to a 5-paise tick (1 paisa below Rs 15)."""
    tick = 0.01 if price < 15 else 0.05
    return round(round(price / tick) * tick, 2)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class MarketEngine:
    """Builds terminal payloads for a session, live or replayed to an earlier minute."""

    def __init__(
        self,
        clock: Optional[Callable[[], dt.datetime]] = None,
        feed: Callable[[dt.date], SessionData] = simulate_day,
        data_mode: str = "simulated",
    ) -> None:
        self._clock = clock or (lambda: dt.datetime.now(tz=IST))
        self._feed = feed
        self._data_mode = data_mode
        self._sector_names = sorted({i.sector for i in UNIVERSE})
        self._sector_of = np.array([self._sector_names.index(i.sector) for i in UNIVERSE])

    # -- time ---------------------------------------------------------------

    def _resolve(self, at: Optional[str]) -> Tuple[SessionClock, int]:
        session = resolve_session(self._clock())
        minute = session.live_minute
        if at:
            minute = max(1, min(session.live_minute, label_to_minute(at)))
        return session, minute

    @staticmethod
    def _bar_end_index(minute: int) -> np.ndarray:
        bars = math.ceil(minute / BAR_MINUTES)
        return np.minimum(np.arange(1, bars + 1) * BAR_MINUTES, minute) - 1

    # -- per-stock series -----------------------------------------------------

    def _candles(self, day: SessionData, i: int, minute: int) -> List[Dict[str, Any]]:
        prices = day.prev_close[i] * (1 + day.pct[i, :minute] / 100.0)
        open_price = day.prev_close[i] * (1 + day.open_pct[i] / 100.0)
        avg_minute_shares = day.avg_turnover_cr[i] * 1e7 / day.prev_close[i] / SESSION_MINUTES
        candles = []
        for k in range(math.ceil(minute / BAR_MINUTES)):
            lo, hi = k * BAR_MINUTES, min((k + 1) * BAR_MINUTES, minute)
            segment = prices[lo:hi]
            bar_open = open_price if k == 0 else prices[lo - 1]
            bar_high = max(bar_open, segment.max()) * (1 + day.wick_up_pct[i, k] / 100.0)
            bar_low = min(bar_open, segment.min()) * (1 - day.wick_down_pct[i, k] / 100.0)
            candles.append({
                "time": minute_to_label(lo),
                "open": tick_round(bar_open),
                "high": tick_round(bar_high),
                "low": tick_round(bar_low),
                "close": tick_round(segment[-1]),
                "volume": int(day.minute_rel_volume[i, lo:hi].sum() * avg_minute_shares),
            })
        return candles

    def _daily_history(self, day: SessionData, i: int, sessions: int = 60) -> List[Dict[str, Any]]:
        """Simulated daily bars ending at the previous session (whose H/L/C feed the pivots)."""
        rng = np.random.default_rng(zlib.crc32(UNIVERSE[i].symbol.encode()) + day.session_date.toordinal())
        span = day.daily_range_pct[i] / 100.0
        closes = [day.prev_close[i]]
        for _ in range(sessions - 1):
            closes.append(closes[-1] / (1 + rng.normal(0.0004, span * 0.55)))
        closes.reverse()

        bars: List[Dict[str, Any]] = []
        date = day.session_date
        dates: List[dt.date] = []
        while len(dates) < sessions:
            date -= dt.timedelta(days=1)
            if date.weekday() < 5:
                dates.append(date)
        dates.reverse()

        previous = closes[0] * (1 - rng.normal(0, span * 0.4))
        for k, close in enumerate(closes):
            bar_open = previous * (1 + rng.normal(0, span * 0.2))
            high = max(bar_open, close) * (1 + abs(rng.normal(0, span * 0.35)))
            low = min(bar_open, close) * (1 - abs(rng.normal(0, span * 0.35)))
            if k == len(closes) - 1:
                high, low = day.prev_high[i], day.prev_low[i]
                bar_open = min(max(bar_open, low), high)
            bars.append({
                "time": dates[k].isoformat(),
                "open": tick_round(bar_open),
                "high": tick_round(max(high, bar_open, close)),
                "low": tick_round(min(low, bar_open, close)),
                "close": tick_round(close),
                "volume": int(day.avg_turnover_cr[i] * 1e7 / close * math.exp(rng.normal(0, 0.3))),
            })
            previous = close
        return bars

    # -- levels -------------------------------------------------------------

    def _levels(self, day: SessionData, i: int, minute: int, bias: str, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reference levels and an illustrative plan for one stock.

        Pivots come from the previous session. The zone is the order block for
        the stock's current bias. The plan assumes a pullback entry at the
        near edge of the zone, a stop just beyond the far edge (0.1 x the
        stock's usual daily range as buffer) and the next pivot as target.
        """
        ltp = candles[-1]["close"]
        pivots = pivot_levels(day.prev_high[i], day.prev_low[i], day.prev_close[i])
        usual_range = day.daily_range_pct[i] / 100.0 * day.prev_close[i]
        if bias == "choppy":
            direction = "bullish" if ltp >= day.prev_close[i] else "bearish"
        else:
            direction = bias
        block = find_order_block(candles, direction)

        out: Dict[str, Any] = {k: tick_round(v) for k, v in pivots.items()}
        out["usual_daily_range"] = round(float(usual_range), 2)
        out["zone"] = None
        out["plan"] = None
        if block is None:
            return out

        zone_low, zone_high = block["low"], block["high"]
        if direction == "bullish":
            state = "broken" if ltp < zone_low else ("retesting" if ltp <= zone_high else "holding")
            entry, stop = zone_high, zone_low - 0.1 * usual_range
            ahead = sorted(p for p in (pivots["r1"], pivots["r2"]) if p > max(ltp, entry))
            targets = ahead or [max(ltp, entry) + 0.5 * usual_range, max(ltp, entry) + usual_range]
        else:
            state = "broken" if ltp > zone_high else ("retesting" if ltp >= zone_low else "holding")
            entry, stop = zone_low, zone_high + 0.1 * usual_range
            ahead = sorted((p for p in (pivots["s1"], pivots["s2"]) if p < min(ltp, entry)), reverse=True)
            targets = ahead or [min(ltp, entry) - 0.5 * usual_range, min(ltp, entry) - usual_range]

        out["zone"] = {
            "low": tick_round(zone_low),
            "high": tick_round(zone_high),
            "side": "demand" if direction == "bullish" else "supply",
            "state": state,
            "formed_at": block["formed_at"],
            "impulse_from": block["impulse_from"],
            "impulse_pct": block["impulse_pct"],
        }
        risk = abs(entry - stop)
        reward = abs(targets[0] - entry)
        if state != "broken" and risk > 0:
            out["plan"] = {
                "direction": "long" if direction == "bullish" else "short",
                "entry": tick_round(entry),
                "stop": tick_round(stop),
                "targets": [tick_round(t) for t in targets[:2]],
                "risk_per_share": round(float(risk), 2),
                "reward_to_risk": round(float(reward / risk), 2),
            }
        return out

    # -- snapshot -------------------------------------------------------------

    def _measure(self, day: SessionData, minute: int) -> Dict[str, np.ndarray]:
        """Vectorised per-stock measures at ``minute``."""
        bar_idx = self._bar_end_index(minute)
        closes = day.pct[:, bar_idx]
        path = np.concatenate([day.open_pct[:, None], closes], axis=1)
        change = day.pct[:, minute - 1]
        oi = day.oi_pct[:, minute - 1]
        vol_ratio = day.volume_ratio[:, minute - 1]
        er = efficiency_ratio(path)

        momentum = np.clip(np.abs(change) / (1.2 * day.daily_range_pct), 0, 1)
        trend = np.clip(er / 0.35, 0, 1)
        positions = np.clip(0.5 + 0.5 * oi / 4.0, 0, 1)
        volume = np.clip((vol_ratio - 0.7) / 1.0, 0, 1)
        score = 100 * (
            SCORE_WEIGHTS["momentum"] * momentum
            + SCORE_WEIGHTS["trend"] * trend
            + SCORE_WEIGHTS["open_interest"] * positions
            + SCORE_WEIGHTS["volume"] * volume
        )
        choppy = (er < CHOPPY_EFFICIENCY) | (np.abs(change) < CHOPPY_MOVE_PCT)
        score = np.where(choppy, 0.0, score)
        return {
            "path": path, "change": change, "oi": oi, "vol_ratio": vol_ratio, "er": er,
            "momentum": momentum, "trend": trend, "positions": positions, "volume": volume,
            "score": score, "choppy": choppy,
        }

    def _stock_row(self, day: SessionData, i: int, minute: int, m: Dict[str, np.ndarray]) -> Dict[str, Any]:
        inst = UNIVERSE[i]
        change = float(m["change"][i])
        bias = "choppy" if m["choppy"][i] else ("bullish" if change > 0 else "bearish")
        buildup = classify_buildup(change, float(m["oi"][i]))
        return {
            "symbol": inst.symbol,
            "name": inst.name,
            "sector": inst.sector,
            "ltp": tick_round(float(day.prev_close[i] * (1 + change / 100.0))),
            "prev_close": tick_round(float(day.prev_close[i])),
            "change_pct": round(change, 2),
            "oi_change_pct": round(float(m["oi"][i]), 2),
            "buildup": buildup,
            "buildup_label": BUILDUP_LABELS[buildup],
            "volume_ratio": round(float(m["vol_ratio"][i]), 2),
            "trend_quality": round(float(m["er"][i]), 2),
            "score": int(round(float(m["score"][i]))),
            "bias": bias,
            "turnover_cr": round(float(day.avg_turnover_cr[i] * day.minute_rel_volume[i, :minute].sum() / SESSION_MINUTES), 1),
            "spark": [round(float(v), 2) for v in m["path"][i]],
        }

    def snapshot(self, at: Optional[str] = None, board_size: int = 10) -> Dict[str, Any]:
        """Everything the Home and Flow screens need, as of ``at`` (default: now)."""
        session, minute = self._resolve(at)
        day = self._feed(session.session_date)
        m = self._measure(day, minute)
        n = len(UNIVERSE)

        stocks = [self._stock_row(day, i, minute, m) for i in range(n)]
        change = m["change"]
        advancing = int((change > FLAT_BAND_PCT).sum())
        declining = int((change < -FLAT_BAND_PCT).sum())

        # Histogram of moves
        bins = np.digitize(change, HISTOGRAM_EDGES)
        histogram = [
            {"label": HISTOGRAM_LABELS[b], "count": int((bins == b).sum()), "side": "down" if b < 5 else "up"}
            for b in range(len(HISTOGRAM_LABELS))
        ]

        # Build-up mix
        buildup_counts: Dict[str, int] = {k: 0 for k in BUILDUP_LABELS}
        for row in stocks:
            buildup_counts[row["buildup"]] += 1

        # Sectors (equal-weighted across members)
        sectors = []
        for s, name in enumerate(self._sector_names):
            members = np.where(self._sector_of == s)[0]
            sector_change = change[members]
            order = members[np.argsort(sector_change)]
            mix = {k: 0 for k in BUILDUP_LABELS}
            for i in members:
                mix[stocks[i]["buildup"]] += 1
            sectors.append({
                "name": name,
                "change_pct": round(float(sector_change.mean()), 2),
                "advancing": int((sector_change > FLAT_BAND_PCT).sum()),
                "declining": int((sector_change < -FLAT_BAND_PCT).sum()),
                "count": int(members.size),
                "turnover_cr": round(float(sum(stocks[i]["turnover_cr"] for i in members)), 1),
                "leader": UNIVERSE[int(order[-1])].symbol,
                "laggard": UNIVERSE[int(order[0])].symbol,
                "buildup_mix": mix,
                "path": [round(float(v), 2) for v in m["path"][members].mean(axis=0)],
            })
        sectors.sort(key=lambda s: s["change_pct"], reverse=True)

        # Boards
        ranked = np.argsort(-m["score"])
        boards: Dict[str, List[Dict[str, Any]]] = {"bullish": [], "bearish": []}
        for i in ranked:
            bias = stocks[i]["bias"]
            if bias == "choppy" or len(boards[bias]) >= board_size:
                continue
            candles = self._candles(day, int(i), minute)
            row = dict(stocks[i])
            row["rank"] = len(boards[bias]) + 1
            row["levels"] = self._levels(day, int(i), minute, bias, candles)
            boards[bias].append(row)

        # Whole-session timeline for the ribbon (to the live minute, not the replay minute)
        live_idx = self._bar_end_index(session.live_minute)
        live_closes = day.pct[:, live_idx]
        timeline = {
            "times": [minute_to_label(int(k) + 1) for k in live_idx],
            "advancing": [int(v) for v in (live_closes > FLAT_BAND_PCT).sum(axis=0)],
            "declining": [int(v) for v in (live_closes < -FLAT_BAND_PCT).sum(axis=0)],
            "average_pct": [round(float(v), 2) for v in live_closes.mean(axis=0)],
        }

        share = advancing / max(1, advancing + declining)
        mood = "Broadly higher" if share >= 0.62 else ("Broadly lower" if share <= 0.38 else "Mixed")

        return {
            "meta": {
                "session_date": session.session_date.isoformat(),
                "market_state": session.state,
                "as_of": minute_to_label(minute),
                "minute": minute,
                "live_minute": session.live_minute,
                "session_minutes": SESSION_MINUTES,
                "bar_minutes": BAR_MINUTES,
                "is_replay": minute < session.live_minute,
                "settling": minute < SETTLE_MINUTES,
                "data_mode": self._data_mode,
                "universe": n,
                "timezone": "Asia/Kolkata",
                "generated_at": self._clock().astimezone(IST).isoformat(timespec="seconds"),
            },
            "breadth": {
                "advancing": advancing,
                "declining": declining,
                "unchanged": n - advancing - declining,
                "average_change_pct": round(float(change.mean()), 2),
                "median_change_pct": round(float(np.median(change)), 2),
                "mood": mood,
                "choppy": int(m["choppy"].sum()),
                "histogram": histogram,
                "buildup": buildup_counts,
            },
            "timeline": timeline,
            "sectors": sectors,
            "stocks": stocks,
            "boards": boards,
            "method": {
                "score_weights": SCORE_WEIGHTS,
                "choppy_rule": f"trend quality below {CHOPPY_EFFICIENCY} or move under {CHOPPY_MOVE_PCT}%",
                "sector_change": "equal-weighted average of member stocks",
            },
        }

    # -- single stock ---------------------------------------------------------

    def stock(self, symbol: str, at: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Full detail for one stock, or None if the symbol is not in the universe."""
        key = symbol.strip().upper()
        if key not in BY_SYMBOL:
            return None
        i = next(k for k, inst in enumerate(UNIVERSE) if inst.symbol == key)
        session, minute = self._resolve(at)
        day = self._feed(session.session_date)
        m = self._measure(day, minute)
        row = self._stock_row(day, i, minute, m)
        candles = self._candles(day, i, minute)
        levels = self._levels(day, i, minute, row["bias"], candles)

        members = np.where(self._sector_of == self._sector_of[i])[0]
        sector_changes = sorted(
            ((float(m["change"][self._sector_of == s].mean()), name) for s, name in enumerate(self._sector_names)),
            reverse=True,
        )
        sector_rank = next(k for k, (_, name) in enumerate(sector_changes, start=1) if name == row["sector"])
        sector_change = float(m["change"][members].mean())

        # Today's developing daily bar on top of the simulated history
        daily = self._daily_history(day, i)
        daily.append({
            "time": session.session_date.isoformat(),
            "open": candles[0]["open"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "close": candles[-1]["close"],
            "volume": sum(c["volume"] for c in candles),
        })

        components = {
            "momentum": round(float(m["momentum"][i]), 2),
            "trend": round(float(m["trend"][i]), 2),
            "open_interest": round(float(m["positions"][i]), 2),
            "volume": round(float(m["volume"][i]), 2),
        }
        row.update({
            "open": candles[0]["open"],
            "day_high": max(c["high"] for c in candles),
            "day_low": min(c["low"] for c in candles),
            "sector_change_pct": round(sector_change, 2),
            "sector_rank": sector_rank,
            "sector_count": len(self._sector_names),
            "levels": levels,
            "score_components": components,
            "score_weights": SCORE_WEIGHTS,
            "reading": self._reading(row, levels, sector_change, sector_rank),
            "candles": {"intraday": candles, "daily": daily},
            "meta": {
                "session_date": session.session_date.isoformat(),
                "as_of": minute_to_label(minute),
                "market_state": session.state,
                "is_replay": minute < session.live_minute,
                "data_mode": self._data_mode,
            },
        })
        return row

    def _reading(self, row: Dict[str, Any], levels: Dict[str, Any], sector_change: float, sector_rank: int) -> List[str]:
        """Plain-language description of what the numbers show. Descriptive, not a call."""
        change, oi = row["change_pct"], row["oi_change_pct"]
        lines: List[str] = []
        meaning = {
            "long_buildup": "price and open interest are rising together, which usually means new long positions are being added",
            "short_covering": "price is rising while open interest falls, which usually means short positions are being closed rather than new longs added",
            "short_buildup": "price is falling while open interest rises, which usually means new short positions are being added",
            "long_unwinding": "price and open interest are falling together, which usually means long positions are being closed",
            "flat": "price is close to yesterday's close, so the open-interest change says little about direction yet",
        }[row["buildup"]]
        lines.append(
            f"{row['symbol']} is {'up' if change >= 0 else 'down'} {abs(change):.2f}% with open interest "
            f"{'up' if oi >= 0 else 'down'} {abs(oi):.2f}%: {meaning}."
        )
        quality = row["trend_quality"]
        tone = "a clean, one-directional move" if quality >= 0.35 else ("a steady move with pullbacks" if quality >= CHOPPY_EFFICIENCY else "back-and-forth trade with little net progress")
        lines.append(f"Trend quality is {quality:.2f} ({tone}) and volume is running at {row['volume_ratio']:.1f}x its usual pace for this time of day.")
        if row["bias"] == "choppy":
            lines.append("It is tagged choppy, so it is left off the bullish and bearish boards.")
        zone = levels.get("zone")
        if zone:
            state = {
                "holding": "Price is trading away from the zone",
                "retesting": "Price is back inside the zone now",
                "broken": "Price has traded through the zone, so it no longer holds",
            }[zone["state"]]
            lines.append(
                f"The strongest push started at {zone['impulse_from']} from a {zone['side']} zone at "
                f"{zone['low']:,.2f}-{zone['high']:,.2f}. {state}."
            )
        lines.append(
            f"{row['sector']} is {'up' if sector_change >= 0 else 'down'} {abs(sector_change):.2f}% on average, "
            f"ranked {sector_rank} of {len(self._sector_names)} sectors."
        )
        return lines
