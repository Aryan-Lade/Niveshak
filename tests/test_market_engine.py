"""
Tests for the market snapshot engine and the terminal API routes.

The engine is driven with a fixed clock so every assertion is deterministic.
"""

import datetime as dt

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stock_engine.analytics import market_engine as me
from stock_engine.analytics.lessons import course_outline
from stock_engine.analytics.universe import SECTORS, UNIVERSE
from stock_engine.api.main import app


def _at(day: int, hour: int, minute: int) -> dt.datetime:
    return dt.datetime(2026, 11, day, hour, minute, tzinfo=me.IST)


@pytest.fixture
def engine():
    # Thursday 12 Nov 2026, 13:37 IST - mid-session
    return me.MarketEngine(clock=lambda: _at(12, 13, 37))


# ---------------------------------------------------------------- clock

def test_session_clock_open_closed_and_weekend():
    mid = me.resolve_session(_at(12, 13, 37))
    assert (mid.state, mid.session_date.day, mid.live_minute) == ("open", 12, 262)

    first_minute = me.resolve_session(_at(12, 9, 15))
    assert first_minute.state == "open" and first_minute.live_minute == 1

    after_close = me.resolve_session(_at(12, 15, 30))
    assert (after_close.state, after_close.live_minute) == ("closed", me.SESSION_MINUTES)

    pre_open = me.resolve_session(_at(12, 8, 0))
    assert (pre_open.state, pre_open.session_date.day) == ("pre_open", 11)

    saturday = me.resolve_session(_at(14, 12, 0))
    assert (saturday.state, saturday.session_date.day) == ("closed", 13)

    monday_pre = me.resolve_session(_at(16, 7, 0))
    assert monday_pre.session_date.day == 13  # rolls back over the weekend


def test_time_labels_round_trip():
    assert me.minute_to_label(0) == "09:15"
    assert me.minute_to_label(375) == "15:30"
    assert me.label_to_minute("11:30") == 135
    with pytest.raises(ValueError):
        me.label_to_minute("half past ten")
    with pytest.raises(ValueError):
        me.label_to_minute("25:99")


# ---------------------------------------------------------------- helpers

@pytest.mark.parametrize(
    "price, oi, expected",
    [
        (1.2, 3.0, "long_buildup"),
        (1.2, -3.0, "short_covering"),
        (-1.2, 3.0, "short_buildup"),
        (-1.2, -3.0, "long_unwinding"),
        (0.01, 5.0, "flat"),
    ],
)
def test_classify_buildup(price, oi, expected):
    assert me.classify_buildup(price, oi) == expected


def test_efficiency_ratio():
    assert me.efficiency_ratio(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert me.efficiency_ratio(np.array([0.0, 1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert me.efficiency_ratio(np.array([0.0, 2.0, 0.5, 1.0])) == pytest.approx(1.0 / 4.0)
    assert me.efficiency_ratio(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)


def test_pivot_levels_match_the_textbook_formula():
    levels = me.pivot_levels(high=510, low=490, close=500)
    assert levels == {"pivot": 500, "r1": 510, "s1": 490, "r2": 520, "s2": 480}


def _candle(time, o, h, l, c):
    return {"time": time, "open": o, "high": h, "low": l, "close": c, "volume": 1}


def test_find_order_block_picks_last_opposite_candle_before_the_push():
    candles = [
        _candle("09:15", 100, 101, 99.5, 100.5),
        _candle("09:20", 100.5, 100.8, 99.8, 100.0),   # down candle  <- the block
        _candle("09:25", 100.0, 102.0, 99.9, 101.9),   # push starts here
        _candle("09:30", 101.9, 104.0, 101.8, 103.8),
        _candle("09:35", 103.8, 105.5, 103.6, 105.2),
        _candle("09:40", 105.2, 105.4, 104.8, 105.0),
    ]
    block = me.find_order_block(candles, "bullish")
    assert block["formed_at"] == "09:20"
    assert (block["low"], block["high"]) == (99.8, 100.8)
    assert block["impulse_from"] == "09:25"
    assert me.find_order_block(candles[:4], "bullish") is None  # too little data


def test_tick_round():
    assert me.tick_round(123.4567) == 123.45
    assert me.tick_round(9.537) == 9.54


# ---------------------------------------------------------------- snapshot

def test_snapshot_is_internally_consistent(engine):
    snap = engine.snapshot()
    meta, breadth = snap["meta"], snap["breadth"]

    assert meta["data_mode"] == "simulated"
    assert meta["as_of"] == "13:37" and meta["market_state"] == "open" and not meta["is_replay"]
    assert meta["universe"] == len(UNIVERSE) == len(snap["stocks"])

    # Breadth adds up and agrees with the stock rows and the histogram
    assert breadth["advancing"] + breadth["declining"] + breadth["unchanged"] == len(UNIVERSE)
    assert breadth["advancing"] == sum(1 for s in snap["stocks"] if s["change_pct"] > me.FLAT_BAND_PCT)
    assert sum(b["count"] for b in breadth["histogram"]) == len(UNIVERSE)
    assert sum(breadth["buildup"].values()) == len(UNIVERSE)

    # Sectors cover the universe exactly once and are sorted strongest first
    assert sorted(s["name"] for s in snap["sectors"]) == SECTORS
    assert sum(s["count"] for s in snap["sectors"]) == len(UNIVERSE)
    changes = [s["change_pct"] for s in snap["sectors"]]
    assert changes == sorted(changes, reverse=True)

    # The ribbon's last point is "now"
    assert snap["timeline"]["advancing"][-1] == breadth["advancing"]
    assert snap["timeline"]["times"][-1] == "13:37"


def test_every_stock_row_follows_the_rules(engine):
    for row in engine.snapshot()["stocks"]:
        assert row["buildup"] == me.classify_buildup(row["change_pct"], row["oi_change_pct"]) or abs(abs(row["change_pct"]) - me.FLAT_BAND_PCT) < 0.01
        assert 0 <= row["score"] <= 100
        if row["bias"] == "choppy":
            assert row["score"] == 0
        else:
            assert row["bias"] == ("bullish" if row["change_pct"] > 0 else "bearish")
        assert row["ltp"] == pytest.approx(row["prev_close"] * (1 + row["change_pct"] / 100), rel=2e-3)


def test_boards_are_ranked_and_plans_are_coherent(engine):
    boards = engine.snapshot()["boards"]
    for side in ("bullish", "bearish"):
        rows = boards[side]
        assert len(rows) == 10
        assert [r["rank"] for r in rows] == list(range(1, 11))
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True) and scores[-1] > 0
        for row in rows:
            assert row["bias"] == side
            levels = row["levels"]
            assert levels["s2"] < levels["s1"] < levels["pivot"] < levels["r1"] < levels["r2"]
            zone, plan = levels["zone"], levels["plan"]
            if zone:
                assert zone["low"] <= zone["high"]
                assert zone["side"] == ("demand" if side == "bullish" else "supply")
            if plan:
                assert zone["state"] != "broken"
                if side == "bullish":
                    assert plan["stop"] < plan["entry"] < plan["targets"][0]
                else:
                    assert plan["stop"] > plan["entry"] > plan["targets"][0]
                assert plan["reward_to_risk"] > 0


def test_snapshot_is_deterministic_and_replay_is_a_prefix(engine):
    assert engine.snapshot()["stocks"] == engine.snapshot()["stocks"]

    live = engine.snapshot()
    replay = engine.snapshot(at="11:00")
    assert replay["meta"]["is_replay"] and replay["meta"]["as_of"] == "11:00"
    # The ribbon always covers the whole live session, even in replay
    assert replay["timeline"] == live["timeline"]
    # Replayed breadth equals the ribbon's value at that time
    idx = live["timeline"]["times"].index("11:00")
    assert replay["breadth"]["advancing"] == live["timeline"]["advancing"][idx]
    # A replay never looks past its own minute
    assert len(replay["stocks"][0]["spark"]) < len(live["stocks"][0]["spark"])
    # Asking for a time in the future clamps to now
    assert engine.snapshot(at="15:00")["meta"]["as_of"] == "13:37"


def test_different_days_differ():
    a = me.MarketEngine(clock=lambda: _at(12, 15, 45)).snapshot()
    b = me.MarketEngine(clock=lambda: _at(11, 15, 45)).snapshot()
    assert a["meta"]["session_date"] != b["meta"]["session_date"]
    assert [s["change_pct"] for s in a["stocks"]] != [s["change_pct"] for s in b["stocks"]]


def test_early_session_is_flagged_as_settling():
    snap = me.MarketEngine(clock=lambda: _at(12, 9, 20)).snapshot()
    assert snap["meta"]["settling"] is True
    assert len(snap["stocks"][0]["spark"]) == 2  # open + one bar


# ---------------------------------------------------------------- stock detail

def test_stock_detail(engine):
    detail = engine.stock("m&m")
    assert detail["symbol"] == "M&M"
    intraday, daily = detail["candles"]["intraday"], detail["candles"]["daily"]
    assert len(intraday) == 53 and intraday[0]["time"] == "09:15"
    for c in intraday + daily:
        assert c["low"] <= min(c["open"], c["close"]) <= max(c["open"], c["close"]) <= c["high"]
    assert len(daily) == 61 and daily[-1]["time"] == "2026-11-12"
    assert daily[-1]["close"] == detail["ltp"] == intraday[-1]["close"]
    assert set(detail["score_components"]) == set(detail["score_weights"])
    assert detail["reading"] and detail["reading"][0].startswith("M&M is ")
    assert engine.stock("NOTASTOCK") is None


def test_score_is_the_weighted_sum_of_its_components(engine):
    detail = engine.stock(engine.snapshot()["boards"]["bullish"][0]["symbol"])
    expected = 100 * sum(detail["score_weights"][k] * detail["score_components"][k] for k in detail["score_weights"])
    assert detail["score"] == pytest.approx(expected, abs=1.5)


# ---------------------------------------------------------------- lessons

def test_course_outline():
    course = course_outline()
    assert course["lesson_count"] == 12 and len(course["modules"]) == 4
    ids = [lesson["id"] for module in course["modules"] for lesson in module["lessons"]]
    assert len(set(ids)) == 12
    for module in course["modules"]:
        for lesson in module["lessons"]:
            assert lesson["body"] and lesson["takeaways"] and lesson["check"]["answer"]


# ---------------------------------------------------------------- API

@pytest.fixture
def client():
    return TestClient(app)


def test_terminal_page_and_static_assets(client):
    for path in ("/", "/niveshak", "/app", "/orderblock"):
        resp = client.get(path)
        assert resp.status_code == 200 and "Niveshak" in resp.text
    assert client.get("/ui/css/app.css").status_code == 200
    assert client.get("/ui/js/main.js").status_code == 200
    assert client.get("/legacy").status_code == 200


def test_snapshot_and_stock_routes(client):
    snap = client.get("/api/v1/niveshak/snapshot")
    assert snap.status_code == 200
    body = snap.json()
    assert {"meta", "breadth", "timeline", "sectors", "stocks", "boards"} <= set(body)

    assert client.get("/api/v1/niveshak/snapshot", params={"at": "nonsense"}).status_code == 400
    assert client.get("/api/v1/niveshak/stocks/M%26M").json()["symbol"] == "M&M"
    assert client.get("/api/v1/niveshak/stocks/BAJAJ-AUTO").status_code == 200
    assert client.get("/api/v1/niveshak/stocks/NOPE").status_code == 404
    assert client.get("/api/v1/niveshak/lessons").json()["lesson_count"] == 12


def test_rebalance_plan_route_reads_positions(client, monkeypatch):
    """Regression: with an open position the route iterated the dict's keys and returned a 500."""
    from stock_engine.api import main as api_main
    from stock_engine.execution.broker import Position

    held = {"TCS.NS": Position("TCS.NS", 10, 3000.0, 3000.0, 0.0)}
    monkeypatch.setattr(api_main.PaperBroker, "get_positions", lambda self: held)
    monkeypatch.setenv("API_KEY", "test-secret-key-123")
    resp = client.post(
        "/portfolio/rebalance-plan",
        json={"target_weights": {"RELIANCE.NS": 0.1}, "market_prices": {"RELIANCE.NS": 1500.0, "TCS.NS": 3000.0}},
        headers={"X-API-Key": "test-secret-key-123"},
    )
    assert resp.status_code == 200, resp.text
    tickers = {order["ticker"] for order in resp.json()["orders"]}
    assert "TCS.NS" in tickers  # the held position is sold down to its zero target
