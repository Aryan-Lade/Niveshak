"""
Lesson content for the Learn section of the Niveshak terminal.

Twelve short, text-first lessons in four modules. Each lesson explains one
idea the terminal uses, in the same words the screens use, so a reader can
check what a number means before acting on it. Nothing here is advice.
"""

from __future__ import annotations

from typing import Any, Dict, List

COURSE: List[Dict[str, Any]] = [
    {
        "id": "tape",
        "title": "Reading the session",
        "summary": "What price, volume and breadth tell you before you look at a single chart.",
        "lessons": [
            {
                "id": "tape-1",
                "title": "Breadth: how many stocks are actually moving",
                "minutes": 4,
                "body": [
                    "An index can be up half a percent because three heavyweights are carrying it while most stocks are flat or red. Breadth counts stocks instead of weighting them: how many are above yesterday's close, how many are below.",
                    "The session ribbon at the top of the terminal draws that count through the day. A market that opens with 140 stocks up and drifts to 90 by lunch is losing participation even if the index has not moved much.",
                    "Breadth is context, not a signal. It tells you whether a long idea is swimming with the tide or against it.",
                ],
                "takeaways": [
                    "Breadth = advancing stocks vs declining stocks, each stock counted once.",
                    "Falling breadth under a flat index means fewer stocks are doing the work.",
                    "Use it to judge the backdrop, not to time an entry.",
                ],
                "check": {
                    "question": "The index is up 0.4% but only 70 of 207 stocks are advancing. What does that suggest?",
                    "answer": "The gain is narrow - a few large stocks are lifting the index while most of the market is lower.",
                },
            },
            {
                "id": "tape-2",
                "title": "Trend quality: a move versus noise",
                "minutes": 5,
                "body": [
                    "Two stocks can both be up 1%. One climbed steadily all morning. The other went up 2%, down 1.5% and back. The first is a trend; the second is chop that happens to be positive right now.",
                    "The terminal measures this with an efficiency ratio: the net move divided by the total distance travelled, using five-minute closes. A straight line scores 1. A stock that travels 6% to end 0.3% higher scores 0.05.",
                    "Stocks with very low trend quality, or with almost no net move, are tagged choppy and kept off the bullish and bearish boards. That is the single most useful filter in the product: it removes the names where direction is not yet decided.",
                ],
                "takeaways": [
                    "Trend quality = |net move| / sum of all the small moves.",
                    "High values mean one-directional trade; low values mean back-and-forth.",
                    "Choppy stocks score zero and are left off the boards.",
                ],
                "check": {
                    "question": "A stock's five-minute closes add up to 4% of travel and it is 1% above where it started. What is its trend quality?",
                    "answer": "0.25 - one quarter of the distance travelled turned into net progress.",
                },
            },
            {
                "id": "tape-3",
                "title": "Relative volume: is anyone behind the move?",
                "minutes": 4,
                "body": [
                    "Raw volume is hard to read because every stock trades heavily in the first and last half hour. Relative volume fixes that by comparing volume so far today with what the stock usually trades by the same time of day.",
                    "A reading of 1.0x is an ordinary day. Above about 1.5x, more participants than usual are involved. A sharp price move on 0.6x volume deserves more suspicion than the same move on 2x.",
                ],
                "takeaways": [
                    "Relative volume compares today with the stock's own usual pace, by time of day.",
                    "Moves on heavy relative volume have more participation behind them.",
                    "Thin moves reverse more easily.",
                ],
                "check": {
                    "question": "Why not just compare today's volume at 11:00 with yesterday's full-day volume?",
                    "answer": "Because volume is not spread evenly through the day; the comparison has to be for the same time of day.",
                },
            },
        ],
    },
    {
        "id": "oi",
        "title": "Open interest",
        "summary": "The one data point futures traders have that cash traders do not, and its limits.",
        "lessons": [
            {
                "id": "oi-1",
                "title": "What open interest is, and is not",
                "minutes": 5,
                "body": [
                    "Open interest (OI) is the number of futures or options contracts that are open - created and not yet closed. Every contract has one buyer and one seller, so OI is not bullish or bearish on its own.",
                    "OI rises when a new buyer meets a new seller. It falls when an existing long and an existing short both close. If a long sells to a new buyer, OI does not change at all.",
                    "Volume counts trades; OI counts positions still standing. A day can have huge volume and flat OI if it was mostly intraday traders passing contracts back and forth.",
                ],
                "takeaways": [
                    "OI = contracts currently open. One long and one short per contract.",
                    "Rising OI means new positions; falling OI means positions being closed.",
                    "OI has no direction until you read it alongside price.",
                ],
                "check": {
                    "question": "A trader holding a long future sells it to someone opening a fresh long. What happens to OI?",
                    "answer": "Nothing. One long replaced another; the number of open contracts is unchanged.",
                },
            },
            {
                "id": "oi-2",
                "title": "The four build-ups",
                "minutes": 6,
                "body": [
                    "Put price change and OI change side by side and you get four cases. Price up with OI up is a long build-up: new positions are being added as price rises. Price down with OI up is a short build-up: new positions are being added as price falls.",
                    "Price up with OI down is short covering: the rise is coming from shorts closing, not fresh buying. Price down with OI down is long unwinding: the fall is coming from longs closing.",
                    "The terminal uses solid markers for the two build-ups (fresh positions) and hollow markers for covering and unwinding (positions closing). Build-ups tend to say more about conviction; covering and unwinding moves can run out once the closing is done.",
                ],
                "takeaways": [
                    "Up + OI up = long build-up. Down + OI up = short build-up.",
                    "Up + OI down = short covering. Down + OI down = long unwinding.",
                    "Solid marker = fresh positions. Hollow marker = positions closing.",
                ],
                "check": {
                    "question": "A stock is up 2.4% and its OI is down 3%. Which class is it, and what is driving the rise?",
                    "answer": "Short covering. Shorts are buying back to close, which lifts price while OI falls.",
                },
            },
            {
                "id": "oi-3",
                "title": "Where the OI read goes wrong",
                "minutes": 5,
                "body": [
                    "The four-way table is a framework, not a fact about who is trading. A rise in OI on a falling stock could be new shorts - or it could be investors hedging shares they already own, which is not a bearish view at all.",
                    "Near monthly expiry, OI in the expiring contract falls and OI in the next contract rises simply because positions are being rolled. Read combined OI across contracts in expiry week, or the picture will look like mass unwinding.",
                    "Exchange OI also updates in steps rather than tick by tick, so in the first minutes of a session the OI change is thin. This is why the boards carry a settling note until a few bars have printed.",
                ],
                "takeaways": [
                    "OI cannot tell hedging from speculation.",
                    "Rollover in expiry week distorts single-contract OI.",
                    "Early-session OI readings are unreliable.",
                ],
                "check": {
                    "question": "In expiry week a stock's near-month OI drops 30% while price is flat. Is that long unwinding?",
                    "answer": "Probably not. Positions are most likely rolling to the next month; check combined OI.",
                },
            },
        ],
    },
    {
        "id": "levels",
        "title": "Zones and levels",
        "summary": "How the terminal draws its levels, so you can redraw them yourself.",
        "lessons": [
            {
                "id": "levels-1",
                "title": "Pivot levels: S1, S2, R1, R2",
                "minutes": 4,
                "body": [
                    "Pivot levels are arithmetic on yesterday's high, low and close. The pivot is their average. R1 is twice the pivot minus yesterday's low; S1 is twice the pivot minus yesterday's high. R2 and S2 sit one full range of yesterday above and below the pivot.",
                    "They matter because many intraday traders watch the same numbers, which makes them natural places for orders to cluster. They are reference points, not walls.",
                ],
                "takeaways": [
                    "Pivot = (high + low + close) / 3 from the previous session.",
                    "R1 = 2 x pivot - low. S1 = 2 x pivot - high.",
                    "R2 / S2 = pivot plus or minus yesterday's range.",
                ],
                "check": {
                    "question": "Yesterday: high 510, low 490, close 500. What are the pivot and R1?",
                    "answer": "Pivot = 500. R1 = 2 x 500 - 490 = 510.",
                },
            },
            {
                "id": "levels-2",
                "title": "Order blocks: where a move began",
                "minutes": 6,
                "body": [
                    "An order block is the last candle that closed against a strong move before that move started. For a rally, it is the last down candle before the push higher. The idea is that large orders were being filled there, and price often reacts when it returns.",
                    "The terminal finds it mechanically. It looks for the strongest three-bar push in the stock's direction on five-minute candles, then walks back up to six bars to the last candle that closed the other way. That candle's high-to-low range is the zone.",
                    "A zone has three states. Holding: price is trading away from it. Retesting: price is back inside it. Broken: price has traded through it, and the zone is dropped. A broken zone gets no plan.",
                ],
                "takeaways": [
                    "Order block = last opposite-colour candle before the strongest push.",
                    "The zone is that candle's full range.",
                    "A zone that price trades through is broken and no longer used.",
                ],
                "check": {
                    "question": "A bullish zone is 498-502 and the stock is now at 495. What state is the zone in?",
                    "answer": "Broken. Price has traded below the bottom of the zone.",
                },
            },
            {
                "id": "levels-3",
                "title": "From a zone to a plan",
                "minutes": 5,
                "body": [
                    "Each board row shows an illustrative plan: an entry at the near edge of the zone, a stop just beyond the far edge, and the next pivot level as the first target. Reward-to-risk is the distance to the target divided by the distance to the stop.",
                    "The plan describes a pullback entry. If price never returns to the zone there is no trade, and chasing it at the current price gives a very different, usually much worse, reward-to-risk.",
                    "Treat the plan as a worked example of how the levels fit together. Whether to trade it, and how much, is your decision.",
                ],
                "takeaways": [
                    "Entry at the near edge of the zone, stop beyond the far edge.",
                    "Reward-to-risk = (target - entry) / (entry - stop) for a long.",
                    "No pullback, no trade. Chasing changes the maths.",
                ],
                "check": {
                    "question": "Entry 500, stop 495, target 515. What is the reward-to-risk?",
                    "answer": "3.0 - fifteen points of reward for five points of risk.",
                },
            },
        ],
    },
    {
        "id": "risk",
        "title": "Risk first",
        "summary": "Sizing, costs and record-keeping - the parts that decide whether an edge survives.",
        "lessons": [
            {
                "id": "risk-1",
                "title": "Position size from the stop, not from conviction",
                "minutes": 5,
                "body": [
                    "Decide how much of the account one trade may lose - a common ceiling is 1% - and let the stop distance set the quantity. Quantity = amount at risk divided by risk per share.",
                    "With Rs 5,00,000 and a 1% limit, the most a trade may lose is Rs 5,000. If the stop is Rs 4 away from entry, that is 1,250 shares. A wider stop means fewer shares, not more risk.",
                    "Niveshak's own risk engine does the same thing with an ATR-based stop, caps any one stock at a share of the portfolio and halts new entries after a daily loss limit. The limits it is running with are listed under Settings.",
                ],
                "takeaways": [
                    "Fix the loss per trade first; quantity follows from the stop distance.",
                    "Quantity = (account x risk %) / risk per share.",
                    "Wider stop, smaller size.",
                ],
                "check": {
                    "question": "Account Rs 2,00,000, 1% risk, stop Rs 5 from entry. How many shares?",
                    "answer": "400 shares: Rs 2,000 at risk divided by Rs 5 per share.",
                },
            },
            {
                "id": "risk-2",
                "title": "Costs: the trade has to clear them first",
                "minutes": 4,
                "body": [
                    "Every round trip in Indian F&O pays brokerage, securities transaction tax, exchange transaction charges, SEBI fees, stamp duty and GST on the fee components, plus slippage between the price you wanted and the price you got.",
                    "These are small per trade and large in aggregate. A strategy that makes ten trades a day needs a much bigger gross edge than one that makes two. The rates change with the Union Budget and exchange circulars, so check the current schedule rather than relying on memory.",
                    "The backtests in this project net out all of these using the schedule in configs/costs.yaml. Keep that file current and the results stay honest.",
                ],
                "takeaways": [
                    "Brokerage, STT, exchange charges, SEBI fees, stamp duty, GST and slippage all apply.",
                    "More trades means a higher hurdle.",
                    "Rates change; verify them before relying on a backtest.",
                ],
                "check": {
                    "question": "Two strategies have the same gross return. One trades five times as often. Which is likely better after costs?",
                    "answer": "The one that trades less - it pays the cost stack far fewer times.",
                },
            },
            {
                "id": "risk-3",
                "title": "Keep a journal, and read it",
                "minutes": 4,
                "body": [
                    "SEBI's study of individual F&O traders found that roughly nine in ten lost money over the period it examined. The few who last tend to have one habit in common: they know their own numbers.",
                    "Log every trade with the setup that triggered it, the planned stop and target, and the result. After thirty trades the journal will tell you things no screen can: which setups pay, whether you cut winners early, and what your average loss is against your average win.",
                    "The Journal section of this terminal computes win rate, average win and loss, profit factor and an equity curve from what you log. It is stored on your device.",
                ],
                "takeaways": [
                    "Most individual F&O traders lose money; assume the base rate applies to you.",
                    "Record setup, plan and result for every trade.",
                    "Review win rate together with average win vs average loss - neither means much alone.",
                ],
                "check": {
                    "question": "A trader wins 60% of trades but the average loss is twice the average win. Profitable?",
                    "answer": "No. 0.6 x 1 - 0.4 x 2 = -0.2 per unit risked; the losses outweigh the wins.",
                },
            },
        ],
    },
]


def course_outline() -> Dict[str, Any]:
    """The full course plus totals, ready to serialise."""
    lessons = [lesson for module in COURSE for lesson in module["lessons"]]
    return {
        "modules": COURSE,
        "lesson_count": len(lessons),
        "total_minutes": sum(lesson["minutes"] for lesson in lessons),
    }
