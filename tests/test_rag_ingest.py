"""
Unit tests for Day 18 RSS news ingestion, deduplication, and entity linking.
"""

import time
from pathlib import Path
import feedparser
import pytest

from stock_engine.rag.ingest.rss import NewsArticle, RSSIngester, SQLiteNewsStore
from stock_engine.rag.link import TickerLinker

# Recorded mock RSS XML fixture (no internet required)
MOCK_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Market Test Feed</title>
  <link>https://example.com</link>
  <description>Test Financial News</description>
  <item>
    <title>Reliance Industries announces new petrochemical facility in Gujarat</title>
    <link>https://example.com/news/1</link>
    <description>RIL management confirmed capital expenditure plans today.</description>
    <pubDate>Mon, 20 Oct 2026 09:30:00 GMT</pubDate>
  </item>
  <item>
    <title>TCS signs landmark 5-year digital transformation deal with Bharti Airtel</title>
    <link>https://example.com/news/2</link>
    <description>Tata Consultancy Services and Airtel partner on cloud infrastructure.</description>
    <pubDate>Mon, 20 Oct 2026 10:15:00 GMT</pubDate>
  </item>
  <item>
    <title>Info Edge acquires stake in AI startup Naukri tech labs</title>
    <link>https://example.com/news/3</link>
    <description>Info Edge ventures into specialized enterprise software.</description>
    <pubDate>Mon, 20 Oct 2026 11:00:00 GMT</pubDate>
  </item>
</channel>
</rss>
"""


def test_rss_xml_parsing_and_timezone_conversion(tmp_path):
    """Test parsing recorded RSS feed into normalized NewsArticle objects with UTC timestamps."""
    parsed = feedparser.parse(MOCK_RSS_XML)
    ingester = RSSIngester(db_path=tmp_path / "test_news.db")

    articles = [ingester.parse_feed_entry(entry, "TestFeed") for entry in parsed.entries]
    assert len(articles) == 3

    art1 = articles[0]
    assert art1.title == "Reliance Industries announces new petrochemical facility in Gujarat"
    assert art1.url == "https://example.com/news/1"
    # Check UTC epoch conversion
    assert isinstance(art1.published_at, int)
    assert art1.published_at > 1700000000
    assert not art1.has_synthetic_timestamp


def test_deduplication_exact_and_fuzzy(tmp_path):
    """Verify exact URL duplicates and fuzzy title duplicates are rejected."""
    store = SQLiteNewsStore(db_path=tmp_path / "dedupe.db")

    art1 = NewsArticle(
        id="sha_1",
        title="HDFC Bank reports 18% profit growth in Q2 results",
        summary="Quarterly results exceed analyst expectations.",
        source="ET",
        url="https://example.com/hdfc-q2",
        published_at=int(time.time()),
        fetched_at=int(time.time()),
    )

    # 1. First insertion must succeed
    assert store.insert_article(art1, ["HDFCBANK.NS"]) is True

    # 2. Duplicate URL / ID must be rejected
    assert store.insert_article(art1, ["HDFCBANK.NS"]) is False

    # 3. Fuzzy duplicate title must be rejected
    fuzzy_art = NewsArticle(
        id="sha_2",
        title="HDFC Bank reports 18% profit growth in Q2 result",  # Near identical
        summary="Different summary text.",
        source="Mint",
        url="https://example.com/hdfc-q2-mint",
        published_at=int(time.time()),
        fetched_at=int(time.time()),
    )
    assert store.insert_article(fuzzy_art, ["HDFCBANK.NS"]) is False


def test_ticker_linking_tricky_names():
    """
    Test entity linking with edge cases:
    - Infosys vs Info Edge
    - Tata Consultancy Services vs Tata Motors
    - Multi-ticker matches
    """
    linker = TickerLinker()

    # 1. Infosys positive match
    res1 = linker.link_text("Infosys beats street estimates with 5% dollar revenue growth")
    assert "INFY.NS" in res1

    # 2. Tricky negative match: Info Edge must NOT match INFY.NS
    res2 = linker.link_text("Info Edge shares surge 4% following Zomato stake sale")
    assert "INFY.NS" not in res2

    # 3. Multi-ticker match: TCS and Bharti Airtel
    res3 = linker.link_text("TCS partners with Bharti Airtel to deploy 5G enterprise networks across India")
    assert "TCS.NS" in res3
    assert "BHARTIARTL.NS" in res3

    # 4. Symbol alias match: SBI
    res4 = linker.link_text("State Bank of India raises deposit rates by 25 bps; SBI stock gains")
    assert "SBIN.NS" in res4
