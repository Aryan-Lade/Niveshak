"""
RSS news ingestion, normalization, deduplication, and SQLite FTS5 persistence.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import yaml

from stock_engine.rag.link import TickerLinker

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Normalized financial news article."""
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: int  # UTC Epoch seconds
    fetched_at: int    # UTC Epoch seconds
    has_synthetic_timestamp: bool = False


class NewsSource(ABC):
    """Abstract interface for historical and live news sources."""

    @abstractmethod
    def fetch_articles(self) -> List[NewsArticle]:
        """Fetch and return normalized articles."""
        pass


class SQLiteNewsStore:
    """
    Manages SQLite storage with full-text search (FTS5) and ticker mapping.
    """

    def __init__(self, db_path: Path = Path("data_store/news.db")) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and FTS5 index if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    url TEXT UNIQUE,
                    published_at INTEGER NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    has_synthetic_timestamp INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_ticker (
                    doc_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    PRIMARY KEY (doc_id, ticker),
                    FOREIGN KEY (doc_id) REFERENCES news_articles(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_published 
                ON news_articles(published_at)
            """)

            # Create FTS5 virtual table if supported
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS news_fts USING fts5(
                        id UNINDEXED,
                        title,
                        summary,
                        source
                    )
                """)
            except sqlite3.OperationalError:
                logger.warning("FTS5 extension not supported in this SQLite build, continuing with standard tables.")

            conn.commit()

    def article_exists(self, article_id: str, url: str) -> bool:
        """Check if article already exists by ID or URL."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM news_articles WHERE id = ? OR url = ? LIMIT 1",
                (article_id, url),
            )
            return cursor.fetchone() is not None

    def is_fuzzy_duplicate(self, title: str, threshold: float = 0.85) -> bool:
        """Check if a substantially similar title was published within the last 48 hours."""
        recent_cutoff = int(time.time()) - (48 * 3600)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT title FROM news_articles WHERE fetched_at >= ? LIMIT 200",
                (recent_cutoff,),
            )
            existing_titles = [row[0] for row in cursor.fetchall()]

        clean_title = title.strip().lower()
        for existing in existing_titles:
            sim = difflib.SequenceMatcher(None, clean_title, existing.strip().lower()).ratio()
            if sim >= threshold:
                return True
        return False

    def insert_article(self, article: NewsArticle, tickers: List[str]) -> bool:
        """Insert article, link tickers, and update FTS5 index."""
        if self.article_exists(article.id, article.url):
            return False
        if self.is_fuzzy_duplicate(article.title):
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO news_articles (
                    id, title, summary, source, url, published_at, fetched_at, has_synthetic_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.id,
                    article.title,
                    article.summary,
                    article.source,
                    article.url,
                    article.published_at,
                    article.fetched_at,
                    1 if article.has_synthetic_timestamp else 0,
                ),
            )

            # Insert ticker links
            for ticker in tickers:
                cursor.execute(
                    "INSERT OR IGNORE INTO news_ticker (doc_id, ticker) VALUES (?, ?)",
                    (article.id, ticker),
                )

            # Insert into FTS5
            try:
                cursor.execute(
                    "INSERT INTO news_fts (id, title, summary, source) VALUES (?, ?, ?, ?)",
                    (article.id, article.title, article.summary, article.source),
                )
            except Exception:
                pass

            conn.commit()
            return True


class RSSIngester(NewsSource):
    """
    Ingests and normalizes RSS feeds into SQLite store.
    """

    def __init__(
        self,
        config_path: Path = Path("configs/news_sources.yaml"),
        db_path: Path = Path("data_store/news.db"),
    ) -> None:
        self.config_path = config_path
        self.store = SQLiteNewsStore(db_path=db_path)
        self.linker = TickerLinker()

    def parse_feed_entry(self, entry: Any, source_name: str) -> Optional[NewsArticle]:
        """Convert a feedparser entry to a normalized NewsArticle."""
        title = entry.get("title", "").strip()
        url = entry.get("link", "").strip()
        if not title or not url:
            return None

        summary = entry.get("summary", "") or entry.get("description", "")
        summary = summary.strip()

        # Generate deterministic sha1 id from canonical url
        article_id = hashlib.sha1(url.encode("utf-8")).hexdigest()

        # Parse publication date
        published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        has_synthetic = False
        now_epoch = int(time.time())

        if published_parsed:
            try:
                dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                published_at = int(dt.timestamp())
            except Exception:
                published_at = now_epoch
                has_synthetic = True
        else:
            published_at = now_epoch
            has_synthetic = True

        return NewsArticle(
            id=article_id,
            title=title,
            summary=summary,
            source=source_name,
            url=url,
            published_at=published_at,
            fetched_at=now_epoch,
            has_synthetic_timestamp=has_synthetic,
        )

    def fetch_feed(self, url: str, source_name: str) -> List[NewsArticle]:
        """Fetch and parse entries from an RSS feed URL."""
        parsed = feedparser.parse(url)
        articles: List[NewsArticle] = []
        for entry in parsed.entries:
            art = self.parse_feed_entry(entry, source_name)
            if art:
                articles.append(art)
        return articles

    def fetch_articles(self) -> List[NewsArticle]:
        """Fetch from all configured enabled feeds."""
        if not self.config_path.exists():
            return []

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        sources = cfg.get("news_sources", [])
        all_articles: List[NewsArticle] = []
        for src in sources:
            if src.get("enabled", True):
                name = src.get("name", "Unknown")
                url = src.get("url")
                if url:
                    try:
                        arts = self.fetch_feed(url, name)
                        all_articles.extend(arts)
                    except Exception as e:
                        logger.error(f"Error fetching RSS feed '{name}' ({url}): {e}")
        return all_articles

    def ingest_once(self) -> int:
        """Run single ingestion pass and return number of newly added articles."""
        articles = self.fetch_articles()
        inserted_count = 0
        for art in articles:
            text_for_linking = f"{art.title} {art.summary}"
            tickers = self.linker.link_text(text_for_linking)
            if self.store.insert_article(art, tickers):
                inserted_count += 1
        logger.info(f"Ingested {inserted_count} new articles across all feeds.")
        return inserted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial news RSS ingester")
    parser.add_argument("--once", action="store_true", help="Run ingestion once and exit")
    args = parser.parse_args()

    ingester = RSSIngester()
    if args.once:
        count = ingester.ingest_once()
        print(f"Ingested {count} new articles.")
