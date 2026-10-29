"""
News ingestion package.
"""
from stock_engine.rag.ingest.rss import RSSIngester, NewsArticle, NewsSource

__all__ = ["RSSIngester", "NewsArticle", "NewsSource"]
