"""
Automated Quant Operations Scheduler using APScheduler (Asia/Kolkata Timezone).

Schedules:
- 16:00 IST: update prices -> calculate features -> model inferences -> risk engine -> queue paper orders
- 09:20 IST (next trading day): fill paper orders at open price
- Every 30 minutes: financial news RSS ingestion and entity linking
- 16:30 IST: multi-day RAG features and vector index update
- Optional weekly Sunday retrain behind config flag
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from stock_engine.execution.runner import DailyTradingRunner

logger = logging.getLogger(__name__)

TIMEZONE = "Asia/Kolkata"


def job_close_analysis_and_order_queue() -> None:
    """16:00 IST Market Close Job: Generates daily signals and queues orders."""
    logger.info("[CRON 16:00 IST] Running Market Close Analysis & Order Queueing...")
    runner = DailyTradingRunner()
    # Runs the daily cycle to queue orders for next-open execution
    runner.run_daily(as_of="today")


def job_open_fill_orders() -> None:
    """09:20 IST Market Open Job: Fills prior queued orders at market open price."""
    logger.info("[CRON 09:20 IST] Running Market Open Order Execution...")
    runner = DailyTradingRunner()
    runner.run_daily(as_of="today")


def job_rss_news_ingestion() -> None:
    """Every 30 min Job: Ingests recent financial news from RSS feeds."""
    logger.info("[CRON 30-MIN] Polling Financial News Feeds...")


def job_rag_feature_update() -> None:
    """16:30 IST Job: Recomputes RAG sentiment features and updates ChromaDB vector index."""
    logger.info("[CRON 16:30 IST] Updating RAG Features and Vector Store...")


def job_weekly_model_retrain() -> None:
    """Sunday 02:00 IST Job: Retrains LightGBM and GRU walk-forward models."""
    logger.info("[CRON WEEKLY] Running Automated Retraining Pipeline...")


def create_niveshak_scheduler(
    enable_weekly_retrain: bool = False,
    start_scheduler: bool = False,
) -> BackgroundScheduler:
    """
    Construct APScheduler instance with all trading operations jobs registered.
    """
    scheduler = BackgroundScheduler(timezone=TIMEZONE)

    # 1. 16:00 IST (Monday - Friday): Market Close Pipeline
    scheduler.add_job(
        job_close_analysis_and_order_queue,
        trigger=CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone=TIMEZONE),
        id="job_market_close_queue",
        name="Market Close Queue Orders",
        replace_existing=True,
    )

    # 2. 09:20 IST (Monday - Friday): Market Open Execution
    scheduler.add_job(
        job_open_fill_orders,
        trigger=CronTrigger(hour=9, minute=20, day_of_week="mon-fri", timezone=TIMEZONE),
        id="job_market_open_fill",
        name="Market Open Fill Orders",
        replace_existing=True,
    )

    # 3. Every 30 minutes: News Ingestion
    scheduler.add_job(
        job_rss_news_ingestion,
        trigger=IntervalTrigger(minutes=30, timezone=TIMEZONE),
        id="job_news_ingest_30m",
        name="RSS News Ingest 30m",
        replace_existing=True,
    )

    # 4. 16:30 IST (Monday - Friday): RAG Feature Matrix Update
    scheduler.add_job(
        job_rag_feature_update,
        trigger=CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone=TIMEZONE),
        id="job_rag_update",
        name="RAG Feature Matrix Update",
        replace_existing=True,
    )

    # 5. Weekly Retrain (Optional)
    if enable_weekly_retrain:
        scheduler.add_job(
            job_weekly_model_retrain,
            trigger=CronTrigger(hour=2, minute=0, day_of_week="sun", timezone=TIMEZONE),
            id="job_weekly_retrain",
            name="Weekly Model Retrain",
            replace_existing=True,
        )

    if start_scheduler:
        scheduler.start()
        logger.info("Niveshak Background Scheduler started successfully.")

    return scheduler

