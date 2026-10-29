import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_level: int = logging.INFO, log_to_file: bool = True, log_dir: str | Path = "logs") -> None:

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        file_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if log_to_file else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )