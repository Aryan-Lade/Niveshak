"""
Persisted Kill Switch for Emergency Trading Halts.

Guarantees:
1. Persists state in SQLite so restarts cannot clear a tripped state.
2. Hard trips require explicit manual confirmation to reset.
3. Every trip records timestamp and human-readable reason.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data_store/trading.db")


class KillSwitch:
    """
    SQLite-backed kill switch ensuring halts survive crashes and process restarts.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kill_switch_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    is_tripped INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    tripped_at TEXT,
                    reset_at TEXT
                )
            """)
            # Initialize single row if missing
            conn.execute("""
                INSERT OR IGNORE INTO kill_switch_state (id, is_tripped, reason, tripped_at, reset_at)
                VALUES (1, 0, NULL, NULL, NULL)
            """)
            conn.commit()

    def is_tripped(self) -> bool:
        """Return True if kill switch is active."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT is_tripped FROM kill_switch_state WHERE id = 1")
            row = cur.fetchone()
            return bool(row[0]) if row else False

    def trip(self, reason: str) -> None:
        """
        Trip the kill switch with an explicit reason.
        Halts all new entries immediately.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE kill_switch_state
                SET is_tripped = 1, reason = ?, tripped_at = ?, reset_at = NULL
                WHERE id = 1
                """,
                (reason, now_iso),
            )
            conn.commit()
        logger.critical(f"KILL SWITCH TRIPPED: {reason}")

    def reset(self, confirm: bool = False) -> bool:
        """
        Reset the kill switch. Requires confirm=True to prevent accidental resets.
        """
        if not confirm:
            logger.warning("Kill switch reset attempted without confirm=True. Reset denied.")
            return False

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE kill_switch_state
                SET is_tripped = 0, reset_at = ?
                WHERE id = 1
                """,
                (now_iso,),
            )
            conn.commit()
        logger.info("Kill switch successfully reset to NORMAL.")
        return True

    def status(self) -> Dict[str, Any]:
        """Get the current kill switch state."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT is_tripped, reason, tripped_at, reset_at FROM kill_switch_state WHERE id = 1")
            row = cur.fetchone()
            if not row:
                return {"is_tripped": False, "reason": None, "tripped_at": None, "reset_at": None}
            return {
                "is_tripped": bool(row[0]),
                "reason": row[1],
                "tripped_at": row[2],
                "reset_at": row[3],
            }

