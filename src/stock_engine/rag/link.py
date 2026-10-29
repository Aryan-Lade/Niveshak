"""
Ticker entity linker using company names, symbols, and word-boundary regex patterns.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class TickerLinker:
    """
    Links financial text to NSE universe tickers using pre-compiled regex patterns.
    Handles tricky edge cases (e.g., 'Infosys' vs 'Info Edge').
    """

    def __init__(self, config_path: Path = Path("configs/aliases.yaml")) -> None:
        self.config_path = config_path
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load and compile regex patterns from aliases.yaml."""
        if not self.config_path.exists():
            # Try alternate path
            alt_path = Path("stock-engine") / self.config_path
            if alt_path.exists():
                self.config_path = alt_path

        if not self.config_path.exists():
            logger.warning(f"Aliases config not found at {self.config_path}, using default patterns.")
            self._init_default_patterns()
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        aliases = cfg.get("aliases", {})
        for ticker, data in aliases.items():
            patterns = data.get("patterns", [])
            compiled = []
            for pat in patterns:
                try:
                    compiled.append(re.compile(pat))
                except re.error as e:
                    logger.error(f"Failed to compile pattern '{pat}' for {ticker}: {e}")
            self.compiled_patterns[ticker] = compiled

    def _init_default_patterns(self) -> None:
        default_map = {
            "RELIANCE.NS": [r"(?i)\breliance\b", r"(?i)\bril\b"],
            "TCS.NS": [r"(?i)\btata\s+consultancy\b", r"(?i)\btcs\b"],
            "INFY.NS": [r"(?i)\binfosys(?!\s+edge)\b", r"(?i)\binfy\b"],
            "HDFCBANK.NS": [r"(?i)\bhdfc\s+bank\b", r"(?i)\bhdfc\b"],
            "ICICIBANK.NS": [r"(?i)\bicici\s+bank\b", r"(?i)\bicici\b"],
            "SBIN.NS": [r"(?i)\bstate\s+bank\s+of\s+india\b", r"(?i)\bsbi\b"],
            "ITC.NS": [r"(?i)\bitc\b"],
            "BHARTIARTL.NS": [r"(?i)\bbharti\s+airtel\b", r"(?i)\bairtel\b"],
            "LT.NS": [r"(?i)\blarsen\s+(?:&|and)\s+toubro\b", r"(?i)\bl&t\b"],
            "WIPRO.NS": [r"(?i)\bwipro\b"],
        }
        for ticker, pats in default_map.items():
            self.compiled_patterns[ticker] = [re.compile(p) for p in pats]

    def link_text(self, text: str) -> List[str]:
        """
        Scan text for all matching universe tickers.
        An article may match multiple tickers.
        """
        matched_tickers: Set[str] = set()

        # Guard against False positives for INFY on 'Info Edge'
        is_info_edge = bool(re.search(r"(?i)\binfo\s*edge\b", text))

        for ticker, patterns in self.compiled_patterns.items():
            for pat in patterns:
                if pat.search(text):
                    if ticker == "INFY.NS" and is_info_edge and not re.search(r"(?i)\binfosys\b", text):
                        # Skip false positive match
                        continue
                    matched_tickers.add(ticker)
                    break

        return sorted(list(matched_tickers))
