"""
Configuration loader for stock-engine.
Loads YAML configuration files and validates them using Pydantic models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError, field_validator
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()


class DataConfig(BaseModel):
    start_date: str
    holdout_start: str
    horizon_days: int
    seed: int

    @field_validator('start_date', 'holdout_start')
    @classmethod
    def date_format(cls, v: str) -> str:
        # Simple validation: expect YYYY-MM-DD
        parts = v.split('-')
        if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class CostsConfig(BaseModel):
    brokerage: float
    stamp_duty: float
    gst: float
    sebi_charges: float
    slippage: float


class RiskConfig(BaseModel):
    max_position_size: float
    max_sector_exposure: float
    max_drawdown_limit: float
    var_limit: float
    leverage_limit: float

    @field_validator('max_position_size', 'max_sector_exposure', 'max_drawdown_limit', 'var_limit', 'leverage_limit')
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class UniverseConfig(BaseModel):
    universe: list[str]
    index_macro: list[str]


class Config(BaseModel):
    data: DataConfig
    costs: CostsConfig
    risk: RiskConfig
    universe: UniverseConfig


def load_config(config_dir: str | Path = "configs") -> Config:
    """
    Load all YAML configuration files from the given directory and return a validated Config object.
    Raises:
        FileNotFoundError: If any required config file is missing.
        ValidationError: If any config values are invalid.
    """
    config_path = Path(config_dir)
    required_files = {
        "data": "data.yaml",
        "costs": "costs.yaml",
        "risk": "risk.yaml",
        "universe": "universe.yaml",
    }

    config_dict: dict[str, Any] = {}
    for key, filename in required_files.items():
        file_path = config_path / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing configuration file: {file_path}")
        try:
            with open(file_path, "rt", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            if content is None:
                content = {}
            config_dict[key] = content
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}") from e

    # Validate and parse
    try:
        return Config(**config_dict)
    except ValidationError as e:
        # Re-raise with a clear message
        raise ValidationError(f"Configuration validation failed:\n{e}") from e


# Convenience: load default config when module is imported (optional)
# config = load_config()