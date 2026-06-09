"""Platform configuration: frozen dataclasses loaded from JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    inputs_folder: str = "Inputs"
    classification_file: str = "Classification.xlsx"
    target_date: str = "2010-01-01"
    columns_to_drop: tuple = ("Saudi Arabia",)
    ffill_limit: int = 63
    publication_lag_days: dict = field(default_factory=dict)  # metric -> days


@dataclass(frozen=True)
class BacktestConfig:
    selection_criteria: str = "relative"       # 'absolute' | 'relative'
    absolute_selection_score: float = 0.75
    relative_selection_score: int = 5
    weighting_method: str = "Equal"            # 'Equal' | 'Risk_Parity'
    risk_parity_lookback: int = 120
    bmk: str = "World"
    bmk_weight: float = 0.5
    mode: str = "blend"                        # 'blend' | 'active'
    periodicity: int = 63
    transaction_cost_bps: float = 2.0
    execution_lag_days: int = 0


@dataclass(frozen=True)
class PlatformConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def load_config(path: str | Path) -> PlatformConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = raw.get("data", {})
    if "columns_to_drop" in data:
        data["columns_to_drop"] = tuple(data["columns_to_drop"])
    return PlatformConfig(
        data=DataConfig(**data),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )
