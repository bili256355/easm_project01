\
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def read_index_anomalies(path: Path, variables: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input index anomaly file not found: {path}. "
            "The lead-lag screen expects local foundation/V1 outputs to exist."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"year", "day", *variables}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    return df


def build_panel(
    df: pd.DataFrame,
    variables: List[str],
    years: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Build panel array with shape (n_years, n_days, n_variables).
    """
    idx = pd.MultiIndex.from_product([years, days], names=["year", "day"])
    sub = df.set_index(["year", "day"]).reindex(idx)
    arr = sub[variables].to_numpy(dtype=float)
    return arr.reshape(len(years), len(days), len(variables))


def make_directed_pairs(variable_families: Dict[str, str], include_same_family: bool = False) -> pd.DataFrame:
    rows = []
    variables = list(variable_families.keys())
    for src in variables:
        for tgt in variables:
            if src == tgt:
                continue
            src_family = variable_families[src]
            tgt_family = variable_families[tgt]
            if (not include_same_family) and src_family == tgt_family:
                continue
            rows.append({
                "source": src,
                "target": tgt,
                "source_family": src_family,
                "target_family": tgt_family,
            })
    return pd.DataFrame(rows)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
