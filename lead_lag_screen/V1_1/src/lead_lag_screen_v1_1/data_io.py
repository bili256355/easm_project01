from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def read_index_anomalies(path: Path, variables: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input index anomaly file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"year", "day", *variables}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    return df


def build_panel(df: pd.DataFrame, variables: List[str], years: np.ndarray, days: np.ndarray) -> np.ndarray:
    idx = pd.MultiIndex.from_product([years, days], names=["year", "day"])
    sub = df.set_index(["year", "day"]).reindex(idx)
    arr = sub[variables].to_numpy(dtype=float)
    return arr.reshape(len(years), len(days), len(variables))


def make_directed_pairs(variable_families: Dict[str, str], include_same_family: bool = False) -> pd.DataFrame:
    """V1_1 main screen is V→P only.

    include_same_family is ignored for the main pair list because V1_1 deliberately
    does not run H/Je/Jw, P→V, or within-family tests. Reverse information is
    still assessed inside the V1-style core as a diagnostic using negative lags.
    """
    rows = []
    for src, sfam in variable_families.items():
        if sfam != "V":
            continue
        for tgt, tfam in variable_families.items():
            if tfam != "P":
                continue
            rows.append({
                "source": src,
                "target": tgt,
                "source_family": sfam,
                "target_family": tfam,
            })
    return pd.DataFrame(rows)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
