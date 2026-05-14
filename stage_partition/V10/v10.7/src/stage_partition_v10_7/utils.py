from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clean_output_root(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for sub in ("tables", "figures", "run_meta"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def safe_nanmean(a: np.ndarray, axis=None, return_valid_count: bool = False):
    arr = np.asarray(a, dtype=float)
    valid = np.isfinite(arr)
    valid_count = valid.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.nansum(arr, axis=axis)
        mean = total / valid_count
    mean = np.where(valid_count > 0, mean, np.nan)
    if return_valid_count:
        return mean, valid_count
    return mean


def day_index_to_month_day(day_index: int) -> str:
    # Day 0 = Apr 1, for Apr-Sep season.
    month_lengths = [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]
    d = int(day_index)
    for month, length in month_lengths:
        if d < length:
            return f"{month:02d}-{d + 1:02d}"
        d -= length
    return f"out_of_range_day_{day_index}"


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return 0.0
    y = y[mask]
    x = x[mask]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
