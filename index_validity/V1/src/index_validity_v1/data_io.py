\
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .index_metadata import VARIABLE_ORDER


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_index_tables(index_values_path: Path, clim_path: Path, anom_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(p) for p in [index_values_path, clim_path, anom_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required index table(s):\n" + "\n".join(missing))

    values = pd.read_csv(index_values_path, encoding="utf-8-sig")
    clim = pd.read_csv(clim_path, encoding="utf-8-sig")
    anom = pd.read_csv(anom_path, encoding="utf-8-sig")

    required_values = {"year", "day", *VARIABLE_ORDER}
    required_clim = {"day", *VARIABLE_ORDER}
    missing_values = sorted(required_values - set(values.columns))
    missing_clim = sorted(required_clim - set(clim.columns))
    missing_anom = sorted(required_values - set(anom.columns))
    if missing_values:
        raise ValueError(f"index_values_smoothed missing columns: {missing_values}")
    if missing_clim:
        raise ValueError(f"index_daily_climatology missing columns: {missing_clim}")
    if missing_anom:
        raise ValueError(f"index_anomalies missing columns: {missing_anom}")

    for df in [values, anom]:
        df["year"] = df["year"].astype(int)
        df["day"] = df["day"].astype(int)
    clim["day"] = clim["day"].astype(int)
    return values, clim, anom


def load_smoothed_bundle(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing smoothed field bundle: {path}\n"
            "Expected foundation/V1/outputs/baseline_smooth5_a/preprocess/smoothed_fields.npz"
        )
    with np.load(path, allow_pickle=False) as data:
        arrays = {str(k): np.asarray(data[k]) for k in data.files}

    required = [
        "precip_smoothed",
        "u200_smoothed",
        "z500_smoothed",
        "v850_smoothed",
        "lat",
        "lon",
        "years",
    ]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise KeyError(f"smoothed_fields.npz missing keys: {missing}")

    return {
        "precip": np.asarray(arrays["precip_smoothed"], dtype=float),
        "u200": np.asarray(arrays["u200_smoothed"], dtype=float),
        "z500": np.asarray(arrays["z500_smoothed"], dtype=float),
        "v850": np.asarray(arrays["v850_smoothed"], dtype=float),
        "lat": np.asarray(arrays["lat"], dtype=float),
        "lon": np.asarray(arrays["lon"], dtype=float),
        "years": np.asarray(arrays["years"]),
    }


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
