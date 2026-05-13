from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_index_anomalies(path: Path, variables: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input index anomaly file not found: {path}. "
            "lead_lag_screen/V2 expects local foundation/V1 smooth5 outputs to exist."
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


def make_directed_pairs(variable_families: Dict[str, str], include_same_family: bool = False) -> pd.DataFrame:
    rows = []
    variables = list(variable_families.keys())
    for source in variables:
        for target in variables:
            if source == target:
                continue
            sf = variable_families[source]
            tf = variable_families[target]
            if (not include_same_family) and sf == tf:
                continue
            rows.append({
                "source": source,
                "target": target,
                "source_family": sf,
                "target_family": tf,
            })
    return pd.DataFrame(rows)


def build_window_panel(
    df: pd.DataFrame,
    variables: List[str],
    years: np.ndarray,
    window_start: int,
    window_end: int,
    tau_max: int,
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Build a tigramite multiple-dataset panel for one target-side window.

    Days are padded backward by tau_max so that the PCMCI+ effective target times
    correspond to the formal target window as closely as possible. Pre-April days
    are retained in the day axis and masked, rather than replaced by cross-year
    values.

    Returns
    -------
    data_dict : dict[int, array]
        Dataset id -> array with shape (n_days_extended, n_variables).
    mask_dict : dict[int, bool array]
        Dataset id -> True where the value is invalid/masked.
    ext_days : np.ndarray
        Extended day labels, possibly including days <= 0 for the first window.
    """
    ext_days = np.arange(int(window_start) - int(tau_max), int(window_end) + 1, dtype=int)
    data_dict: Dict[int, np.ndarray] = {}
    mask_dict: Dict[int, np.ndarray] = {}

    base = df.set_index(["year", "day"])
    for ds_id, year in enumerate(years):
        idx = pd.MultiIndex.from_product([[int(year)], ext_days], names=["year", "day"])
        sub = base.reindex(idx)
        arr = sub[variables].to_numpy(dtype=float)
        mask = ~np.isfinite(arr)
        # Tigramite receives finite placeholders plus a mask. Do not use these
        # placeholders as data; the mask is mandatory.
        filled = np.where(mask, 0.0, arr)
        data_dict[int(ds_id)] = filled
        mask_dict[int(ds_id)] = mask

    return data_dict, mask_dict, ext_days


def read_v1_evidence(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    keep = [
        "window", "source", "target", "source_family", "target_family",
        "lead_lag_label", "lead_lag_group", "direction_label",
        "same_day_coupling_flag", "positive_peak_lag", "positive_peak_signed_r",
        "positive_peak_abs_r", "lag0_signed_r", "lag0_abs_r",
        "evidence_tier", "recommended_usage", "failure_reason", "risk_note",
        "suggested_reverse_direction", "pair_phi_risk",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()
