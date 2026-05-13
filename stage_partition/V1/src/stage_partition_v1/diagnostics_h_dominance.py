from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .score import build_general_score_curve


def _flatten_valid(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def _rms(arr: np.ndarray) -> float:
    vals = _flatten_valid(arr)
    if vals.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(vals ** 2)))


def _mean_abs(arr: np.ndarray) -> float:
    vals = _flatten_valid(arr)
    if vals.size == 0:
        return np.nan
    return float(np.mean(np.abs(vals)))


def _p95_abs(arr: np.ndarray) -> float:
    vals = _flatten_valid(arr)
    if vals.size == 0:
        return np.nan
    return float(np.quantile(np.abs(vals), 0.95))


def _max_abs(arr: np.ndarray) -> float:
    vals = _flatten_valid(arr)
    if vals.size == 0:
        return np.nan
    return float(np.max(np.abs(vals)))


def _daily_energy_curve(block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=np.float64)
    energy = np.nanmean(arr ** 2, axis=(0, 2))
    return np.asarray(energy, dtype=np.float64)


def summarize_block_energy(block_map: Dict[str, np.ndarray], stage_label: str) -> pd.DataFrame:
    rows = []
    for object_name, block in block_map.items():
        vals = _flatten_valid(block)
        curve = _daily_energy_curve(block)
        rows.append({
            'stage_label': stage_label,
            'object_name': object_name,
            'feature_count': int(block.shape[-1]),
            'sample_count': int(vals.size),
            'rms': _rms(block),
            'mean_abs': _mean_abs(block),
            'p95_abs': _p95_abs(block),
            'max_abs': _max_abs(block),
            'daily_energy_mean': float(np.nanmean(curve)) if np.isfinite(curve).any() else np.nan,
            'daily_energy_p95': float(np.nanquantile(curve, 0.95)) if np.isfinite(curve).any() else np.nan,
            'daily_peak_day': int(np.nanargmax(curve)) if np.isfinite(curve).any() else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_block_weight_effect(standardized_blocks: Dict[str, np.ndarray], weighted_blocks: Dict[str, np.ndarray], block_weights: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for object_name in standardized_blocks:
        std_block = standardized_blocks[object_name]
        w_block = weighted_blocks[object_name]
        std_rms = _rms(std_block)
        weighted_rms = _rms(w_block)
        rows.append({
            'object_name': object_name,
            'feature_count': int(std_block.shape[-1]),
            'applied_block_weight': float(block_weights.get(object_name, np.nan)),
            'standardized_rms': std_rms,
            'weighted_rms': weighted_rms,
            'weighted_to_standardized_rms_ratio': float(weighted_rms / std_rms) if np.isfinite(std_rms) and std_rms != 0 else np.nan,
            'standardized_daily_energy_mean': float(np.nanmean(_daily_energy_curve(std_block))) if np.isfinite(_daily_energy_curve(std_block)).any() else np.nan,
            'weighted_daily_energy_mean': float(np.nanmean(_daily_energy_curve(w_block))) if np.isfinite(_daily_energy_curve(w_block)).any() else np.nan,
        })
    return pd.DataFrame(rows)


def build_block_score_contribution(state_mean: np.ndarray, block_slices: Dict[str, tuple[int, int]], candidate_df: pd.DataFrame, settings: StagePartitionV1Settings) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(columns=['window_id', 'peak_day', 'block_name', 'full_peak_score', 'isolated_peak_score', 'isolated_to_full_ratio'])

    full_curve = build_general_score_curve(state_mean, settings)
    full_lookup = full_curve.set_index('day_index')['score_smooth']
    rows = []

    for _, candidate in candidate_df.iterrows():
        peak_day = int(candidate['peak_day'])
        full_peak = float(full_lookup.get(peak_day, np.nan))
        for block_name, (start, end) in block_slices.items():
            isolated = np.zeros_like(state_mean, dtype=np.float64)
            isolated[:, start:end] = state_mean[:, start:end]
            iso_curve = build_general_score_curve(isolated, settings)
            iso_lookup = iso_curve.set_index('day_index')['score_smooth']
            iso_peak = float(iso_lookup.get(peak_day, np.nan))
            rows.append({
                'window_id': candidate['window_id'],
                'peak_day': peak_day,
                'block_name': block_name,
                'full_peak_score': full_peak,
                'isolated_peak_score': iso_peak,
                'isolated_to_full_ratio': float(iso_peak / full_peak) if np.isfinite(full_peak) and full_peak != 0 else np.nan,
            })
    return pd.DataFrame(rows)
