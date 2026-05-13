
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .score import build_general_score_curve


def run_block_ablation_audit(state_mean: np.ndarray, block_slices: dict, candidate_df: pd.DataFrame, settings: StagePartitionV1Settings) -> pd.DataFrame:
    full_curve = build_general_score_curve(state_mean, settings)
    rows = []
    for _, row in candidate_df.iterrows():
        peak_day = int(row['peak_day'])
        full_peak_series = full_curve.loc[full_curve['day_index'] == peak_day, 'score_smooth']
        full_peak = float(full_peak_series.iloc[0]) if not full_peak_series.empty else np.nan
        for block_name, (start, end) in block_slices.items():
            modified = np.array(state_mean, copy=True)
            modified[:, start:end] = 0.0
            drop_curve = build_general_score_curve(modified, settings)
            drop_peak_series = drop_curve.loc[drop_curve['day_index'] == peak_day, 'score_smooth']
            drop_peak = float(drop_peak_series.iloc[0]) if not drop_peak_series.empty else np.nan
            ratio = float(drop_peak / full_peak) if np.isfinite(full_peak) and full_peak != 0 else np.nan
            rows.append({
                'window_id': row['window_id'],
                'drop_block': block_name,
                'peak_ratio_after_drop': ratio,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = df.groupby('window_id', as_index=False)['peak_ratio_after_drop'].min().rename(columns={'peak_ratio_after_drop': 'min_peak_ratio_after_drop'})
    dominant = df.loc[df.groupby('window_id')['peak_ratio_after_drop'].idxmin()][['window_id', 'drop_block']].rename(columns={'drop_block': 'most_critical_block'})
    out = df.merge(summary, on='window_id', how='left').merge(dominant, on='window_id', how='left')
    return out
