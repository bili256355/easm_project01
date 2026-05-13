
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .utils import contiguous_segments, merge_segments


def _segments_to_df(segments, score_df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for idx, (start, end) in enumerate(segments, start=1):
        sub = score_df.iloc[start:end+1]
        peak_idx = int(sub['score_smooth'].idxmax())
        rows.append({
            'window_id': f'{source.upper()}_{idx:02d}',
            'start_day': int(start),
            'end_day': int(end),
            'peak_day': int(score_df.loc[peak_idx, 'day_index']),
            'width_days': int(end - start + 1),
            'peak_score_raw': float(score_df.loc[peak_idx, 'score_raw']),
            'peak_score_smooth': float(score_df.loc[peak_idx, 'score_smooth']),
            'candidate_source': source,
        })
    return pd.DataFrame(rows)


def build_general_candidates(score_df: pd.DataFrame, settings: StagePartitionV1Settings) -> pd.DataFrame:
    valid = score_df['score_smooth'].dropna()
    if valid.empty:
        return pd.DataFrame(columns=['window_id','start_day','end_day','peak_day','width_days','peak_score_raw','peak_score_smooth','candidate_source','candidate_rank'])
    base_th = float(valid.quantile(settings.candidate.base_quantile))
    edge_th = float(valid.quantile(settings.candidate.edge_quantile))
    day_values = score_df['score_smooth'].to_numpy(dtype=np.float64)
    base_mask = day_values >= base_th
    edge_mask = (day_values >= edge_th) & ~base_mask
    base_segments = [seg for seg in merge_segments(contiguous_segments(base_mask), settings.candidate.merge_gap_days) if (seg[1] - seg[0] + 1) >= settings.candidate.min_width_days]
    base_df = _segments_to_df(base_segments, score_df, 'base')
    # edge candidates: slightly shorter allowed if local peak survives
    edge_segments = [seg for seg in merge_segments(contiguous_segments(edge_mask), settings.candidate.merge_gap_days) if (seg[1] - seg[0] + 1) >= max(2, settings.candidate.min_width_days - 1)]
    edge_df = _segments_to_df(edge_segments, score_df, 'edge')
    all_df = pd.concat([base_df, edge_df], ignore_index=True)
    if all_df.empty:
        return all_df
    all_df = all_df.sort_values(['peak_score_smooth', 'width_days'], ascending=[False, False]).reset_index(drop=True)
    all_df['candidate_rank'] = np.arange(1, len(all_df) + 1, dtype=int)
    all_df['base_threshold'] = base_th
    all_df['edge_threshold'] = edge_th
    all_df['window_id'] = [f'GW{i:02d}' for i in range(1, len(all_df) + 1)]
    return all_df
