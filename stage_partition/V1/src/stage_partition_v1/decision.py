
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings


def decide_general_status(candidate_df: pd.DataFrame, test_df: pd.DataFrame, block_df: pd.DataFrame, settings: StagePartitionV1Settings) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(columns=['window_id','candidate_source','general_status','failure_mode','notes'])
    if block_df.empty:
        block_summary = pd.DataFrame({'window_id': candidate_df['window_id'], 'single_block_dominance_flag': False})
    else:
        block_summary = block_df.groupby('window_id', as_index=False).agg(
            min_peak_ratio_after_drop=('peak_ratio_after_drop', 'min'),
            most_critical_block=('most_critical_block', 'first'),
        )
        block_summary['single_block_dominance_flag'] = block_summary['min_peak_ratio_after_drop'] < settings.tests.dominance_drop_ratio_threshold
    merged = candidate_df.merge(test_df, on='window_id', how='left').merge(block_summary, on='window_id', how='left')
    rows = []
    for _, row in merged.iterrows():
        struct_ok = bool(row.get('struct_test_pass', False))
        continuity_ok = bool(row.get('continuity_pass', False))
        yearwise_ok = bool(row.get('yearwise_pass', False))
        dominance = bool(row.get('single_block_dominance_flag', False))
        source = row['candidate_source']
        if dominance:
            status, failure = 'general_fail', 'single_block_dominant'
        elif source == 'base' and struct_ok and continuity_ok and yearwise_ok:
            status, failure = 'general_pass', ''
        elif source == 'edge' and struct_ok and continuity_ok and not dominance:
            status, failure = 'general_edge_pass', ''
        elif struct_ok or continuity_ok or yearwise_ok:
            status, failure = 'general_undetermined', 'evidence_not_closed'
        else:
            if not struct_ok:
                failure = 'struct_not_supported'
            elif not continuity_ok:
                failure = 'continuity_not_supported'
            else:
                failure = 'yearwise_too_sparse'
            status = 'general_fail'
        rows.append({
            'window_id': row['window_id'],
            'candidate_source': source,
            'general_status': status,
            'failure_mode': failure,
            'notes': row.get('most_critical_block', '') if dominance else '',
        })
    return pd.DataFrame(rows)
