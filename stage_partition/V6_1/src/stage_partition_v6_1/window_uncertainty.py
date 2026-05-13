from __future__ import annotations
import numpy as np
import pandas as pd


def summarize_window_uncertainty(windows_df: pd.DataFrame, membership_df: pd.DataFrame, bootstrap_records_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame | None, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    scols = [
        'window_id','main_peak_day','main_peak_candidate_id','main_peak_bootstrap_match_fraction',
        'n_returned_replicates_for_interval','return_day_median','return_day_q02_5','return_day_q97_5',
        'return_day_p10','return_day_p90','return_day_iqr','return_day_width95','return_day_width80'
    ]
    dcols = ['window_id','candidate_id','replicate_id','matched_peak_day','match_type']
    if windows_df is None or windows_df.empty or membership_df is None or membership_df.empty or bootstrap_records_df is None or bootstrap_records_df.empty:
        return pd.DataFrame(columns=scols), pd.DataFrame(columns=dcols)
    allowed = set(cfg.interval_match_types)
    boot = {}
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty:
        boot = bootstrap_summary_df.set_index('candidate_id')['bootstrap_match_fraction'].to_dict()
    rows = []
    detail_rows = []
    for _, win in windows_df.iterrows():
        wid = str(win['window_id'])
        main = membership_df[(membership_df['window_id'] == wid) & (membership_df['is_main_peak'])]
        if main.empty:
            continue
        main = main.iloc[0]
        cid = str(main['candidate_id'])
        sub = bootstrap_records_df[(bootstrap_records_df['candidate_id'] == cid) & (bootstrap_records_df['match_type'].isin(allowed))].copy()
        for _, rec in sub.iterrows():
            detail_rows.append({
                'window_id': wid,
                'candidate_id': cid,
                'replicate_id': int(rec['replicate_id']),
                'matched_peak_day': int(rec['matched_peak_day']) if pd.notna(rec['matched_peak_day']) else np.nan,
                'match_type': str(rec['match_type']),
            })
        vals = pd.to_numeric(sub['matched_peak_day'], errors='coerce').dropna().astype(float)
        if vals.empty:
            rows.append({
                'window_id': wid,
                'main_peak_day': int(win['main_peak_day']),
                'main_peak_candidate_id': cid,
                'main_peak_bootstrap_match_fraction': float(boot.get(cid, np.nan)),
                'n_returned_replicates_for_interval': 0,
                'return_day_median': np.nan,
                'return_day_q02_5': np.nan,
                'return_day_q97_5': np.nan,
                'return_day_p10': np.nan,
                'return_day_p90': np.nan,
                'return_day_iqr': np.nan,
                'return_day_width95': np.nan,
                'return_day_width80': np.nan,
            })
            continue
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        p10, p90 = np.quantile(vals, [0.10, 0.90])
        q025, q975 = np.quantile(vals, [0.025, 0.975])
        rows.append({
            'window_id': wid,
            'main_peak_day': int(win['main_peak_day']),
            'main_peak_candidate_id': cid,
            'main_peak_bootstrap_match_fraction': float(boot.get(cid, np.nan)),
            'n_returned_replicates_for_interval': int(vals.size),
            'return_day_median': float(np.median(vals)),
            'return_day_q02_5': float(q025),
            'return_day_q97_5': float(q975),
            'return_day_p10': float(p10),
            'return_day_p90': float(p90),
            'return_day_iqr': float(q75 - q25),
            'return_day_width95': float(q975 - q025),
            'return_day_width80': float(p90 - p10),
        })
    return pd.DataFrame(rows, columns=scols), pd.DataFrame(detail_rows, columns=dcols)
