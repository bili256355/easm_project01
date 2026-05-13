from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_selection_frequency_summary(
    freq_df: pd.DataFrame,
    peak_counts_df: pd.DataFrame | None,
    maxima_df: pd.DataFrame | None,
    settings,
    *,
    plot_meta: dict | None = None,
) -> dict:
    top_days = []
    if freq_df is not None and not freq_df.empty:
        top = freq_df.sort_values('selection_frequency_raw', ascending=False).head(15)
        for _, row in top.iterrows():
            top_days.append({
                'day': int(row['day']),
                'month_day': row.get('month_day'),
                'selection_frequency_raw': float(row['selection_frequency_raw']),
                'selection_frequency_smooth': float(row['selection_frequency_smooth']) if 'selection_frequency_smooth' in row and pd.notna(row['selection_frequency_smooth']) else None,
            })
    return {
        'layer_name': 'stage_partition',
        'version_name': 'V6',
        'run_scope': 'selection_frequency_experiment_only',
        'experiment_output_tag': settings.experiment.output_tag,
        'headline_metric': 'bootstrap_selection_frequency_by_day',
        'raw_frequency_definition': 'fraction of bootstrap replicates in which the detector selected a local peak at this day',
        'smoothed_frequency_definition': f"{int(settings.experiment.smoothing_window_days)}-day centered rolling mean of raw selection frequency" if bool(settings.experiment.use_smoothing) else 'not enabled',
        'window_judgement_included': False,
        'competition_included': False,
        'parameter_path_included': False,
        'final_judgement_included': False,
        'n_bootstrap_replicates_requested': int(settings.bootstrap.n_bootstrap),
        'n_bootstrap_replicates_summarized': int(len(peak_counts_df)) if peak_counts_df is not None else 0,
        'n_local_maxima_reported': int(len(maxima_df)) if maxima_df is not None else 0,
        'curve_png_written': bool(plot_meta is not None and plot_meta.get('status') == 'written'),
        'curve_png_status': None if plot_meta is None else plot_meta.get('status'),
        'curve_png_path': None if plot_meta is None else plot_meta.get('output_path'),
        'top_frequency_days': top_days,
    }
