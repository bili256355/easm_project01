from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _plot_series(series: pd.Series, path: Path, title: str) -> None:
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index.to_numpy(), series.to_numpy(), linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('day')
    ax.set_ylabel('score')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_detector_profile(profile: pd.Series, output_root: Path) -> None:
    _plot_series(profile, output_root / 'ruptures_profile.png', 'V3 ruptures.Window profile')


def plot_main_windows(main_windows_df: pd.DataFrame, output_root: Path) -> None:
    if main_windows_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for _, row in main_windows_df.iterrows():
        ax.hlines(y=1, xmin=row['start_day'], xmax=row['end_day'], colors='tab:blue', linewidth=4)
        ax.scatter([row['center_day']], [1], color='black', s=12)
        ax.text(row['center_day'], 1.02, row['window_id'], fontsize=8, ha='center')
    ax.set_yticks([1])
    ax.set_yticklabels(['main windows'])
    ax.set_xlabel('day')
    ax.set_title('V3 main window overview')
    fig.tight_layout()
    fig.savefig(output_root / 'stage_partition_main_windows_overview.png', dpi=150)
    plt.close(fig)


def build_summary(main_windows_df: pd.DataFrame, evidence_df: pd.DataFrame, support_audit_df: pd.DataFrame, retention_audit_df: pd.DataFrame, support_rule_comparison_df: pd.DataFrame | None = None, yearwise_peak_df: pd.DataFrame | None = None, yearwise_support_summary_df: pd.DataFrame | None = None, point_summary: dict | None = None, point_btrack_summary: dict | None = None) -> dict:
    orphan_points = 0
    orphan_bands = 0
    if not evidence_df.empty:
        orphan_points = int(((evidence_df['evidence_object_type'] == 'point') & (evidence_df['evidence_status'] == 'orphan_point')).sum())
        orphan_bands = int(((evidence_df['evidence_object_type'] == 'band') & (evidence_df['evidence_status'] == 'orphan_band')).sum())
    limited_support = 0
    if not support_audit_df.empty and 'support_reliability_flag' in support_audit_df.columns:
        limited_support = int((support_audit_df['support_reliability_flag'] != 'high').sum())
    unresolved_retention = 0
    if not retention_audit_df.empty and 'is_window_native_consistent' in retention_audit_df.columns:
        unresolved_retention = int((retention_audit_df['is_window_native_consistent'] != 'true').sum())
    rule_changes = 0
    if support_rule_comparison_df is not None and not support_rule_comparison_df.empty and 'support_rule_change_flag' in support_rule_comparison_df.columns:
        rule_changes = int(support_rule_comparison_df['support_rule_change_flag'].sum())

    legacy_years_supporting = {}
    if yearwise_peak_df is not None and not yearwise_peak_df.empty:
        tmp = yearwise_peak_df[yearwise_peak_df['is_peak_detected_near_window'] == True]
        if not tmp.empty:
            legacy_years_supporting = tmp.groupby('window_id')['year'].nunique().astype(int).to_dict()

    yearwise_support_summary = {}
    if yearwise_support_summary_df is not None and not yearwise_support_summary_df.empty:
        for _, row in yearwise_support_summary_df.iterrows():
            yearwise_support_summary[str(row['window_id'])] = {
                'near_peak_years': int(row['n_years_near_peak']),
                'overlap_any_years': int(row['n_years_overlap_any']),
                'strict_overlap_years': int(row['n_years_strict_overlap']),
                'headline_support_mode': row['headline_support_mode'],
                'headline_support_count': int(row['headline_support_count']),
                'headline_support_fraction': float(row['headline_support_fraction']),
            }

    w004_summary = {}
    if not support_audit_df.empty and (support_audit_df['window_id'] == 'W004').any():
        row = support_audit_df[support_audit_df['window_id'] == 'W004'].iloc[0]
        w004_summary = {
            'strict_support_score': float(row['support_score']) if pd.notna(row['support_score']) else None,
            'legacy_support_score': float(row['legacy_bootstrap_match_fraction']) if pd.notna(row['legacy_bootstrap_match_fraction']) else None,
            'param_path_hit_fraction': float(row['param_path_hit_fraction']) if pd.notna(row['param_path_hit_fraction']) else None,
            'support_reliability_flag': row.get('support_reliability_flag'),
        }

    point_summary = point_summary or {}
    point_btrack_summary = point_btrack_summary or {}
    strong_vs_weak_gradient_present = bool(point_summary.get('strong_vs_weak_gradient_present_conditional', point_summary.get('strong_vs_weak_gradient_present', False)))

    return {
        'n_main_windows': int(len(main_windows_df)),
        'n_evidence_rows': int(len(evidence_df)),
        'n_orphan_points': orphan_points,
        'n_orphan_bands': orphan_bands,
        'n_limited_support_windows': limited_support,
        'n_nonclean_retention_windows': unresolved_retention,
        'n_support_rule_changes': rule_changes,
        'legacy_near_peak_years_by_window': legacy_years_supporting,
        'yearwise_support_summary': yearwise_support_summary,
        'W004_status_summary': w004_summary,
        'point_significance_summary': point_summary,
        'point_btrack_summary': point_btrack_summary,
        'point_btrack_backend_mode': point_btrack_summary.get('btrack_backend_mode'),
        'point_btrack_backend_independence': point_btrack_summary.get('btrack_backend_independence'),
        'point_btrack_formal_primary_counts': point_btrack_summary.get('formal_primary_counts', {}),
        'point_btrack_neighbor_competition_counts': point_btrack_summary.get('neighbor_competition_counts', {}),
        'point_null_scale_warning_count': int(point_summary.get('point_null_scale_warning_count', 0)),
        'n_significant_points_global': int(point_summary.get('n_significant_points_global', 0)),
        'n_significant_points_local': int(point_summary.get('n_significant_points_local', 0)),
        'neighbor_competition_closed_pairs': int(point_summary.get('neighbor_competition_closed_pairs', 0)),
        'neighbor_competition_ties': int(point_summary.get('neighbor_competition_ties', 0)),
        'strong_vs_weak_gradient_present': strong_vs_weak_gradient_present,
        'strong_vs_weak_gradient_present_conditional': bool(point_summary.get('strong_vs_weak_gradient_present_conditional', False)),
        'n_points_local_exist_frequent': int(point_summary.get('n_points_local_exist_frequent', 0)),
        'n_points_local_conditional_strong': int(point_summary.get('n_points_local_conditional_strong', 0)),
    }
