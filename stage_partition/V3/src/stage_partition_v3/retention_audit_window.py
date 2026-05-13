from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings


def build_window_retention_audit(main_windows_df: pd.DataFrame, evidence_df: pd.DataFrame, support_audit_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = [
        'window_id', 'retained_or_dropped', 'retention_reason_primary', 'retention_reason_secondary',
        'related_point_ids', 'related_band_ids', 'n_point_evidence_rows', 'n_band_evidence_rows',
        'support_rule_mode', 'retention_basis_support_score', 'retention_basis_bootstrap_match_fraction',
        'retention_basis_param_hit_fraction', 'n_bootstrap_effective', 'retention_rule_version',
        'retention_semantics', 'is_window_native_consistent', 'retention_change_vs_legacy', 'retention_warning'
    ]
    if main_windows_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, win in main_windows_df.iterrows():
        win_id = str(win['window_id'])
        point_rows = evidence_df[(evidence_df['window_id'] == win_id) & (evidence_df['evidence_object_type'] == 'point')] if not evidence_df.empty else pd.DataFrame()
        band_rows = evidence_df[(evidence_df['window_id'] == win_id) & (evidence_df['evidence_object_type'] == 'band')] if not evidence_df.empty else pd.DataFrame()
        point_ids = point_rows['point_id'].dropna().astype(str).tolist() if not point_rows.empty else []
        band_ids = band_rows['band_id'].dropna().astype(str).tolist() if not band_rows.empty else []
        support_row = support_audit_df[support_audit_df['window_id'] == win_id].iloc[0] if not support_audit_df.empty and (support_audit_df['window_id'] == win_id).any() else None
        support_score = float(support_row['support_score']) if support_row is not None and pd.notna(support_row['support_score']) else np.nan
        boot_frac = float(support_row['strict_bootstrap_match_fraction']) if support_row is not None and pd.notna(support_row['strict_bootstrap_match_fraction']) else np.nan
        param_frac = float(support_row['param_path_hit_fraction']) if support_row is not None and pd.notna(support_row['param_path_hit_fraction']) else np.nan
        n_boot_effective = int(support_row['n_bootstrap_effective']) if support_row is not None and pd.notna(support_row['n_bootstrap_effective']) else 0
        legacy_support = float(support_row['legacy_bootstrap_match_fraction']) if support_row is not None and pd.notna(support_row['legacy_bootstrap_match_fraction']) else np.nan
        warnings = []
        if not point_ids:
            warnings.append('missing_point_evidence')
        if not band_ids:
            warnings.append('missing_band_evidence')
        retained = 'retained_with_incomplete_evidence'
        reason_primary = 'main_window_exists_but_evidence_chain_incomplete'
        reason_secondary = 'window_object_kept_for_manual_review'
        consistent = 'false'
        change_vs_legacy = 'unknown'
        if point_ids and band_ids:
            if np.isnan(support_score):
                retained = 'retained_with_unresolved_support'
                reason_primary = 'evidence_chain_present_but_support_unresolved'
                reason_secondary = 'support_audit_requires_followup'
                consistent = 'false'
            else:
                passes = (
                    support_score >= settings.retention.min_support_score and
                    n_boot_effective >= settings.retention.min_bootstrap_effective and
                    (not np.isfinite(param_frac) or param_frac >= settings.retention.min_param_path_hit_fraction)
                )
                if passes:
                    retained = 'retained'
                    reason_primary = 'strict_support_and_evidence_chain_pass'
                    reason_secondary = 'window_native_main_window'
                    consistent = 'true'
                else:
                    retained = 'dropped_under_strict_support'
                    reason_primary = 'strict_support_below_retention_floor'
                    reason_secondary = 'requires_followup_or_rule_revision'
                    consistent = 'true'
                    warnings.append('strict_support_failed_retention')
                if np.isfinite(legacy_support):
                    if legacy_support >= settings.retention.min_support_score and support_score < settings.retention.min_support_score:
                        change_vs_legacy = 'dropped_after_strict_support'
                    elif legacy_support < settings.retention.min_support_score and support_score >= settings.retention.min_support_score:
                        change_vs_legacy = 'retained_after_strict_support'
                    else:
                        change_vs_legacy = 'same_retention_status_as_legacy'
        rows.append({
            'window_id': win_id,
            'retained_or_dropped': retained,
            'retention_reason_primary': reason_primary,
            'retention_reason_secondary': reason_secondary,
            'related_point_ids': ','.join(point_ids),
            'related_band_ids': ','.join(band_ids),
            'n_point_evidence_rows': int(len(point_rows)),
            'n_band_evidence_rows': int(len(band_rows)),
            'support_rule_mode': settings.support.match_mode,
            'retention_basis_support_score': support_score,
            'retention_basis_bootstrap_match_fraction': boot_frac,
            'retention_basis_param_hit_fraction': param_frac,
            'n_bootstrap_effective': n_boot_effective,
            'retention_rule_version': 'window_native_v3_b_strict_support',
            'retention_semantics': 'current_window_native',
            'is_window_native_consistent': consistent,
            'retention_change_vs_legacy': change_vs_legacy,
            'retention_warning': ';'.join(warnings),
        })
    return pd.DataFrame(rows, columns=cols)
