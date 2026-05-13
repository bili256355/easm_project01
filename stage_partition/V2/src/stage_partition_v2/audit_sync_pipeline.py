from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from .audit_sync_config import AuditSyncSettings
from .audit_sync_io import (
    prepare_output_dirs,
    collect_source_result_files,
    build_input_manifest,
    read_csv_if_exists,
    read_json_if_exists,
    write_dataframe,
    write_json,
)


def _now_utc() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def _to_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _split_int_tokens(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, float) and not math.isnan(value):
        return [int(value)]
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    out: list[int] = []
    for part in parts:
        try:
            out.append(int(float(part)))
        except Exception:
            continue
    return out


def _format_id(prefix: str, idx: int) -> str:
    return f'{prefix}{idx:03d}'


def _safe_series_first(df: pd.DataFrame, col: str, default: Any = None) -> Any:
    if df.empty or col not in df.columns:
        return default
    return df.iloc[0][col]


def _select_backend_row(df: pd.DataFrame, backend: str) -> pd.DataFrame:
    if df.empty or 'backend' not in df.columns:
        return df.iloc[0:0].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    sub = df[df['backend'] == backend].copy()
    return sub if not sub.empty else df.iloc[0:0].copy()


def _load_source_tables(file_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        'main_windows': read_csv_if_exists(file_paths['main_windows']),
        'window_catalog': read_csv_if_exists(file_paths['window_catalog']),
        'primary_points': read_csv_if_exists(file_paths['primary_points']),
        'point_to_window_audit': read_csv_if_exists(file_paths['point_to_window_audit']),
        'band_merge_audit': read_csv_if_exists(file_paths['band_merge_audit']),
        'support_bands': read_csv_if_exists(file_paths['support_bands']),
        'support_summary': read_json_if_exists(file_paths['support_summary']),
        'support_invalid_sample_audit': read_csv_if_exists(file_paths['support_invalid_sample_audit']),
        'support_sample_validity_summary': read_csv_if_exists(file_paths['support_sample_validity_summary']),
        'support_bootstrap': read_csv_if_exists(file_paths['support_bootstrap']),
        'support_param_path': read_csv_if_exists(file_paths['support_param_path']),
        'support_permutation': read_csv_if_exists(file_paths['support_permutation']),
        'retention_audit': read_csv_if_exists(file_paths['retention_audit']),
        'run_meta': read_json_if_exists(file_paths['run_meta']),
    }


def _build_point_table(primary_points_df: pd.DataFrame, point_to_window_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not point_to_window_df.empty:
        ordered = point_to_window_df.copy()
        if 'peak_day' in ordered.columns:
            ordered = ordered.sort_values(['peak_day']).reset_index(drop=True)
        for i, row in ordered.iterrows():
            rows.append({
                'point_id': _format_id('RP', i + 1),
                'point_day': _to_int(row.get('peak_day')),
                'peak_value': row.get('peak_value'),
                'peak_prominence': row.get('peak_prominence'),
                'assigned_window_id_legacy': row.get('assigned_window_id'),
                'source': 'ruptures_point_to_window_audit',
            })
    elif not primary_points_df.empty:
        col = primary_points_df.columns[0]
        ordered = primary_points_df.copy().sort_values(col).reset_index(drop=True)
        for i, row in ordered.iterrows():
            rows.append({
                'point_id': _format_id('RP', i + 1),
                'point_day': _to_int(row.get(col)),
                'peak_value': np.nan,
                'peak_prominence': np.nan,
                'assigned_window_id_legacy': None,
                'source': 'ruptures_primary_points',
            })
    return pd.DataFrame(rows)


def _build_band_table(support_bands_df: pd.DataFrame) -> pd.DataFrame:
    if support_bands_df.empty:
        return pd.DataFrame(columns=['band_id', 'band_start_day', 'band_end_day', 'band_threshold'])
    out = support_bands_df.copy()
    if 'band_id' not in out.columns:
        out.insert(0, 'band_id', [_format_id('RB', i + 1) for i in range(len(out))])
    return out


def build_main_window_table(
    primary_windows_df: pd.DataFrame,
    window_catalog_df: pd.DataFrame,
    point_df: pd.DataFrame,
    band_df: pd.DataFrame,
    settings: AuditSyncSettings,
) -> pd.DataFrame:
    cols = [
        'window_id', 'legacy_window_id', 'start_day', 'end_day', 'center_day', 'width_days',
        'main_peak_day', 'origin_detector', 'window_build_version',
        'source_primary_point_ids', 'source_support_band_ids',
        'is_main_window', 'window_status', 'status_reason', 'catalog_match_status', 'notes'
    ]
    if primary_windows_df.empty:
        return pd.DataFrame(columns=cols)

    out = primary_windows_df.copy()
    sort_cols = [c for c in ['center_day', 'start_day', 'end_day'] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    catalog = window_catalog_df.copy() if not window_catalog_df.empty else pd.DataFrame()
    catalog_primary = catalog[catalog['window_type'] == 'primary'].copy() if ('window_type' in catalog.columns) else catalog.iloc[0:0].copy()

    rows: list[dict[str, Any]] = []
    for i, row in out.iterrows():
        start_day = _to_int(row.get('start_day'))
        end_day = _to_int(row.get('end_day'))
        center_day = _to_int(row.get('center_day'))
        main_peak_day = _to_int(row.get('main_peak_day'))
        if start_day is None or end_day is None:
            width_days = None
        else:
            width_days = int(end_day - start_day + 1)

        points_in_window = []
        if not point_df.empty and start_day is not None and end_day is not None:
            mask = point_df['point_day'].between(start_day, end_day, inclusive='both')
            points_in_window = point_df.loc[mask, 'point_id'].tolist()

        bands_in_window = []
        if not band_df.empty and start_day is not None and end_day is not None:
            overlap_mask = (band_df['band_start_day'] <= end_day) & (band_df['band_end_day'] >= start_day)
            bands_in_window = band_df.loc[overlap_mask, 'band_id'].tolist()

        catalog_match_status = 'catalog_missing'
        if not catalog_primary.empty and center_day is not None:
            cands = catalog_primary[
                (catalog_primary['start_day'] == start_day) &
                (catalog_primary['end_day'] == end_day) &
                (catalog_primary['center_day'] == center_day)
            ]
            catalog_match_status = 'exact_match' if not cands.empty else 'mismatch'

        rows.append({
            'window_id': _format_id('W', i + 1),
            'legacy_window_id': row.get('window_id'),
            'start_day': start_day,
            'end_day': end_day,
            'center_day': center_day,
            'width_days': width_days,
            'main_peak_day': main_peak_day,
            'origin_detector': row.get('backend', settings.expected_detector),
            'window_build_version': settings.window_build_version,
            'source_primary_point_ids': ','.join(points_in_window),
            'source_support_band_ids': ','.join(bands_in_window),
            'is_main_window': True,
            'window_status': 'main',
            'status_reason': 'from_ruptures_primary_windows',
            'catalog_match_status': catalog_match_status,
            'notes': 'catalog is advisory only; primary_windows is authoritative main-object source',
        })
    return pd.DataFrame(rows, columns=cols)


def build_window_evidence_mapping(
    main_windows_df: pd.DataFrame,
    point_df: pd.DataFrame,
    band_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        'window_id', 'legacy_window_id', 'evidence_object_type', 'evidence_object_id',
        'point_id', 'point_day', 'band_id', 'band_start_day', 'band_end_day',
        'band_window_overlap_days', 'band_window_overlap_ratio',
        'used_in_window_consolidation', 'evidence_status', 'warning_flag'
    ]
    rows: list[dict[str, Any]] = []

    if main_windows_df.empty:
        return pd.DataFrame(columns=cols)

    for _, win in main_windows_df.iterrows():
        win_id = str(win['window_id'])
        legacy_window_id = win.get('legacy_window_id')
        start_day = int(win['start_day'])
        end_day = int(win['end_day'])
        win_width = max(1, int(win['width_days']))

        if not point_df.empty:
            for _, p in point_df.iterrows():
                point_day = _to_int(p.get('point_day'))
                if point_day is None:
                    continue
                in_window = start_day <= point_day <= end_day
                if in_window:
                    legacy_match = (p.get('assigned_window_id_legacy') == legacy_window_id)
                    rows.append({
                        'window_id': win_id,
                        'legacy_window_id': legacy_window_id,
                        'evidence_object_type': 'point',
                        'evidence_object_id': p.get('point_id'),
                        'point_id': p.get('point_id'),
                        'point_day': point_day,
                        'band_id': None,
                        'band_start_day': None,
                        'band_end_day': None,
                        'band_window_overlap_days': None,
                        'band_window_overlap_ratio': None,
                        'used_in_window_consolidation': bool(legacy_match or True),
                        'evidence_status': 'mapped_to_main_window',
                        'warning_flag': '' if legacy_match or pd.isna(p.get('assigned_window_id_legacy')) else 'legacy_window_id_mismatch',
                    })

        if not band_df.empty:
            for _, b in band_df.iterrows():
                band_start = _to_int(b.get('band_start_day'))
                band_end = _to_int(b.get('band_end_day'))
                if band_start is None or band_end is None:
                    continue
                overlap_days = max(0, min(end_day, band_end) - max(start_day, band_start) + 1)
                if overlap_days <= 0:
                    continue
                overlap_ratio = overlap_days / float(max(1, band_end - band_start + 1))
                rows.append({
                    'window_id': win_id,
                    'legacy_window_id': legacy_window_id,
                    'evidence_object_type': 'band',
                    'evidence_object_id': b.get('band_id'),
                    'point_id': None,
                    'point_day': None,
                    'band_id': b.get('band_id'),
                    'band_start_day': band_start,
                    'band_end_day': band_end,
                    'band_window_overlap_days': int(overlap_days),
                    'band_window_overlap_ratio': float(overlap_ratio),
                    'used_in_window_consolidation': bool(overlap_ratio > 0.0),
                    'evidence_status': 'mapped_to_main_window',
                    'warning_flag': '' if overlap_ratio >= 0.5 else 'partial_overlap_only',
                })

    if not point_df.empty:
        mapped_point_ids = {row['point_id'] for row in rows if row['evidence_object_type'] == 'point'}
        for _, p in point_df.iterrows():
            if p.get('point_id') in mapped_point_ids:
                continue
            rows.append({
                'window_id': None,
                'legacy_window_id': None,
                'evidence_object_type': 'point',
                'evidence_object_id': p.get('point_id'),
                'point_id': p.get('point_id'),
                'point_day': _to_int(p.get('point_day')),
                'band_id': None,
                'band_start_day': None,
                'band_end_day': None,
                'band_window_overlap_days': None,
                'band_window_overlap_ratio': None,
                'used_in_window_consolidation': False,
                'evidence_status': 'orphan_point',
                'warning_flag': 'point_not_mapped_to_any_main_window',
            })

    if not band_df.empty:
        mapped_band_ids = {row['band_id'] for row in rows if row['evidence_object_type'] == 'band'}
        for _, b in band_df.iterrows():
            if b.get('band_id') in mapped_band_ids:
                continue
            rows.append({
                'window_id': None,
                'legacy_window_id': None,
                'evidence_object_type': 'band',
                'evidence_object_id': b.get('band_id'),
                'point_id': None,
                'point_day': None,
                'band_id': b.get('band_id'),
                'band_start_day': _to_int(b.get('band_start_day')),
                'band_end_day': _to_int(b.get('band_end_day')),
                'band_window_overlap_days': 0,
                'band_window_overlap_ratio': 0.0,
                'used_in_window_consolidation': False,
                'evidence_status': 'orphan_band',
                'warning_flag': 'band_not_mapped_to_any_main_window',
            })

    out = pd.DataFrame(rows, columns=cols)
    return out


def recompute_window_support_audit(
    main_windows_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    support_summary: dict[str, Any],
    support_invalid_df: pd.DataFrame,
    support_validity_df: pd.DataFrame,
    support_bootstrap_df: pd.DataFrame,
    support_param_df: pd.DataFrame,
    support_perm_df: pd.DataFrame,
    settings: AuditSyncSettings,
) -> pd.DataFrame:
    cols = [
        'window_id',
        'n_total_samples',
        'n_valid_samples',
        'n_invalid_samples',
        'invalid_ratio',
        'sample_scope',
        'n_bootstrap_requested',
        'n_bootstrap_effective',
        'support_score',
        'support_score_source',
        'param_path_hits',
        'permutation_empirical_pvalue_global',
        'support_status',
        'support_reliability_flag',
        'support_warning',
    ]
    if main_windows_df.empty:
        return pd.DataFrame(columns=cols)

    validity_backend = _select_backend_row(support_validity_df, settings.expected_detector)
    validity_row = validity_backend.iloc[0] if not validity_backend.empty else None
    n_total_samples = int(validity_row['n_total_samples']) if validity_row is not None and 'n_total_samples' in validity_backend.columns else None
    n_valid_samples = int(validity_row['n_valid_samples']) if validity_row is not None and 'n_valid_samples' in validity_backend.columns else None
    n_invalid_samples = int(validity_row['n_invalid_samples']) if validity_row is not None and 'n_invalid_samples' in validity_backend.columns else None
    invalid_ratio = float(validity_row['invalid_fraction']) if validity_row is not None and 'invalid_fraction' in validity_backend.columns else None

    if n_total_samples is None and support_summary:
        n_total_samples = support_summary.get('n_support_invalid_samples')
    support_invalid_backend = _select_backend_row(support_invalid_df, settings.expected_detector)
    bootstrap_reps_requested = None
    if not support_bootstrap_df.empty and 'rep' in support_bootstrap_df.columns:
        bootstrap_reps_requested = int(support_bootstrap_df['rep'].max()) + 1
    elif n_total_samples is not None:
        bootstrap_reps_requested = n_total_samples

    permutation_backend = _select_backend_row(support_perm_df, settings.expected_detector)
    global_perm = None
    if not permutation_backend.empty and 'empirical_pvalue' in permutation_backend.columns:
        global_perm = float(permutation_backend.iloc[0]['empirical_pvalue'])

    rows: list[dict[str, Any]] = []
    for _, win in main_windows_df.iterrows():
        window_id = str(win['window_id'])
        start_day = int(win['start_day'])
        end_day = int(win['end_day'])
        center_day = int(win['center_day'])

        if not support_bootstrap_df.empty:
            boot = support_bootstrap_df.copy()
            overlap = (boot['start_day'] <= end_day) & (boot['end_day'] >= start_day)
            boot_hits = boot.loc[overlap].copy()
            n_bootstrap_effective = int(boot_hits['rep'].nunique()) if 'rep' in boot_hits.columns else len(boot_hits)
        else:
            n_bootstrap_effective = None

        if bootstrap_reps_requested and bootstrap_reps_requested > 0 and n_bootstrap_effective is not None:
            support_score = float(n_bootstrap_effective) / float(bootstrap_reps_requested)
            support_score_source = 'bootstrap_window_overlap_fraction'
        else:
            support_score = np.nan
            support_score_source = 'unresolved_no_bootstrap_table'

        if not support_param_df.empty:
            param_hits = support_param_df[
                (support_param_df['start_day'] <= end_day) &
                (support_param_df['end_day'] >= start_day)
            ]
            param_path_hits = int(len(param_hits))
        else:
            param_path_hits = 0

        warnings = []
        if bootstrap_reps_requested is None:
            warnings.append('missing_bootstrap_requested')
        if support_bootstrap_df.empty:
            warnings.append('missing_support_bootstrap_table')
        if validity_backend.empty:
            warnings.append('missing_support_sample_validity_summary')
        if support_invalid_backend.empty:
            warnings.append('missing_support_invalid_sample_audit')
        if np.isnan(support_score):
            support_status = 'unresolved_mapping'
            reliability = 'low'
        else:
            support_status = 'resolved_window_level'
            if invalid_ratio is not None and invalid_ratio <= 0.05 and support_score >= 0.50 and (bootstrap_reps_requested or 0) >= 20:
                reliability = 'high'
            elif support_score >= 0.20:
                reliability = 'limited'
            else:
                reliability = 'low'

        rows.append({
            'window_id': window_id,
            'n_total_samples': n_total_samples,
            'n_valid_samples': n_valid_samples,
            'n_invalid_samples': n_invalid_samples,
            'invalid_ratio': invalid_ratio,
            'sample_scope': 'backend_global_summary',
            'n_bootstrap_requested': bootstrap_reps_requested,
            'n_bootstrap_effective': n_bootstrap_effective,
            'support_score': support_score,
            'support_score_source': support_score_source,
            'param_path_hits': param_path_hits,
            'permutation_empirical_pvalue_global': global_perm,
            'support_status': support_status,
            'support_reliability_flag': reliability,
            'support_warning': ';'.join(warnings),
        })
    return pd.DataFrame(rows, columns=cols)


def recompute_window_retention_audit(
    main_windows_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    retention_df: pd.DataFrame,
    settings: AuditSyncSettings,
) -> pd.DataFrame:
    cols = [
        'window_id',
        'retained_or_dropped',
        'retention_reason_primary',
        'retention_reason_secondary',
        'related_point_ids',
        'related_band_ids',
        'n_mapped_retention_rows',
        'n_primary_rows',
        'n_edge_rows',
        'n_dropped_rows',
        'retention_rule_version',
        'retention_semantics',
        'is_window_native_consistent',
        'retention_warning',
    ]
    if main_windows_df.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for _, win in main_windows_df.iterrows():
        window_id = str(win['window_id'])
        start_day = int(win['start_day'])
        end_day = int(win['end_day'])

        if not retention_df.empty and 'peak_day' in retention_df.columns:
            mapped = retention_df[(retention_df['peak_day'] >= start_day) & (retention_df['peak_day'] <= end_day)].copy()
        else:
            mapped = pd.DataFrame()

        related_point_ids = []
        related_band_ids = []
        if not evidence_df.empty:
            related_point_ids = evidence_df[
                (evidence_df['window_id'] == window_id) &
                (evidence_df['evidence_object_type'] == 'point')
            ]['point_id'].dropna().astype(str).tolist()
            related_band_ids = evidence_df[
                (evidence_df['window_id'] == window_id) &
                (evidence_df['evidence_object_type'] == 'band')
            ]['band_id'].dropna().astype(str).tolist()

        warnings = []
        if mapped.empty:
            retained_or_dropped = 'unknown'
            primary_reason = 'no_matching_legacy_retention_rows'
            secondary_reason = 'retention_table_not_window_native'
            is_consistent = 'unknown'
            n_primary_rows = 0
            n_edge_rows = 0
            n_dropped_rows = 0
        else:
            retained_values = mapped['retained_as'].fillna('dropped').astype(str).tolist() if 'retained_as' in mapped.columns else ['unknown'] * len(mapped)
            n_primary_rows = int(sum(v == 'primary' for v in retained_values))
            n_edge_rows = int(sum(v == 'edge' for v in retained_values))
            n_dropped_rows = int(sum(v == 'dropped' for v in retained_values))
            if n_primary_rows > 0:
                retained_or_dropped = 'retained'
                primary_reason = 'has_legacy_primary_peak'
            elif n_edge_rows > 0:
                retained_or_dropped = 'retained_with_semantic_conflict'
                primary_reason = 'legacy_rows_only_edge_for_current_main_window'
                warnings.append('legacy_retention_marks_main_window_as_edge')
            else:
                retained_or_dropped = 'dropped'
                primary_reason = 'legacy_rows_not_retained'
            secondary_reason = 'aggregated_from_legacy_peak_rows'
            is_consistent = 'true' if (n_primary_rows > 0 and n_edge_rows == 0 and n_dropped_rows == 0) else 'false'

        if not related_point_ids:
            warnings.append('no_point_evidence_ids_found')
        rows.append({
            'window_id': window_id,
            'retained_or_dropped': retained_or_dropped,
            'retention_reason_primary': primary_reason,
            'retention_reason_secondary': secondary_reason,
            'related_point_ids': ','.join(related_point_ids),
            'related_band_ids': ','.join(related_band_ids),
            'n_mapped_retention_rows': int(len(mapped)),
            'n_primary_rows': n_primary_rows,
            'n_edge_rows': n_edge_rows,
            'n_dropped_rows': n_dropped_rows,
            'retention_rule_version': 'audit_sync_window_native_v1',
            'retention_semantics': 'window_native_aggregated_from_legacy_peak_rows',
            'is_window_native_consistent': is_consistent,
            'retention_warning': ';'.join(warnings),
        })
    return pd.DataFrame(rows, columns=cols)


def build_audit_trust_tiers(
    main_windows_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    support_audit_df: pd.DataFrame,
    retention_audit_df: pd.DataFrame,
    settings: AuditSyncSettings,
) -> pd.DataFrame:
    rows = [
        {
            'artifact_name': settings.main_window_table_filename,
            'artifact_type': 'main_window_table',
            'scope': 'window_object',
            'trust_tier': 'A_main_basis',
            'can_be_used_as_main_basis': True,
            'must_be_qualified_when_cited': False,
            'cannot_be_used_for': 'window_internal_order_or_pathway',
            'notes': '唯一主窗口对象主表',
        },
        {
            'artifact_name': settings.window_evidence_mapping_filename,
            'artifact_type': 'evidence_mapping',
            'scope': 'point_and_band_to_window',
            'trust_tier': 'B_supporting_evidence',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'replace_main_window_table',
            'notes': '点和band只作窗口支撑证据',
        },
        {
            'artifact_name': settings.window_support_audit_filename,
            'artifact_type': 'support_audit',
            'scope': 'window_level_audit',
            'trust_tier': 'B_supporting_evidence',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'direct_physical_interpretation',
            'notes': 'support分数需带样本口径一起引用',
        },
        {
            'artifact_name': settings.window_retention_audit_filename,
            'artifact_type': 'retention_audit',
            'scope': 'window_level_audit',
            'trust_tier': 'C_restricted_audit_only',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'defining_main_windows',
            'notes': '仍是从旧peak语义聚合过来的窗口级解释',
        },
        {
            'artifact_name': settings.support_summary_filename,
            'artifact_type': 'legacy_summary',
            'scope': 'legacy_global_summary',
            'trust_tier': 'D_legacy_do_not_use',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'main_window_level_support',
            'notes': '旧global summary不再作为逐窗主依据',
        },
        {
            'artifact_name': settings.retention_audit_filename,
            'artifact_type': 'legacy_retention_audit',
            'scope': 'legacy_peak_level',
            'trust_tier': 'D_legacy_do_not_use',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'window_native_retention_explanation',
            'notes': '旧retention表按peak语义生成，不能直接当窗口解释',
        },
    ]
    return pd.DataFrame(rows)


def build_audit_sync_summary(
    manifest: dict[str, Any],
    main_windows_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    support_audit_df: pd.DataFrame,
    retention_audit_df: pd.DataFrame,
) -> dict[str, Any]:
    orphan_points = 0
    orphan_bands = 0
    if not evidence_df.empty:
        orphan_points = int(((evidence_df['evidence_object_type'] == 'point') & (evidence_df['evidence_status'] == 'orphan_point')).sum())
        orphan_bands = int(((evidence_df['evidence_object_type'] == 'band') & (evidence_df['evidence_status'] == 'orphan_band')).sum())

    unresolved_support = 0
    if not support_audit_df.empty and 'support_status' in support_audit_df.columns:
        unresolved_support = int((support_audit_df['support_status'] != 'resolved_window_level').sum())

    unresolved_retention = 0
    if not retention_audit_df.empty and 'is_window_native_consistent' in retention_audit_df.columns:
        unresolved_retention = int((retention_audit_df['is_window_native_consistent'] != 'true').sum())

    return {
        'n_main_windows': int(len(main_windows_df)),
        'n_evidence_rows': int(len(evidence_df)),
        'n_orphan_points': orphan_points,
        'n_orphan_bands': orphan_bands,
        'n_unresolved_support_windows': unresolved_support,
        'n_nonclean_retention_windows': unresolved_retention,
        'suspected_mixed_run_artifacts': bool(manifest.get('suspected_mixed_run_artifacts', False)),
    }


def run_stage_partition_v2_audit_sync(settings: AuditSyncSettings | None = None) -> dict[str, Any]:
    settings = settings or AuditSyncSettings()
    started_at = _now_utc()
    dirs = prepare_output_dirs(settings)
    output_root = dirs['output_root']
    settings.write_json(output_root / 'config_used.json')

    file_paths = collect_source_result_files(settings)
    manifest = build_input_manifest(file_paths, settings)
    if manifest['missing_required_files']:
        raise FileNotFoundError(f'Missing required source files: {manifest["missing_required_files"]}')
    write_json(manifest, output_root / settings.freeze_manifest_filename)

    tables = _load_source_tables(file_paths)
    point_df = _build_point_table(tables['primary_points'], tables['point_to_window_audit'])
    band_df = _build_band_table(tables['support_bands'])
    main_windows_df = build_main_window_table(
        tables['main_windows'],
        tables['window_catalog'],
        point_df,
        band_df,
        settings,
    )
    evidence_df = build_window_evidence_mapping(main_windows_df, point_df, band_df)
    support_audit_df = recompute_window_support_audit(
        main_windows_df,
        evidence_df,
        tables['support_summary'],
        tables['support_invalid_sample_audit'],
        tables['support_sample_validity_summary'],
        tables['support_bootstrap'],
        tables['support_param_path'],
        tables['support_permutation'],
        settings,
    )
    retention_audit_df = recompute_window_retention_audit(
        main_windows_df,
        evidence_df,
        tables['retention_audit'],
        settings,
    )
    trust_tiers_df = build_audit_trust_tiers(
        main_windows_df,
        evidence_df,
        support_audit_df,
        retention_audit_df,
        settings,
    )
    audit_summary = build_audit_sync_summary(
        manifest,
        main_windows_df,
        evidence_df,
        support_audit_df,
        retention_audit_df,
    )

    write_dataframe(main_windows_df, output_root / settings.main_window_table_filename)
    write_dataframe(evidence_df, output_root / settings.window_evidence_mapping_filename)
    write_dataframe(support_audit_df, output_root / settings.window_support_audit_filename)
    write_dataframe(retention_audit_df, output_root / settings.window_retention_audit_filename)
    write_dataframe(trust_tiers_df, output_root / settings.trust_tiers_filename)
    write_json(audit_summary, output_root / settings.audit_sync_summary_filename)

    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': _now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V2',
        'audit_name': 'audit_sync_v1_a',
        'source_results_dir': str(settings.source_results_dir()),
        'output_root': str(output_root),
        'notes': [
            'This run does not rerun the detector; it audits and resynchronizes existing V2 result tables.',
            'stage_partition_main_windows.csv is the only main window-object table produced by this audit.',
            'window_evidence_mapping.csv separates point/band evidence from main windows.',
            'window_support_audit.csv and window_retention_audit.csv are restricted audit layers, not replacement main results.',
        ],
    }
    write_json(run_meta, output_root / settings.run_meta_out_filename)
    return {
        'output_root': output_root,
        'summary': audit_summary,
    }
