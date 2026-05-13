from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd

from .config import StagePartitionV2Settings
from .io import prepare_output_dirs, resolve_smoothed_fields_path, load_smoothed_fields, write_dataframe, write_json
from .profiles import build_profiles, summarize_profile_validity, summarize_profile_empty_slices
from .state_vector import build_state_matrix
from .backend_ruptures_window import run_ruptures_window
from .window_native_ruptures import build_primary_windows_from_ruptures, build_edge_windows_from_ruptures
from .support_audit import parameter_path_stability_ruptures, bootstrap_stability_ruptures, permutation_significance_ruptures
from .report import plot_backend_scores, plot_window_catalog


def _build_catalog(rw_primary: pd.DataFrame, rw_edge: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not rw_primary.empty:
        out = rw_primary.copy()
        out['support_class'] = 'primary_ruptures'
        frames.append(out)
    if not rw_edge.empty:
        out = rw_edge.copy()
        out['support_class'] = 'edge_ruptures'
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=['window_id', 'backend', 'window_type', 'support_class', 'start_day', 'end_day', 'center_day'])
    return pd.concat(frames, ignore_index=True)


def run_stage_partition_v2(settings: StagePartitionV2Settings | None = None) -> dict:
    settings = settings or StagePartitionV2Settings()
    t0 = datetime.utcnow().isoformat() + 'Z'
    dirs = prepare_output_dirs(settings)
    output_root = dirs['output_root']
    settings.write_json(output_root / 'config_used.json')

    smoothed_path = resolve_smoothed_fields_path(settings)
    smoothed = load_smoothed_fields(smoothed_path)

    profiles = build_profiles(smoothed, settings.profile)
    profile_validity = summarize_profile_validity(profiles)
    profile_empty_audit = summarize_profile_empty_slices(profiles)
    write_dataframe(profile_validity, output_root / 'profile_validity_summary.csv')
    write_dataframe(profile_empty_audit, output_root / 'profile_empty_slice_audit.csv')

    state = build_state_matrix(profiles, settings.state)
    np.save(output_root / 'state_matrix.npy', state['state_matrix'])
    write_json(state['state_vector_meta'], output_root / 'state_vector_meta.json')
    write_dataframe(state['feature_table'], output_root / 'feature_table.csv')
    write_dataframe(state['state_feature_scale_before_after'], output_root / 'state_feature_scale_before_after.csv')
    write_dataframe(state['state_block_energy_before_after']['raw'], output_root / 'state_block_energy_raw.csv')
    write_dataframe(state['state_block_energy_before_after']['standardized'], output_root / 'state_block_energy_standardized.csv')
    write_dataframe(state['state_block_energy_before_after']['weights'], output_root / 'state_block_weight_effect.csv')
    write_dataframe(state['state_empty_feature_audit'], output_root / 'state_empty_feature_audit.csv')
    write_dataframe(pd.DataFrame({
        'day_index': np.arange(state['state_matrix'].shape[0], dtype=int),
        'detector_valid': state['valid_day_mask'].astype(int),
    }), output_root / 'detector_valid_day_mask.csv')

    state_valid = state['state_matrix'][state['valid_day_mask'], :]
    valid_day_index = state['valid_day_index']

    rw_out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=valid_day_index)
    write_dataframe(pd.DataFrame({'breakpoint': rw_out['points']}), output_root / 'ruptures_primary_points.csv')
    write_dataframe(pd.DataFrame({'day': rw_out['profile'].index, 'profile': rw_out['profile'].to_numpy()}), output_root / 'ruptures_window_profile.csv')

    rw_primary, point_to_window_audit, band_merge_audit, peak_width_zero_audit = build_primary_windows_from_ruptures(
        rw_out['points'],
        rw_out['profile'],
        settings.edge,
        settings.ruptures_window_construction,
    )
    rw_edge = build_edge_windows_from_ruptures(rw_out['profile'], rw_primary, settings.edge, settings.ruptures_window_construction)

    support_bands = point_to_window_audit[['band_start_day', 'band_end_day', 'band_threshold']].copy() if not point_to_window_audit.empty else pd.DataFrame(columns=['band_start_day', 'band_end_day', 'band_threshold'])
    if not support_bands.empty:
        support_bands.insert(0, 'band_id', [f'RB{i:02d}' for i in range(1, len(support_bands) + 1)])

    write_dataframe(support_bands, output_root / 'ruptures_support_bands.csv')
    write_dataframe(rw_primary, output_root / 'ruptures_primary_windows.csv')
    write_dataframe(rw_edge, output_root / 'ruptures_edge_windows.csv')
    write_dataframe(point_to_window_audit, output_root / 'ruptures_point_to_window_audit.csv')
    write_dataframe(band_merge_audit, output_root / 'ruptures_band_merge_audit.csv')
    write_dataframe(peak_width_zero_audit, output_root / 'peak_width_zero_audit.csv')

    if settings.support.enable_param_path:
        rw_param = parameter_path_stability_ruptures(state['state_matrix'], settings)
        write_dataframe(rw_param, output_root / 'support_param_path_ruptures.csv')
    else:
        rw_param = pd.DataFrame()

    if settings.support.enable_bootstrap:
        rw_boot, support_invalid_audit, support_validity_summary = bootstrap_stability_ruptures(profiles, settings)
        write_dataframe(rw_boot, output_root / 'support_bootstrap_ruptures.csv')
        write_dataframe(support_invalid_audit, output_root / 'support_invalid_sample_audit.csv')
        write_dataframe(support_validity_summary, output_root / 'support_sample_validity_summary.csv')
    else:
        rw_boot = pd.DataFrame()
        support_invalid_audit = pd.DataFrame()
        support_validity_summary = pd.DataFrame()

    if settings.support.enable_permutation:
        rw_perm = permutation_significance_ruptures(state['state_matrix'], settings)
        write_dataframe(rw_perm, output_root / 'support_permutation_ruptures.csv')
    else:
        rw_perm = pd.DataFrame()

    catalog = _build_catalog(rw_primary, rw_edge)
    write_dataframe(catalog, output_root / 'window_catalog.csv')

    plot_backend_scores(pd.Series(dtype=float), rw_out['profile'], output_root)
    plot_window_catalog(catalog, output_root)

    summary = {
        'ruptures_primary_point_count': int(len(rw_out['points'])),
        'ruptures_primary_window_count': int(len(rw_primary)),
        'ruptures_edge_count': int(len(rw_edge)),
        'catalog_count': int(len(catalog)),
        'n_valid_days': int(valid_day_index.size),
        'n_invalid_days': int(state['state_matrix'].shape[0] - valid_day_index.size),
        'warning_cleaning_enabled': True,
        'n_profile_empty_slices': int(len(profile_empty_audit)),
        'n_state_all_nan_features': int(state['state_empty_feature_audit']['all_nan_flag'].sum()) if not state['state_empty_feature_audit'].empty else 0,
        'n_support_invalid_samples': int(len(support_invalid_audit)) if not support_invalid_audit.empty else 0,
    }
    write_json(summary, output_root / 'support_summary.json')

    run_meta = {
        'status': 'success',
        'started_at_utc': t0,
        'ended_at_utc': datetime.utcnow().isoformat() + 'Z',
        'layer_name': 'stage_partition',
        'version_name': 'V2',
        'foundation_dependency': 'V1',
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'Ruptures.Window is the only active main detector in this run.',
            'Primary windows are built by a window-native construction layer from breakpoint-supported profile bands.',
            'peak_width=0 is treated as a peak-shape audit signal, not as the primary determinant of window width.',
            'Multiple nearby primary points can be consolidated into a single transition window.',
            'Empty-slice warnings are converted into explicit missing-slice and invalid-sample audits.',
        ],
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {
        'output_root': output_root,
        'catalog': catalog,
        'summary': summary,
    }
