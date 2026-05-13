
from __future__ import annotations
from pathlib import Path
import pandas as pd
from stage_partition_v6.io import load_smoothed_fields, write_dataframe, write_json
from stage_partition_v6.state_builder import build_profiles, build_state_matrix
from stage_partition_v6.detector_ruptures_window import run_point_detector
from .window_layer_config import StagePartitionV61Settings
from .window_band_builder import build_candidate_point_bands
from .window_registry import merge_candidate_bands_into_windows
from .window_uncertainty import summarize_window_uncertainty
from .window_layer_report import now_utc, build_summary


def _prepare_dirs(settings: StagePartitionV61Settings) -> dict[str, Path]:
    out = settings.output_root()
    log = settings.log_root()
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return {'output_root': out, 'log_root': log}


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing required V6 input file: {path}')
    return pd.read_csv(path)


def run_stage_partition_v6_1(settings: StagePartitionV61Settings | None = None):
    settings = settings or StagePartitionV61Settings()
    started_at = now_utc()
    roots = _prepare_dirs(settings)
    output_root = roots['output_root']
    settings.write_json(output_root / 'config_used.json')

    source_root = settings.source_v6.source_output_root()
    registry_df = _read_required_csv(source_root / 'baseline_detected_peaks_registry.csv')
    bootstrap_summary_df = _read_required_csv(source_root / 'candidate_points_bootstrap_summary.csv')
    bootstrap_records_df = _read_required_csv(source_root / 'candidate_points_bootstrap_match_records.csv')

    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, settings.profile)
    state = build_state_matrix(profiles, settings.state)
    det = run_point_detector(
        state['state_matrix'][state['valid_day_mask'], :],
        state['valid_day_index'],
        settings.detector,
        local_peak_distance_days=settings.detector.local_peak_min_distance_days,
    )

    write_dataframe(det['primary_points_df'], output_root / 'baseline_ruptures_primary_points.csv')
    write_dataframe(det['local_peaks_df'], output_root / 'baseline_detector_local_peaks_all.csv')
    write_dataframe(det['profile'].rename_axis('day').reset_index(name='profile_score'), output_root / 'baseline_detector_profile.csv')

    bands_df = build_candidate_point_bands(registry_df, det['profile'], settings.band)
    write_dataframe(bands_df, output_root / 'candidate_point_bands.csv')

    windows_df, membership_df = merge_candidate_bands_into_windows(bands_df, bootstrap_summary_df, settings.band)
    write_dataframe(windows_df, output_root / 'derived_windows_registry.csv')
    write_dataframe(membership_df, output_root / 'window_point_membership.csv')

    uncertainty_df, dist_df = summarize_window_uncertainty(
        windows_df, membership_df, bootstrap_records_df, bootstrap_summary_df, settings.uncertainty
    )
    write_dataframe(uncertainty_df, output_root / 'window_uncertainty_summary.csv')
    write_dataframe(dist_df, output_root / 'window_return_day_distribution.csv')

    summary = build_summary(windows_df, membership_df, uncertainty_df, settings)
    write_json(summary, output_root / 'summary.json')
    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V6_1',
        'run_label': settings.output.output_tag,
        'source_v6_output_tag': settings.source_v6.source_v6_output_tag,
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'V6_1 is a child version derived from V6 point-layer outputs.',
            'V6_1 consumes V6 candidate registry and bootstrap records as upstream inputs.',
            'Band construction now respects intervening candidate boundaries and can truncate at local valleys.',
            'Window merge now uses group-maximum end-day tracking instead of last-band end-day only.',
            'Protected merge is enabled for already-significant candidate peaks; close neighbors remain mergeable.',
            'Window uncertainty is attached to the main peak only and is not a second window system.',
            'Competition is not included.',
            'Parameter-path is not included.',
            'Final judgement is not included.',
            'Yearwise does not gate window construction.',
        ],
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {'output_root': output_root, 'summary': summary}
