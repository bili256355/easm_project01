from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any
import pandas as pd

from .audit_sync_config import AuditSyncSettings


def prepare_output_dirs(settings: AuditSyncSettings) -> dict[str, Path]:
    out = settings.output_root()
    log = settings.log_root()
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return {'output_root': out, 'log_root': log}


def collect_source_result_files(settings: AuditSyncSettings) -> dict[str, Path]:
    src = settings.source_results_dir()
    return {
        'main_windows': src / settings.main_windows_filename,
        'window_catalog': src / settings.window_catalog_filename,
        'primary_points': src / settings.primary_points_filename,
        'point_to_window_audit': src / settings.point_to_window_audit_filename,
        'band_merge_audit': src / settings.band_merge_audit_filename,
        'support_bands': src / settings.support_bands_filename,
        'support_summary': src / settings.support_summary_filename,
        'support_invalid_sample_audit': src / settings.support_invalid_sample_audit_filename,
        'support_sample_validity_summary': src / settings.support_sample_validity_summary_filename,
        'support_bootstrap': src / settings.support_bootstrap_filename,
        'support_param_path': src / settings.support_param_path_filename,
        'support_permutation': src / settings.support_permutation_filename,
        'retention_audit': src / settings.retention_audit_filename,
        'run_meta': src / settings.run_meta_filename,
    }


def file_md5(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_input_manifest(file_paths: dict[str, Path], settings: AuditSyncSettings) -> dict[str, Any]:
    required_keys = ['main_windows', 'primary_points', 'support_bands']
    files: dict[str, Any] = {}
    missing_required: list[str] = []
    existing_mtimes = []
    for key, path in file_paths.items():
        exists = path.exists()
        stat = path.stat() if exists else None
        mtime = float(stat.st_mtime) if stat else None
        if mtime is not None:
            existing_mtimes.append(mtime)
        files[key] = {
            'path': str(path),
            'exists': exists,
            'required': key in required_keys,
            'size_bytes': int(stat.st_size) if stat else None,
            'modified_epoch': mtime,
            'md5': file_md5(path) if exists else None,
        }
        if key in required_keys and not exists:
            missing_required.append(key)

    time_spread_hours = 0.0
    if len(existing_mtimes) >= 2:
        time_spread_hours = (max(existing_mtimes) - min(existing_mtimes)) / 3600.0

    return {
        'source_results_dir': str(settings.source_results_dir()),
        'expected_detector': settings.expected_detector,
        'window_build_version': settings.window_build_version,
        'missing_required_files': missing_required,
        'source_files': files,
        'modified_time_spread_hours': float(time_spread_hours),
        'suspected_mixed_run_artifacts': bool(time_spread_hours > settings.mixed_run_spread_hours_threshold),
    }


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
