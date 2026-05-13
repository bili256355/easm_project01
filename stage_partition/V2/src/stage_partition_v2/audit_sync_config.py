from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import os


def _default_project_root() -> Path:
    return Path(os.environ.get('EASM_PROJECT01_ROOT', r'D:\easm_project01'))


@dataclass
class AuditSyncSettings:
    project_root: Path = field(default_factory=_default_project_root)
    source_output_tag: str = 'baseline_a'
    audit_output_tag: str = 'audit_sync_v1_a'
    expected_detector: str = 'ruptures_window'
    window_build_version: str = 'window_native_v1'
    mixed_run_spread_hours_threshold: float = 1.0

    main_windows_filename: str = 'ruptures_primary_windows.csv'
    window_catalog_filename: str = 'window_catalog.csv'
    primary_points_filename: str = 'ruptures_primary_points.csv'
    point_to_window_audit_filename: str = 'ruptures_point_to_window_audit.csv'
    band_merge_audit_filename: str = 'ruptures_band_merge_audit.csv'
    support_bands_filename: str = 'ruptures_support_bands.csv'
    support_summary_filename: str = 'support_summary.json'
    support_invalid_sample_audit_filename: str = 'support_invalid_sample_audit.csv'
    support_sample_validity_summary_filename: str = 'support_sample_validity_summary.csv'
    support_bootstrap_filename: str = 'support_bootstrap_ruptures.csv'
    support_param_path_filename: str = 'support_param_path_ruptures.csv'
    support_permutation_filename: str = 'support_permutation_ruptures.csv'
    retention_audit_filename: str = 'ruptures_retention_audit.csv'
    run_meta_filename: str = 'run_meta.json'

    freeze_manifest_filename: str = 'audit_input_manifest.json'
    main_window_table_filename: str = 'stage_partition_main_windows.csv'
    window_evidence_mapping_filename: str = 'window_evidence_mapping.csv'
    window_support_audit_filename: str = 'window_support_audit.csv'
    window_retention_audit_filename: str = 'window_retention_audit.csv'
    trust_tiers_filename: str = 'audit_trust_tiers.csv'
    audit_sync_summary_filename: str = 'audit_sync_summary.json'
    run_meta_out_filename: str = 'run_meta.json'

    def layer_root(self) -> Path:
        return self.project_root / 'stage_partition' / 'V2'

    def source_results_dir(self) -> Path:
        return self.layer_root() / 'outputs' / self.source_output_tag

    def output_root(self) -> Path:
        return self.layer_root() / 'outputs' / self.audit_output_tag

    def log_root(self) -> Path:
        return self.layer_root() / 'logs' / self.audit_output_tag

    def to_dict(self) -> dict:
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return [convert(x) for x in obj]
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if hasattr(obj, '__dataclass_fields__'):
                return convert(asdict(obj))
            return obj
        return convert(self)

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
