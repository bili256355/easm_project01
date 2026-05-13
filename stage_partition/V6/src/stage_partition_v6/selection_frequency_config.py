from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json


# IMPORTANT:
# This file is intentionally self-contained and does NOT import from
# stage_partition_v6.config. The selection-frequency experiment must stay
# isolated from the main V6-a chain, so changes to V6-a bootstrap/output
# settings do not leak into V6-b.


@dataclass
class FoundationInputConfig:
    project_root: Path = Path(r"D:\easm_project01")
    foundation_layer: str = "foundation"
    foundation_version: str = "V1"
    preprocess_output_tag: str = "baseline_a"

    def foundation_root(self) -> Path:
        return self.project_root / self.foundation_layer / self.foundation_version

    def preprocess_dir(self) -> Path:
        return self.foundation_root() / "outputs" / self.preprocess_output_tag / "preprocess"

    def smoothed_fields_path(self) -> Path:
        return self.preprocess_dir() / "smoothed_fields.npz"


@dataclass
class ProfileGridConfig:
    lat_step_deg: float = 2.0
    p_lon_range: tuple[float, float] = (105.0, 125.0)
    p_lat_range: tuple[float, float] = (15.0, 39.0)
    v_lon_range: tuple[float, float] = (105.0, 125.0)
    v_lat_range: tuple[float, float] = (10.0, 30.0)
    h_lon_range: tuple[float, float] = (110.0, 140.0)
    h_lat_range: tuple[float, float] = (15.0, 35.0)
    je_lon_range: tuple[float, float] = (120.0, 150.0)
    je_lat_range: tuple[float, float] = (25.0, 45.0)
    jw_lon_range: tuple[float, float] = (80.0, 110.0)
    jw_lat_range: tuple[float, float] = (25.0, 45.0)


@dataclass
class StateBuilderConfig:
    standardize: bool = True
    block_equal_contribution: bool = True
    trim_invalid_days: bool = True


@dataclass
class RupturesWindowConfig:
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: str = "pen"
    fixed_n_bkps: int | None = None
    pen: float | None = 4.0
    epsilon: float | None = None
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10


@dataclass
class BootstrapConfig:
    # V6-b keeps its own bootstrap count here and must not inherit V6-a changes.
    n_bootstrap: int = 1000
    random_seed: int = 42
    progress: bool = True
    strict_match_max_abs_offset_days: int = 2
    match_max_abs_offset_days: int = 4
    near_max_abs_offset_days: int = 8


@dataclass
class SelectionFrequencyExperimentConfig:
    output_tag: str = 'selection_frequency_v6_b'
    use_smoothing: bool = True
    smoothing_window_days: int = 3
    emit_local_maxima: bool = True
    local_maxima_min_frequency: float = 0.10
    write_replicate_peak_counts: bool = True
    write_curve_png: bool = True
    curve_png_filename: str = 'bootstrap_selection_frequency_curve.png'
    annotate_baseline_peaks: bool = True
    curve_dpi: int = 200


@dataclass
class StagePartitionV6SelectionFrequencySettings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    detector: RupturesWindowConfig = field(default_factory=RupturesWindowConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    experiment: SelectionFrequencyExperimentConfig = field(default_factory=SelectionFrequencyExperimentConfig)

    def layer_root(self) -> Path:
        return self.foundation.project_root / 'stage_partition' / 'V6'

    def output_root(self) -> Path:
        return self.layer_root() / 'outputs' / self.experiment.output_tag

    def log_root(self) -> Path:
        return self.layer_root() / 'logs' / self.experiment.output_tag

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
