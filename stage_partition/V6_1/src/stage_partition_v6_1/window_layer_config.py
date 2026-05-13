
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json


@dataclass
class SourceV6Config:
    project_root: Path = Path(r"D:\easm_project01")
    source_v6_output_tag: str = 'mainline_v6_a'

    def v6_root(self) -> Path:
        return self.project_root / 'stage_partition' / 'V6'

    def source_output_root(self) -> Path:
        return self.v6_root() / 'outputs' / self.source_v6_output_tag


@dataclass
class FoundationInputConfig:
    project_root: Path = Path(r"D:\easm_project01")
    foundation_layer: str = 'foundation'
    foundation_version: str = 'V1'
    preprocess_output_tag: str = 'baseline_a'

    def foundation_root(self) -> Path:
        return self.project_root / self.foundation_layer / self.foundation_version

    def preprocess_dir(self) -> Path:
        return self.foundation_root() / 'outputs' / self.preprocess_output_tag / 'preprocess'

    def smoothed_fields_path(self) -> Path:
        return self.preprocess_dir() / 'smoothed_fields.npz'


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
    model: str = 'l2'
    min_size: int = 2
    jump: int = 1
    selection_mode: str = 'pen'
    fixed_n_bkps: int | None = None
    pen: float | None = 4.0
    epsilon: float | None = None
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10


@dataclass
class WindowBandConfig:
    min_band_half_width_days: int = 2
    max_band_half_width_days: int = 10
    peak_floor_quantile: float = 0.35
    prominence_ratio_threshold: float = 0.50
    respect_candidate_boundaries: bool = True
    truncate_at_intervening_candidate: bool = True
    truncate_at_local_valley: bool = True
    allow_band_merge: bool = True
    merge_gap_days: int = 1
    close_neighbor_exemption_days: int = 4
    protect_significant_peaks_from_merge: bool = True
    significant_peak_threshold: float = 0.95


@dataclass
class WindowUncertaintyConfig:
    interval_match_types: tuple[str, ...] = ('strict', 'matched', 'near')
    emit_width80: bool = True
    emit_width95: bool = True


@dataclass
class OutputConfig:
    output_tag: str = 'mainline_v6_1_a'


@dataclass
class StagePartitionV61Settings:
    source_v6: SourceV6Config = field(default_factory=SourceV6Config)
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    detector: RupturesWindowConfig = field(default_factory=RupturesWindowConfig)
    band: WindowBandConfig = field(default_factory=WindowBandConfig)
    uncertainty: WindowUncertaintyConfig = field(default_factory=WindowUncertaintyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def layer_root(self) -> Path:
        return self.foundation.project_root / 'stage_partition' / 'V6_1'

    def output_root(self) -> Path:
        return self.layer_root() / 'outputs' / self.output.output_tag

    def log_root(self) -> Path:
        return self.layer_root() / 'logs' / self.output.output_tag

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
                return {k: convert(v) for k, v in asdict(obj).items()}
            return obj
        return convert(asdict(self))

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding='utf-8')
