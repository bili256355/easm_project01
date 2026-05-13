from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Literal
import json


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
class StateVectorConfig:
    standardize: bool = True
    block_equal_contribution: bool = True


@dataclass
class MovingWindowConfig:
    enabled: bool = False
    bandwidth: int = 15
    threshold_scale: Optional[float] = 2.0
    level: float = 0.01
    min_detection_interval: int = 1


@dataclass
class RupturesWindowConfig:
    enabled: bool = True
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: Literal["fixed_n_bkps", "pen", "epsilon"] = "pen"
    fixed_n_bkps: Optional[int] = None
    pen: Optional[float] = 4.0
    epsilon: Optional[float] = None


@dataclass
class EdgeConfig:
    min_distance_days: int = 3
    prominence_quantile: float = 0.80
    nearest_peak_search_radius: int = 10


@dataclass
class RupturesWindowConstructionConfig:
    nearest_peak_search_radius: int = 10
    support_rel_prominence: float = 0.35
    support_floor_quantile: float = 0.60
    support_floor_ratio: float = 0.15
    merge_gap_days: int = 3
    min_band_width_days: int = 1
    valley_relief_tolerance: float = 0.10


@dataclass
class SupportAuditConfig:
    enable_param_path: bool = True
    enable_bootstrap: bool = True
    enable_permutation: bool = True
    bootstrap_reps: int = 100
    random_seed: int = 42
    ruptures_width_grid: list[int] = field(default_factory=lambda: [12, 16, 20, 24, 28])
    ruptures_pen_grid: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    permutation_reps: int = 200


@dataclass
class OutputConfig:
    output_tag: str = "baseline_a"
    write_plots: bool = True


@dataclass
class StagePartitionV2Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateVectorConfig = field(default_factory=StateVectorConfig)
    movingwindow: MovingWindowConfig = field(default_factory=MovingWindowConfig)
    ruptures_window: RupturesWindowConfig = field(default_factory=RupturesWindowConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    ruptures_window_construction: RupturesWindowConstructionConfig = field(default_factory=RupturesWindowConstructionConfig)
    support: SupportAuditConfig = field(default_factory=SupportAuditConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def layer_root(self) -> Path:
        return self.foundation.project_root / "stage_partition" / "V2"

    def output_root(self) -> Path:
        return self.layer_root() / "outputs" / self.output.output_tag

    def log_root(self) -> Path:
        return self.layer_root() / "logs" / self.output.output_tag / "mainline"

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
