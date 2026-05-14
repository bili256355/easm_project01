from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import os


@dataclass
class FoundationInputConfig:
    project_root: Path = Path(r"D:\easm_project01")
    foundation_layer: str = "foundation"
    foundation_version: str = "V1"
    preprocess_output_tag: str = "baseline_a"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"

    def smoothed_fields_path(self) -> Path:
        env = os.environ.get(self.smoothed_env_var)
        if env:
            return Path(env)
        return (
            self.project_root
            / self.foundation_layer
            / self.foundation_version
            / "outputs"
            / self.preprocess_output_tag
            / "preprocess"
            / "smoothed_fields.npz"
        )


@dataclass
class HProfileConfig:
    object_name: str = "H"
    field_key: str = "z500_smoothed"
    lat_key: str = "lat"
    lon_key: str = "lon"
    lon_range: tuple[float, float] = (110.0, 140.0)
    lat_range: tuple[float, float] = (15.0, 35.0)
    lat_step_deg: float = 2.0


@dataclass
class StateBuilderConfig:
    standardize: bool = True
    block_equal_contribution: bool = True
    trim_invalid_days: bool = True


@dataclass
class DetectorConfig:
    widths: tuple[int, ...] = (12, 16, 20, 24, 28)
    baseline_width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    pen: float = 4.0
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10


@dataclass
class WindowConfig:
    strong_windows: tuple[tuple[str, int, int, int], ...] = (
        ("W045", 45, 40, 48),
        ("W081", 81, 75, 87),
        ("W113", 113, 108, 118),
        ("W160", 160, 155, 165),
    )
    pre_search_days: int = 20
    post_search_days: int = 20


@dataclass
class ReferenceConfig:
    # V10.2 method-layer baseline recorded in current migration/root-log context.
    expected_h_candidates_width20: tuple[int, ...] = (19, 35, 57, 77, 95, 115, 129, 155)
    expected_match_tolerance_days: int = 2
    v10_2_candidate_catalog_relpath: str = (
        "stage_partition/V10/v10.2/outputs/object_native_peak_discovery_v10_2/"
        "cross_object/object_native_candidate_catalog_all_objects_v10_2.csv"
    )


@dataclass
class OutputConfig:
    output_tag: str = "h_object_event_atlas_v10_7_a"

    def output_root(self, project_root: Path) -> Path:
        return project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / self.output_tag


@dataclass
class Settings:
    project_root: Path = Path(r"D:\easm_project01")
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    h_profile: HProfileConfig = field(default_factory=HProfileConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def with_project_root(self, project_root: Path) -> "Settings":
        self.project_root = Path(project_root)
        self.foundation.project_root = Path(project_root)
        return self

    def to_dict(self) -> dict[str, Any]:
        def convert(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [convert(v) for v in x]
            if isinstance(x, list):
                return [convert(v) for v in x]
            if isinstance(x, dict):
                return {str(k): convert(v) for k, v in x.items()}
            return x

        return convert(asdict(self))
