from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import FoundationInputConfig, HProfileConfig, StateBuilderConfig


@dataclass
class ScaleDiagnosticConfig:
    sigmas: tuple[float, ...] = (2, 3, 4, 5, 7, 9, 11, 14)
    day_min: int = 0
    day_max: int = 70
    focus_day_min: int = 10
    focus_day_max: int = 62
    target_radius_days: int = 3
    ridge_link_radius_days: int = 3
    local_max_percentile_threshold: float = 75.0
    local_max_min_prominence_norm: float = 0.08
    boundary_sigma_multiplier: float = 3.0
    selected_panel_sigmas: tuple[float, ...] = (3, 5, 9, 14)


@dataclass
class TargetDayConfig:
    target_days: dict[str, int] = field(
        default_factory=lambda: {"H19": 19, "H35": 35, "H45": 45, "H57": 57}
    )


@dataclass
class V10_7_AReferenceConfig:
    output_tag: str = "h_object_event_atlas_v10_7_a"

    def output_root(self, project_root: Path) -> Path:
        return project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / self.output_tag

    def candidate_catalog_path(self, project_root: Path) -> Path:
        return self.output_root(project_root) / "tables" / "h_candidate_catalog_by_width_v10_7_a.csv"

    def event_atlas_path(self, project_root: Path) -> Path:
        return self.output_root(project_root) / "tables" / "h_event_atlas_by_window_width_v10_7_a.csv"


@dataclass
class ScaleOutputConfig:
    output_tag: str = "h_w045_scale_diagnostic_v10_7_b"

    def output_root(self, project_root: Path) -> Path:
        return project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / self.output_tag


@dataclass
class ScaleSettings:
    project_root: Path = Path(r"D:\easm_project01")
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    h_profile: HProfileConfig = field(default_factory=HProfileConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    scale: ScaleDiagnosticConfig = field(default_factory=ScaleDiagnosticConfig)
    targets: TargetDayConfig = field(default_factory=TargetDayConfig)
    v10_7_a_reference: V10_7_AReferenceConfig = field(default_factory=V10_7_AReferenceConfig)
    output: ScaleOutputConfig = field(default_factory=ScaleOutputConfig)

    def with_project_root(self, project_root: Path) -> "ScaleSettings":
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
