from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class V10_8_A_Settings:
    """Settings for V10.8_a object-internal transition content audit.

    V10.8_a is intentionally object-internal only:
    - no joint-window alignment
    - no accepted-window grouping
    - no cross-object coupling/precursor/pathway interpretation
    - no breakpoint re-detection
    """

    project_root: Path
    progress: bool = True
    version: str = "v10.8_a"
    output_tag: str = "object_internal_transition_content_v10_8_a"
    v10_2_output_tag: str = "object_native_peak_discovery_v10_2"
    detector_width_override: int | None = None
    flank_half_width_override: int | None = None
    smoothed_fields_path_override: Path | None = None
    clean_output: bool = True
    write_root_log: bool = True
    make_figures: bool = True

    def v10_2_output_root(self) -> Path:
        return (
            self.project_root
            / "stage_partition"
            / "V10"
            / "v10.2"
            / "outputs"
            / self.v10_2_output_tag
        )

    def v10_2_windows_path(self) -> Path:
        return (
            self.v10_2_output_root()
            / "cross_object"
            / "object_native_derived_windows_all_objects_v10_2.csv"
        )

    def v10_2_config_path(self) -> Path:
        return self.v10_2_output_root() / "config_used.json"

    def smoothed_fields_path(self) -> Path:
        if self.smoothed_fields_path_override is not None:
            return self.smoothed_fields_path_override
        return (
            self.project_root
            / "foundation"
            / "V1"
            / "outputs"
            / "baseline_a"
            / "preprocess"
            / "smoothed_fields.npz"
        )

    def output_root(self) -> Path:
        return (
            self.project_root
            / "stage_partition"
            / "V10"
            / "v10.8"
            / "outputs"
            / self.output_tag
        )

    def root_log_path(self) -> Path:
        return self.project_root / "ROOT_LOG_V10_8_A_OBJECT_INTERNAL_TRANSITION_CONTENT.md"

    def to_dict(self) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [conv(v) for v in x]
            if isinstance(x, list):
                return [conv(v) for v in x]
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            return x

        return conv(asdict(self))
