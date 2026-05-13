from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


WINDOW_DAY_RANGES: dict[str, tuple[int, int]] = {
    "S1": (1, 41),
    "T1": (39, 53),
    "S2": (51, 73),
    "T2": (74, 88),
    "S3": (90, 107),
    "T3": (106, 120),
    "S4": (120, 158),
    "T4": (154, 168),
    "S5": (164, 183),
}


@dataclass(frozen=True)
class T3VToPLag0ReductionSettings:
    project_root: Path
    input_tag: str = "lead_lag_screen_v1_smooth5_a"
    stability_tag: str = "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b"
    previous_audit_tag: str = "lead_lag_screen_v1_smooth5_a_t3_v_to_p_audit"
    output_tag: str = "lead_lag_screen_v1_smooth5_a_t3_v_to_p_lag0_reduction_audit"
    foundation_tag: str = "baseline_smooth5_a"
    focus_window: str = "T3"
    comparison_windows: tuple[str, ...] = ("S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5")
    high_quantile: float = 0.75
    low_quantile: float = 0.25
    max_lag: int = 5
    make_figures: bool = True
    no_cartopy: bool = False
    map_extent: tuple[float, float, float, float] = (20.0, 150.0, 10.0, 60.0)  # lon_min, lon_max, lat_min, lat_max

    @property
    def v1_root(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1"

    @property
    def input_dir(self) -> Path:
        return self.v1_root / "outputs" / self.input_tag

    @property
    def stability_dir(self) -> Path:
        return self.v1_root / "outputs" / self.stability_tag

    @property
    def previous_audit_dir(self) -> Path:
        return self.v1_root / "outputs" / self.previous_audit_tag

    @property
    def output_dir(self) -> Path:
        return self.v1_root / "outputs" / self.output_tag

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"

    @property
    def foundation_dir(self) -> Path:
        return self.project_root / "foundation" / "V1" / "outputs" / self.foundation_tag

    @property
    def smoothed_fields_path(self) -> Path:
        return self.foundation_dir / "preprocess" / "smoothed_fields.npz"

    @property
    def index_values_path(self) -> Path:
        return self.foundation_dir / "indices" / "index_values_smoothed.csv"

    @property
    def index_validity_tables_dir(self) -> Path:
        return (
            self.project_root
            / "index_validity"
            / "V1_b_window_family_guardrail"
            / "outputs"
            / "window_family_guardrail_v1_b_smoothed_a"
            / "tables"
        )

    def focus_days(self) -> tuple[int, int]:
        key = self.focus_window.upper()
        if key not in WINDOW_DAY_RANGES:
            raise KeyError(f"Unknown focus window {self.focus_window!r}. Known: {sorted(WINDOW_DAY_RANGES)}")
        return WINDOW_DAY_RANGES[key]

    def focus_subwindows(self) -> dict[str, tuple[int, int]]:
        start, end = self.focus_days()
        days = list(range(start, end + 1))
        mid = len(days) // 2
        early = (days[0], days[mid - 1])
        late = (days[mid], days[-1])
        return {
            f"{self.focus_window.upper()}_early": early,
            f"{self.focus_window.upper()}_late": late,
        }
