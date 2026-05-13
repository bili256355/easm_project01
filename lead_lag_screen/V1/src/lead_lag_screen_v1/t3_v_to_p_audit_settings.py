from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class T3VToPAuditSettings:
    project_root: Path
    input_tag: str = "lead_lag_screen_v1_smooth5_a"
    stability_tag: str = "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b"
    output_tag: str = "lead_lag_screen_v1_smooth5_a_t3_v_to_p_audit"
    focus_window: str = "T3"
    comparison_windows: tuple[str, ...] = ("S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5")
    include_index_validity_context: bool = True

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
    def output_dir(self) -> Path:
        return self.v1_root / "outputs" / self.output_tag

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"

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
