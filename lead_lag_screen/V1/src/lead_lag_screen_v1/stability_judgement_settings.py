from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class V1StabilityJudgementSettings:
    """Post-processing settings for V1 lag-vs-tau0 stability judgement.

    This layer reads existing lead_lag_screen/V1 outputs and adds stricter
    interpretation labels. It does not rerun correlations, surrogates, or
    lead-lag screening.
    """

    project_root: Path = Path(r"D:\easm_project01")
    input_tag: str = "lead_lag_screen_v1_smooth5_a"
    output_tag: str = "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b"

    ci_level: str = "90"
    p_lag_gt_tau0_threshold: float = 0.90
    p_forward_gt_reverse_threshold: float = 0.90

    core_evidence_prefixes: Tuple[str, ...] = ("Tier1a", "Tier1b", "Tier2")
    sensitive_evidence_prefixes: Tuple[str, ...] = ("Tier3",)

    @property
    def input_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1" / "outputs" / self.input_tag

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1" / "outputs" / self.output_tag

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"
