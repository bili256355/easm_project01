from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class LeadLagScreenV2BAuditSettings:
    """
    lead_lag_screen/V2_b audit for pcmci_plus_smooth5_v2_a.

    This audit does not rerun PCMCI+. It reads the completed V2_a outputs and
    decomposes why the final supported-edge layer is much narrower than V1:

    * graph-selected vs raw-p vs window-FDR layers;
    * V1 Tier/Familiy candidates' fate under PCMCI+;
    * tau=0 contemporaneous diagnostics vs lagged edges;
    * interpretation guardrails for treating V2_a as a strict direct-edge lower bound.

    Same-family conditioning sensitivity requires a separate PCMCI+ rerun with a
    changed conditioning design. This fast audit records that limitation instead
    of pretending it has tested the variant.
    """

    project_root: Path = Path(r"D:\easm_project01")
    v2a_output_tag: str = "pcmci_plus_smooth5_v2_a"
    output_tag: str = "pcmci_plus_smooth5_v2_b_audit"

    raw_p_alpha: float = 0.05
    fdr_alpha: float = 0.05
    top_n_examples: int = 80

    @property
    def v2a_output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V2" / "outputs" / self.v2a_output_tag

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V2" / "outputs" / self.output_tag

    @property
    def log_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V2" / "logs" / self.output_tag

    @property
    def v1_evidence_tier_summary(self) -> Path:
        return (
            self.project_root
            / "lead_lag_screen"
            / "V1"
            / "outputs"
            / "lead_lag_screen_v1_smooth5_a"
            / "lead_lag_evidence_tier_summary.csv"
        )

    def to_jsonable_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in asdict(self).items():
            out[k] = str(v) if isinstance(v, Path) else v
        out["v2a_output_dir"] = str(self.v2a_output_dir)
        out["output_dir"] = str(self.output_dir)
        out["log_dir"] = str(self.log_dir)
        out["v1_evidence_tier_summary"] = str(self.v1_evidence_tier_summary)
        return out
