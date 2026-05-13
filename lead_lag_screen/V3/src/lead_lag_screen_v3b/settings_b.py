from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Any

from lead_lag_screen_v3.settings import LeadLagScreenV3Settings


@dataclass(frozen=True)
class StabilitySettings:
    """Additional formal stability-judgement settings for V3_b."""

    output_tag: str = "eof_pc1_smooth5_v3_b_stability_quick_a"
    random_seed_offset: int = 31017

    # Bootstrap sizes. Relation bootstrap defaults to the V3_a relation bootstrap count,
    # but is kept explicit in the run metadata.
    n_relation_bootstrap: int = 1000
    n_pc1_mode_bootstrap: int = 30
    bootstrap_chunk_size: int = 100

    # Formal stability cutoffs.
    stability_ci_level_primary: float = 0.90
    stable_probability_cutoff: float = 0.90
    lag_tau0_close_margin: float = 0.03
    forward_reverse_close_margin: float = 0.03
    peak_lag_stable_mode_fraction: float = 0.60
    peak_lag_moderate_mode_fraction: float = 0.40

    # PC1 mode stability cutoffs. These are guardrails, not physical conclusions.
    pc1_loading_corr_stable_median: float = 0.75
    pc1_loading_corr_stable_p10: float = 0.50
    pc1_loading_corr_moderate_median: float = 0.60
    pc1_sign_consistency_stable: float = 0.90
    pc1_sign_consistency_moderate: float = 0.80
    pc1_score_stability_moderate: float = 0.50
    pc1_score_stability_stable: float = 0.65

    # Main positive-lag support gates; these mirror the V3_a / V1-style gates.
    p_supported: float = 0.05
    q_supported: float = 0.10
    audit_p_supported: float = 0.05
    audit_q_supported: float = 0.10

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


def make_base_settings(base: LeadLagScreenV3Settings | None = None, output_tag: str = "eof_pc1_smooth5_v3_b_stability_quick_a") -> LeadLagScreenV3Settings:
    """Return a V3 base settings object with the V3_b output tag."""
    if base is not None:
        # dataclass is frozen, so reconstruct from public fields through asdict where possible.
        d = base.to_jsonable()
        # Keep only constructor fields known to LeadLagScreenV3Settings.
        allowed = LeadLagScreenV3Settings.__dataclass_fields__.keys()
        kwargs = {k: getattr(base, k) for k in allowed if hasattr(base, k)}
        kwargs["output_tag"] = output_tag
        return LeadLagScreenV3Settings(**kwargs)
    return LeadLagScreenV3Settings(output_tag=output_tag)
