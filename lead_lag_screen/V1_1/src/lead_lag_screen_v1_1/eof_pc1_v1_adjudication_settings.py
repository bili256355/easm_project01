from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class EOFPC1V1AdjudicationSettings:
    """EOF-PC1 adjudication audit against the V1/V1_1 old-index problem space.

    This audit does not ask whether EOF-PC1 contains the V1_1 high-latitude
    branch.  It asks whether EOF-PC1 is actually eligible to adjudicate the V1
    old-index T3 weakening result.
    """

    project_root: Path = Path(r"D:\easm_project01")
    output_tag: str = "lead_lag_screen_v1_1_eof_pc1_v1_adjudication_a"

    eof_interpretability_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_eof_pc1_interpretability_audit_v1_a"
    )
    v1_1_main_output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_structural_vp_a"
    )
    v1_1_index_anomalies: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_structural_vp_a\indices\v1_1_index_values_doy_anomaly.csv"
    )

    max_lag: int = 5
    min_pairs: int = 30
    lag_tau0_margin: float = 0.02
    stable_abs_corr_floor: float = 0.10

    # Alignment thresholds: diagnostic only, not a significance test.
    aligned_r2_threshold: float = 0.50
    partial_r2_threshold: float = 0.25
    strong_corr_threshold: float = 0.50
    moderate_corr_threshold: float = 0.30

    windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "S1": (1, 39),
        "T1": (40, 48),
        "S2": (49, 74),
        "T2": (75, 86),
        "S3": (87, 106),
        "T3": (107, 117),
        "S4": (118, 154),
        "T4": (155, 164),
        "S5": (165, 183),
    })

    old_v_indices: Tuple[str, ...] = (
        "V_strength",
        "V_pos_centroid_lat",
        "V_NS_diff",
    )
    old_p_indices: Tuple[str, ...] = (
        "P_main_band_share",
        "P_south_band_share_18_24",
        "P_main_minus_south",
        "P_spread_lat",
        "P_north_band_share_35_45",
        "P_north_minus_main_35_45",
        "P_total_centroid_lat_10_50",
    )

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1_1" / "outputs" / self.output_tag

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1_1" / "logs" / self.output_tag

    @property
    def p_pc_scores_path(self) -> Path:
        return self.eof_interpretability_dir / "tables" / "p_eof_pc_scores.csv"

    @property
    def v_pc_scores_path(self) -> Path:
        return self.eof_interpretability_dir / "tables" / "v_eof_pc_scores.csv"

    @property
    def v1_1_classified_pairs_path(self) -> Path:
        return self.v1_1_main_output_dir / "tables" / "v1_1_v_to_p_classified_pairs.csv"

    @property
    def v1_1_recovery_summary_path(self) -> Path:
        return self.v1_1_main_output_dir / "tables" / "v1_vs_v1_1_pair_recovery_summary.csv"
