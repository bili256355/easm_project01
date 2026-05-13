
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LeadLagScreenSettings:
    """
    Hard-coded default settings for the first formal lead-lag temporal screen.

    This layer is intentionally independent from old pathway/V1-V2 outputs.
    It reads foundation/V1 anomaly indices and writes a clean lead_lag_screen/V1
    output package.
    """

    project_root: Path = Path(r"D:\easm_project01")
    input_index_anomalies: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv"
    )

    output_tag: str = "lead_lag_screen_v1_smooth5_a"
    random_seed: int = 20260427

    # Main judgement lags.
    max_lag: int = 5
    # Extra lags are only written to the curve table as diagnostics when enabled.
    diagnostic_max_lag: int = 10
    write_diagnostic_lags: bool = True

    # Formal null / resampling sizes.
    n_surrogates: int = 1000
    n_direction_bootstrap: int = 1000
    surrogate_chunk_size: int = 100
    bootstrap_chunk_size: int = 100

    # Main null uses a pooled window-variable AR(1) persistence structure.
    # Audit null keeps pooled phi but rescales each year's synthetic series by
    # that year's mean/std to expose sensitivity to yearwise amplitude differences.
    surrogate_mode: str = "pooled_window_variable_ar1"
    run_audit_surrogate_null: bool = True
    audit_surrogate_mode: str = "pooled_phi_yearwise_scale_ar1"
    n_audit_surrogates: int = 1000

    # Minimum sample controls. If violated, the pair is not evaluable, not "no".
    min_valid_year_fraction: float = 0.70
    min_pairs: int = 30

    # Statistical support thresholds.
    p_supported: float = 0.05
    p_marginal: float = 0.10
    q_within_window_supported: float = 0.10

    # Windows are target-side windows: Y(t) must lie in W.
    # Day convention follows foundation index table: day 1 = Apr 1.
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

    variable_families: Dict[str, str] = field(default_factory=lambda: {
        "P_main_band_share": "P",
        "P_south_band_share_18_24": "P",
        "P_main_minus_south": "P",
        "P_spread_lat": "P",
        "P_north_band_share_35_45": "P",
        "P_north_minus_main_35_45": "P",
        "P_total_centroid_lat_10_50": "P",

        "V_strength": "V",
        "V_pos_centroid_lat": "V",
        "V_NS_diff": "V",

        "H_strength": "H",
        "H_centroid_lat": "H",
        "H_west_extent_lon": "H",
        "H_zonal_width": "H",

        "Je_strength": "Je",
        "Je_axis_lat": "Je",
        "Je_meridional_width": "Je",

        "Jw_strength": "Jw",
        "Jw_axis_lat": "Jw",
        "Jw_meridional_width": "Jw",
    })

    include_same_family_pairs: bool = False

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1" / "outputs" / self.output_tag

    @property
    def log_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1" / "logs" / self.output_tag

    @property
    def variables(self) -> List[str]:
        return list(self.variable_families.keys())
