from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LeadLagScreenSettings:
    """
    V1_1 structural V→P screen.

    Contract:
    - V1 is read-only.
    - V1_1 computes new structural P/V indices from smooth5 fields.
    - V1_1 reuses the V1 lead-lag/surrogate/lag-vs-tau0 judgement semantics.
    - Main direction is V→P only across all 9 V1 windows.
    """

    project_root: Path = Path(r"D:\easm_project01")

    # Read-only V1 / foundation inputs.
    input_v1_index_anomalies: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv"
    )
    input_v1_index_values_smoothed: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_values_smoothed.csv"
    )
    input_smoothed_fields: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\preprocess\smoothed_fields.npz"
    )
    previous_v1_stability_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables"
    )

    # V1_1 generated main input to the V1-style screen.
    generated_index_anomalies_name: str = "v1_1_index_values_doy_anomaly.csv"
    generated_index_raw_name: str = "v1_1_index_values_raw.csv"
    output_tag: str = "lead_lag_screen_v1_1_structural_vp_a"
    random_seed: int = 20260429

    # Main judgement lags: inherited from V1.
    max_lag: int = 5
    diagnostic_max_lag: int = 10
    write_diagnostic_lags: bool = True

    # Formal null / resampling sizes: inherited from V1 defaults.
    n_surrogates: int = 1000
    n_direction_bootstrap: int = 1000
    surrogate_chunk_size: int = 100
    bootstrap_chunk_size: int = 100
    surrogate_mode: str = "pooled_window_variable_ar1"
    run_audit_surrogate_null: bool = True
    audit_surrogate_mode: str = "pooled_phi_yearwise_scale_ar1"
    n_audit_surrogates: int = 1000

    min_valid_year_fraction: float = 0.70
    min_pairs: int = 30
    p_supported: float = 0.05
    p_marginal: float = 0.10
    q_within_window_supported: float = 0.10

    # Windows are target-side windows; day 1 = Apr 1.
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

    # Old V1 V/P indices retained as baseline.
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

    # New V1_1 structural V indices.
    new_v_indices: Tuple[str, ...] = (
        "V_pos_north_edge_lat",
        "V_pos_south_edge_lat",
        "V_pos_band_width",
        "V_pos_centroid_lat_recomputed",
        "V_highlat_35_55_mean",
        "V_lowlat_20_30_mean",
        "V_high_minus_low_35_55_minus_20_30",
        "V_lowlat_weakening_proxy_20_30",
    )

    # New V1_1 structural P indices.
    new_p_indices: Tuple[str, ...] = (
        "P_main_28_35_mean",
        "P_south_10_25_mean",
        "P_scs_10_20_mean",
        "P_highlat_40_60_mean",
        "P_highlat_35_60_mean",
        "P_highlat_minus_main",
        "P_south_minus_main",
        "P_south_plus_highlat_minus_main",
    )

    # Structural index geometry.
    index_lon_range: Tuple[float, float] = (100.0, 135.0)
    p_main_lat_range: Tuple[float, float] = (28.0, 35.0)
    p_main_lon_range: Tuple[float, float] = (100.0, 125.0)
    p_south_lat_range: Tuple[float, float] = (10.0, 25.0)
    p_south_lon_range: Tuple[float, float] = (105.0, 130.0)
    p_scs_lat_range: Tuple[float, float] = (10.0, 20.0)
    p_scs_lon_range: Tuple[float, float] = (105.0, 130.0)
    p_highlat_40_60_lat_range: Tuple[float, float] = (40.0, 60.0)
    p_highlat_35_60_lat_range: Tuple[float, float] = (35.0, 60.0)
    v_highlat_lat_range: Tuple[float, float] = (35.0, 55.0)
    v_lowlat_lat_range: Tuple[float, float] = (20.0, 30.0)
    v_positive_threshold_abs: float = 0.10
    v_positive_threshold_fraction_of_max: float = 0.10
    v_positive_min_consecutive_grid: int = 2

    include_same_family_pairs: bool = False

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1_1" / "outputs" / self.output_tag

    @property
    def log_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1_1" / "logs" / self.output_tag

    @property
    def index_dir(self) -> Path:
        return self.output_dir / "indices"

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"

    @property
    def generated_index_anomalies(self) -> Path:
        return self.index_dir / self.generated_index_anomalies_name

    @property
    def generated_index_raw(self) -> Path:
        return self.index_dir / self.generated_index_raw_name

    @property
    def input_index_anomalies(self) -> Path:
        """Path consumed by the inherited V1-style lead-lag core."""
        return self.generated_index_anomalies

    @property
    def variable_families(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for v in self.old_v_indices + self.new_v_indices:
            out[v] = "V"
        for p in self.old_p_indices + self.new_p_indices:
            out[p] = "P"
        return out

    @property
    def variables(self) -> List[str]:
        return list(self.variable_families.keys())

    @property
    def index_type_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for v in self.old_v_indices + self.old_p_indices:
            out[v] = "old_v1"
        for v in self.new_v_indices + self.new_p_indices:
            out[v] = "new_v1_1"
        return out
