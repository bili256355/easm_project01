from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class EOFPC1InterpretabilitySettings:
    """V1_1 EOF-PC1 interpretability audit.

    This audit is intentionally diagnostic.  It asks whether EOF-PC1 actually
    carries the T3 high-latitude / boundary / retreat structures that V1_1
    structural indices exposed.  It does not replace the V1_1 lead-lag screen.
    """

    project_root: Path = Path(r"D:\easm_project01")
    input_smoothed_fields: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\preprocess\smoothed_fields.npz"
    )
    input_v1_1_structural_indices: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_structural_vp_a\indices\v1_1_index_values_doy_anomaly.csv"
    )

    output_tag: str = "lead_lag_screen_v1_1_eof_pc1_interpretability_audit_v1_a"
    random_seed: int = 20260429

    # EOF domain and calculation mode.
    eof_lat_range: Tuple[float, float] = (10.0, 60.0)
    eof_lon_range: Tuple[float, float] = (100.0, 135.0)
    eof_value_mode: str = "doy_anomaly"  # doy_anomaly or raw_centered
    eof_method: str = "iterative_topk"  # deterministic subspace iteration for top modes
    n_modes: int = 5
    n_iter: int = 28
    spatial_stride: int = 1
    use_coslat_weight: bool = True

    # Lead-lag audit, simple PC-level comparison only.
    max_lag: int = 5
    min_pairs: int = 30
    lag_tau0_margin: float = 0.02

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

    reconstruction_windows: Tuple[str, ...] = ("S3", "T3", "S4")
    reconstruction_comparisons: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "T3_minus_S3": ("T3", "S3"),
        "S4_minus_T3": ("S4", "T3"),
    })

    # Regions used only for loading summaries and reconstruction skill.
    regions: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "main_28_35": {"lat": (28.0, 35.0), "lon": (100.0, 125.0)},
        "south_10_25": {"lat": (10.0, 25.0), "lon": (105.0, 130.0)},
        "scs_10_20": {"lat": (10.0, 20.0), "lon": (105.0, 130.0)},
        "highlat_40_60": {"lat": (40.0, 60.0), "lon": (100.0, 135.0)},
        "highlat_35_60": {"lat": (35.0, 60.0), "lon": (100.0, 135.0)},
        "lowlat_20_30": {"lat": (20.0, 30.0), "lon": (100.0, 135.0)},
        "main_easm_domain": {"lat": (10.0, 60.0), "lon": (100.0, 135.0)},
    })

    p_structural_indices: Tuple[str, ...] = (
        "P_highlat_40_60_mean",
        "P_highlat_35_60_mean",
        "P_main_28_35_mean",
        "P_south_10_25_mean",
        "P_highlat_minus_main",
        "P_south_minus_main",
    )
    v_structural_indices: Tuple[str, ...] = (
        "V_pos_north_edge_lat",
        "V_pos_band_width",
        "V_highlat_35_55_mean",
        "V_lowlat_20_30_mean",
        "V_high_minus_low_35_55_minus_20_30",
    )

    selected_structural_pairs: Tuple[Tuple[str, str], ...] = (
        ("V_pos_north_edge_lat", "P_highlat_40_60_mean"),
        ("V_highlat_35_55_mean", "P_highlat_40_60_mean"),
        ("V_high_minus_low_35_55_minus_20_30", "P_highlat_40_60_mean"),
        ("V_pos_band_width", "P_highlat_40_60_mean"),
        ("V_pos_north_edge_lat", "P_highlat_35_60_mean"),
    )

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1_1" / "outputs" / self.output_tag

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
    def log_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1_1" / "logs" / self.output_tag
