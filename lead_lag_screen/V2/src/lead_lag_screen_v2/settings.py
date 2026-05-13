from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LeadLagScreenV2Settings:
    """
    lead_lag_screen/V2: PCMCI+ smooth5 direct-edge control.

    Scope
    -----
    * Reads the 5-day foundation anomaly-index table.
    * Uses the same 20-variable universe and same 9 target-side windows as V1.
    * Runs PCMCI+ with ParCorr as a conditional direct-edge control.
    * Reports only cross-family source -> target edges for tau=1..5 in the main table.
    * Estimates tau=0 contemporaneous links but writes them to a separate diagnostic table.
    * Allows same-family variables to remain in the PCMCI+ conditioning pool.
    * Does not create pathway chains, mediator claims, or physical-mechanism labels.

    Important: tigramite is required. This layer must fail loudly if tigramite is
    unavailable; it must not silently fall back to correlation, Granger, or a proxy.
    """

    project_root: Path = Path(r"D:\easm_project01")
    input_index_anomalies: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv"
    )
    v1_evidence_tier_summary: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a\lead_lag_evidence_tier_summary.csv"
    )

    output_tag: str = "pcmci_plus_smooth5_v2_a"
    random_seed: int = 20260427

    # V2 is fixed to 5-day smooth-index anomalies. 9-day is not run here.
    smooth_version: str = "smooth5"

    # PCMCI+ settings.
    tau_min: int = 0
    tau_max: int = 5
    main_tau_min: int = 1
    main_tau_max: int = 5
    pc_alpha: float = 0.05
    fdr_alpha: float = 0.05
    parcorr_significance: str = "analytic"
    tigramite_verbosity: int = 0

    # Runtime behavior.
    resume: bool = True
    force_recompute: bool = False
    write_cache: bool = True

    # The V1 semantics are target-side windows: Y(t) must lie in W, while X(t-lag)
    # may come from the immediately preceding days.  To preserve this in PCMCI+,
    # each window panel is padded backward by tau_max days. Missing pre-April days
    # are represented by mask entries, not by cross-year stitching.
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

    # Output/reporting rule. Same-family variables are allowed in the conditioning
    # pool but not reported as source-target edges and not used for chain building.
    include_same_family_reported_edges: bool = False
    allow_same_family_conditioning_pool: bool = True

    @property
    def output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V2" / "outputs" / self.output_tag

    @property
    def log_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V2" / "logs" / self.output_tag

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "_task_cache"

    @property
    def runtime_status_dir(self) -> Path:
        return self.output_dir / "_runtime_status"

    @property
    def variables(self) -> List[str]:
        return list(self.variable_families.keys())

    def to_jsonable_dict(self) -> dict:
        d = asdict(self)
        for key in ["project_root", "input_index_anomalies", "v1_evidence_tier_summary"]:
            d[key] = str(d[key])
        return d
