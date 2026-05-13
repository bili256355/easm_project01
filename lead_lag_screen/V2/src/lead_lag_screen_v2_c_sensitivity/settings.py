from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LeadLagScreenV2CSensitivitySettings:
    """
    lead_lag_screen/V2_c: PCMCI+ sensitivity reruns for the V2_a failure mode.

    This layer is deliberately NOT a replacement for V1 lead-lag and NOT a pathway
    layer. It reruns PCMCI+ variants to audit why V2_a lost physically expected
    V->P / H->P signals.

    Variant C1 (targeted_no_same_family_controls)
    ------------------------------------------------
    Reruns PCMCI+ only for the targeted V->P relation in the targeted Meiyu windows
    S3 and T3. For each concrete source -> target variable pair, the PCMCI+ panel contains:
        source variable, target variable, and variables from families other than
        source_family and target_family.
    This exactly removes other variables from the source family and target family
    from the conditioning pool for that concrete tested pair. It is targeted and
    diagnostic; it is not a full-system replacement graph.

    Variant C2 (family_representative_standard)
    -------------------------------------------
    Reruns PCMCI+ on a smaller representative variable pool to test whether the
    20-index highly collinear system causes graph-selection collapse. This keeps the
    normal PCMCI+ semantics but reduces within-family redundancy.
    """

    project_root: Path = Path(r"D:\easm_project01")
    input_index_anomalies: Path = Path(
        r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv"
    )
    v1_evidence_tier_summary: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a\lead_lag_evidence_tier_summary.csv"
    )
    v2a_output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_smooth5_v2_a"
    )

    output_tag: str = "pcmci_plus_smooth5_v2_c_s3_t3_vp_c2_all_a"
    random_seed: int = 20260428
    smooth_version: str = "smooth5"

    tau_min: int = 0
    tau_max: int = 5
    main_tau_min: int = 1
    main_tau_max: int = 5
    pc_alpha: float = 0.05
    fdr_alpha: float = 0.05
    parcorr_significance: str = "analytic"
    tigramite_verbosity: int = 0

    resume: bool = True
    force_recompute: bool = False
    write_cache: bool = True

    run_targeted_no_same_family_controls: bool = True
    run_family_representative_standard: bool = True

    # Targeted diagnostics: these are the relations that must not disappear in a
    # scientifically plausible control layer. More can be added later, but do not
    # silently broaden this in C without documenting the change.
    targeted_family_pairs: Tuple[Tuple[str, str], ...] = (("V", "P"),)

    # C1 is intentionally restricted to the windows where V2_a produced the most
    # unacceptable V->P loss in the Meiyu / post-Meiyu transition evidence check.
    # C2 still runs on all windows below.
    c1_targeted_windows: Tuple[str, ...] = ("S3", "T3")

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

    # Representative pool intentionally keeps more than one variable for P/V/H,
    # but greatly reduces same-field redundancy compared with the 20-index system.
    representative_variables: Tuple[str, ...] = (
        "P_main_minus_south",
        "P_spread_lat",
        "P_total_centroid_lat_10_50",
        "V_strength",
        "V_NS_diff",
        "H_strength",
        "H_centroid_lat",
        "Je_strength",
        "Je_axis_lat",
        "Jw_strength",
        "Jw_axis_lat",
    )

    include_same_family_reported_edges: bool = False

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
    def variables(self) -> List[str]:
        return list(self.variable_families.keys())

    def to_jsonable_dict(self) -> dict:
        d = asdict(self)
        for key in ["project_root", "input_index_anomalies", "v1_evidence_tier_summary", "v2a_output_dir"]:
            d[key] = str(d[key])
        d["targeted_family_pairs"] = [list(x) for x in self.targeted_family_pairs]
        d["c1_targeted_windows"] = list(self.c1_targeted_windows)
        d["representative_variables"] = list(self.representative_variables)
        return d
