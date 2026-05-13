from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


OBJECT_ORDER: Tuple[str, ...] = ("P", "V", "H", "Je", "Jw")


@dataclass(frozen=True)
class RegionSpec:
    object_name: str
    source_field: str
    lon_range: Tuple[float, float]
    lat_range: Tuple[float, float]
    note: str


# Reuse the same object domains as foundation/V1 and lead_lag_screen/V3.
REGION_SPECS: Dict[str, RegionSpec] = {
    "P": RegionSpec("P", "precip", (105.0, 125.0), (10.0, 50.0), "P domain reused from foundation/V1."),
    "V": RegionSpec("V", "v850", (105.0, 125.0), (10.0, 30.0), "V domain reused from foundation/V1; south boundary fixed at 10N."),
    "H": RegionSpec("H", "z500", (110.0, 140.0), (10.0, 40.0), "H core domain reused from foundation/V1."),
    "Jw": RegionSpec("Jw", "u200", (80.0, 110.0), (20.0, 50.0), "Upstream jet core domain reused from foundation/V1."),
    "Je": RegionSpec("Je", "u200", (120.0, 150.0), (20.0, 50.0), "Downstream jet core domain reused from foundation/V1."),
}


@dataclass(frozen=True)
class LeadLagScreenV4Settings:
    """
    V4: smooth5 field lagged-CCA audit.

    Scope
    -----
    This layer is a field-to-field coupling-mode audit after V1/V3. It uses
    smooth5 anomaly fields, window-wise EOF dimensionality reduction, and lagged
    CCA on selected core field pairs. It is designed to test whether an apparent
    index-level weakening/collapse is an index applicability artifact.

    Non-goals
    ---------
    - no pathway reconstruction;
    - no PCMCI/causal discovery;
    - no all-pair exhaustive field network in the first version;
    - no replacement of V1 lead-lag index results.
    """

    project_root: Path = Path(r"D:\easm_project01")
    foundation_runtime_tag: str = "baseline_smooth5_a"
    output_tag: str = "field_cca_smooth5_v4_a"
    random_seed: int = 20260428

    # Inputs for comparison only.
    v1_output_tag: str = "lead_lag_screen_v1_smooth5_a"
    v3_output_tag: str = "eof_pc1_smooth5_v3_a"

    # Core pair directions: X(t-lag) field vs Y(t) field.
    # Keep this intentionally small for the first CCA audit.
    core_pairs: Tuple[Tuple[str, str], ...] = (
        ("V", "P"),
        ("H", "P"),
        ("H", "V"),
        ("Jw", "Je"),
        ("Je", "H"),
    )

    # Window definition mirrors V1/V3. Day 1 = Apr 1.
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

    # CCA lag design.
    max_lag: int = 5                  # lag = 0..5; lag 0 is tau0/contemporaneous
    min_pairs: int = 30
    cv_folds: int = 5
    ridge: float = 1e-5
    min_cv_pairs: int = 20

    # EOF controls. k=5 is headline; k=3 is lower-dimensional sensitivity.
    eof_k_values: Tuple[int, ...] = (5, 3)
    eof_max_modes: int = 5
    eof_weighting: str = "sqrt_cos_lat"
    eof_grid_min_finite_fraction: float = 0.70
    eof_row_min_finite_fraction: float = 0.50
    eof_quality_good_variance_ratio: float = 0.35
    eof_quality_moderate_variance_ratio: float = 0.20

    # Permutation/bootstrap controls. These are audits, not causal proof.
    n_permutations: int = 200
    n_bootstrap: int = 200
    bootstrap_abs_r_threshold: float = 0.30
    permutation_mode: str = "year_block_permute_target_preserve_within_year_days"
    bootstrap_mode: str = "year_block_resample_at_observed_best_lag"

    # Classification thresholds for interpretation hints.
    strong_abs_cv_r: float = 0.50
    moderate_abs_cv_r: float = 0.30
    lag_tau0_close_margin: float = 0.05
    p_supported: float = 0.05
    p_marginal: float = 0.10

    @property
    def layer_root(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V4"

    @property
    def output_dir(self) -> Path:
        return self.layer_root / "outputs" / self.output_tag

    @property
    def log_dir(self) -> Path:
        return self.layer_root / "logs" / self.output_tag

    @property
    def preprocess_dir(self) -> Path:
        return self.project_root / "foundation" / "V1" / "outputs" / self.foundation_runtime_tag / "preprocess"

    @property
    def indices_dir(self) -> Path:
        return self.project_root / "foundation" / "V1" / "outputs" / self.foundation_runtime_tag / "indices"

    @property
    def v1_output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1" / "outputs" / self.v1_output_tag

    @property
    def v3_output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V3" / "outputs" / self.v3_output_tag

    @property
    def objects(self) -> List[str]:
        return list(OBJECT_ORDER)

    def to_jsonable(self) -> Dict[str, object]:
        out = asdict(self)
        out["project_root"] = str(self.project_root)
        out["preprocess_dir"] = str(self.preprocess_dir)
        out["indices_dir"] = str(self.indices_dir)
        out["v1_output_dir"] = str(self.v1_output_dir)
        out["v3_output_dir"] = str(self.v3_output_dir)
        out["output_dir"] = str(self.output_dir)
        out["log_dir"] = str(self.log_dir)
        out["region_specs"] = {
            k: {
                "object_name": v.object_name,
                "source_field": v.source_field,
                "lon_range": v.lon_range,
                "lat_range": v.lat_range,
                "note": v.note,
            }
            for k, v in REGION_SPECS.items()
        }
        return out
