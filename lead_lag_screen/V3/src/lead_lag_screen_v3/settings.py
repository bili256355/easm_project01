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


# Same object domains as foundation/V1 object_index_builder.REGION_SPECS.
REGION_SPECS: Dict[str, RegionSpec] = {
    "P": RegionSpec(
        "P", "precip", (105.0, 125.0), (10.0, 50.0),
        "P domain reused from foundation/V1 index object definition.",
    ),
    "V": RegionSpec(
        "V", "v850", (105.0, 125.0), (10.0, 30.0),
        "V domain reused from foundation/V1; south boundary fixed at 10N.",
    ),
    "H": RegionSpec(
        "H", "z500", (110.0, 140.0), (10.0, 40.0),
        "H core domain reused from foundation/V1.",
    ),
    "Jw": RegionSpec(
        "Jw", "u200", (80.0, 110.0), (20.0, 50.0),
        "Upstream jet core domain reused from foundation/V1.",
    ),
    "Je": RegionSpec(
        "Je", "u200", (120.0, 150.0), (20.0, 50.0),
        "Downstream jet core domain reused from foundation/V1.",
    ),
}


@dataclass(frozen=True)
class LeadLagScreenV3Settings:
    """
    V3: smooth5 field EOF-PC1 lead-lag audit.

    Scope
    -----
    This layer is an index-applicability audit for V1 smooth5 lead-lag results.
    It replaces manually designed indices with one window-wise EOF PC1 per field
    object and reuses the V1 target-window lead-lag/statistical semantics as much
    as possible.

    Non-goals
    ---------
    - no PCMCI/causal discovery;
    - no pathway reconstruction;
    - no PC2/PC3 in this first audit;
    - no new stage/window definition.
    """

    project_root: Path = Path(r"D:\easm_project01")
    foundation_runtime_tag: str = "baseline_smooth5_a"
    output_tag: str = "eof_pc1_smooth5_v3_a"
    random_seed: int = 20260428

    # Field input preference. The default is anomaly fields to match V1 index_anomalies.
    field_input_mode: str = "anomaly_fields"  # anomaly_fields | smoothed_minus_climatology

    # V1 comparison input.
    v1_output_tag: str = "lead_lag_screen_v1_smooth5_a"

    # Lag and null settings: mirror V1 smooth5 as closely as possible.
    max_lag: int = 5
    diagnostic_max_lag: int = 10
    write_diagnostic_lags: bool = True
    n_surrogates: int = 1000
    n_direction_bootstrap: int = 1000
    n_audit_surrogates: int = 1000
    surrogate_chunk_size: int = 100
    bootstrap_chunk_size: int = 100
    surrogate_mode: str = "pooled_window_object_pc1_ar1"
    audit_surrogate_mode: str = "pooled_phi_yearwise_scale_ar1"
    run_audit_surrogate_null: bool = True

    min_valid_year_fraction: float = 0.70
    min_pairs: int = 30
    p_supported: float = 0.05
    p_marginal: float = 0.10
    q_within_window_supported: float = 0.10

    # EOF controls.
    eof_training_scope: str = "window_only_loading_project_to_padding"
    eof_pc_count: int = 1
    eof_weighting: str = "sqrt_cos_lat"
    eof_grid_min_finite_fraction: float = 0.70
    eof_row_min_finite_fraction: float = 0.50
    eof_quality_good_variance_ratio: float = 0.35
    eof_quality_moderate_variance_ratio: float = 0.20
    sign_reference_min_abs_corr: float = 0.15

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

    sign_reference_variables: Dict[str, str] = field(default_factory=lambda: {
        "P": "P_total_centroid_lat_10_50",
        "V": "V_NS_diff",
        "H": "H_centroid_lat",
        "Je": "Je_axis_lat",
        "Jw": "Jw_axis_lat",
    })

    @property
    def layer_root(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V3"

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
    def input_index_anomalies(self) -> Path:
        return self.indices_dir / "index_anomalies.csv"

    @property
    def v1_output_dir(self) -> Path:
        return self.project_root / "lead_lag_screen" / "V1" / "outputs" / self.v1_output_tag

    @property
    def objects(self) -> List[str]:
        return list(OBJECT_ORDER)

    def to_jsonable(self) -> Dict[str, object]:
        out = asdict(self)
        out["project_root"] = str(self.project_root)
        out["preprocess_dir"] = str(self.preprocess_dir)
        out["indices_dir"] = str(self.indices_dir)
        out["input_index_anomalies"] = str(self.input_index_anomalies)
        out["output_dir"] = str(self.output_dir)
        out["log_dir"] = str(self.log_dir)
        out["v1_output_dir"] = str(self.v1_output_dir)
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
