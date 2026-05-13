from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


WINDOWS: Dict[str, Tuple[int, int]] = {
    "S1": (1, 39),
    "T1": (40, 48),
    "S2": (49, 74),
    "T2": (75, 86),
    "S3": (87, 106),
    "T3": (107, 117),
    "S4": (118, 154),
    "T4": (155, 164),
    "S5": (165, 183),
}


@dataclass(frozen=True)
class IndexValidityV1BSettings:
    """
    V1_b: window-family index representativeness guardrail.

    Default semantics
    -----------------
    The main guardrail uses **smoothed field + smoothed index values** because
    index_validity asks whether an index still indicates its own field object.
    The anomaly mode is retained only as an optional auxiliary audit for
    lead-lag-input semantics; it is not the default.

    This layer does not test lead-lag, pathway, causality, or downstream
    relationships.
    """

    project_root: Path = Path(r"D:\easm_project01")
    foundation_runtime_tag: str = "baseline_smooth5_a"
    output_tag: str = "window_family_guardrail_v1_b_smoothed_a"
    random_seed: int = 20260428

    # Main mode must be smoothed for index-to-field validity.
    # Set to "anomaly" only for optional auxiliary audit matching lead-lag input semantics.
    data_mode: str = "smoothed"

    # Quantile/composite settings.
    high_quantile: float = 0.75
    low_quantile: float = 0.25
    bootstrap_year_reps: int = 200
    eof_n_modes: int = 5
    figure_dpi: int = 170

    # Output control.
    make_figures: bool = True
    use_cartopy_if_available: bool = True
    # User-approved maximum display extent: lon 20-150E, lat 10-60N.
    display_lon_range: Tuple[float, float] = (20.0, 150.0)
    display_lat_range: Tuple[float, float] = (10.0, 60.0)

    # Keep figure count under control. Always include all T3 indices, all family best indices,
    # and high-risk/not-supported indices. If still huge, the cap truncates after priority sort.
    max_diagnostic_figures: int = 140

    # Scoring thresholds. These are conservative guardrail thresholds, not physical proof.
    tier_strong_threshold: float = 0.70
    tier_moderate_threshold: float = 0.55
    tier_weak_threshold: float = 0.40
    family_best_low_threshold: float = 0.40

    windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: dict(WINDOWS))

    @property
    def layer_root(self) -> Path:
        return self.project_root / "index_validity" / "V1_b_window_family_guardrail"

    @property
    def output_dir(self) -> Path:
        return self.layer_root / "outputs" / self.output_tag

    @property
    def log_dir(self) -> Path:
        return self.layer_root / "logs" / self.output_tag

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"

    @property
    def foundation_dir(self) -> Path:
        return self.project_root / "foundation" / "V1" / "outputs" / self.foundation_runtime_tag

    @property
    def preprocess_dir(self) -> Path:
        return self.foundation_dir / "preprocess"

    @property
    def indices_dir(self) -> Path:
        return self.foundation_dir / "indices"

    @property
    def index_anomalies_path(self) -> Path:
        return self.indices_dir / "index_anomalies.csv"

    @property
    def index_values_smoothed_path(self) -> Path:
        return self.indices_dir / "index_values_smoothed.csv"

    @property
    def selected_index_path(self) -> Path:
        if self.data_mode == "smoothed":
            return self.index_values_smoothed_path
        if self.data_mode == "anomaly":
            return self.index_anomalies_path
        raise ValueError(f"Unsupported data_mode={self.data_mode!r}; expected 'smoothed' or 'anomaly'.")

    @property
    def smoothed_fields_bundle(self) -> Path:
        return self.preprocess_dir / "smoothed_fields.npz"

    @property
    def anomaly_fields_bundle(self) -> Path:
        return self.preprocess_dir / "anomaly_fields.npz"

    @property
    def climatology_bundle(self) -> Path:
        return self.preprocess_dir / "daily_climatology.npz"

    def to_jsonable(self) -> Dict[str, object]:
        out = asdict(self)
        for key in ["project_root"]:
            out[key] = str(out[key])
        out.update({
            "data_mode": self.data_mode,
            "layer_root": str(self.layer_root),
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir),
            "index_anomalies_path": str(self.index_anomalies_path),
            "index_values_smoothed_path": str(self.index_values_smoothed_path),
            "selected_index_path": str(self.selected_index_path),
            "anomaly_fields_bundle": str(self.anomaly_fields_bundle),
            "smoothed_fields_bundle": str(self.smoothed_fields_bundle),
            "climatology_bundle": str(self.climatology_bundle),
        })
        return out
