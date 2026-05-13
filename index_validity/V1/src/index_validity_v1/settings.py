\
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexValiditySettings:
    """
    Smooth5 index validity diagnostics.

    This layer checks:
      1) whether 5-day smoothed index yearwise trajectories look abnormal;
      2) whether high/low index samples correspond to the intended physical fields;
      3) only basic anomaly reconstruction consistency.

    It explicitly does not evaluate lead-lag, pathway, autocorrelation, or downstream impacts.
    """

    project_root: Path = Path(r"D:\easm_project01")
    runtime_tag: str = "smooth5_v1_a"

    high_quantile: float = 0.80
    low_quantile: float = 0.20
    yearwise_flag_quantile: float = 0.90
    figure_dpi: int = 180
    tolerance: float = 1e-8

    # Map drawing: use cartopy when installed. If unavailable, write non-projected maps and record it.
    use_cartopy_if_available: bool = True

    @property
    def foundation_root(self) -> Path:
        return self.project_root / "foundation" / "V1" / "outputs" / "baseline_smooth5_a"

    @property
    def indices_root(self) -> Path:
        return self.foundation_root / "indices"

    @property
    def preprocess_root(self) -> Path:
        return self.foundation_root / "preprocess"

    @property
    def index_values_path(self) -> Path:
        return self.indices_root / "index_values_smoothed.csv"

    @property
    def index_climatology_path(self) -> Path:
        return self.indices_root / "index_daily_climatology.csv"

    @property
    def index_anomalies_path(self) -> Path:
        return self.indices_root / "index_anomalies.csv"

    @property
    def smoothed_fields_bundle(self) -> Path:
        return self.preprocess_root / "smoothed_fields.npz"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "index_validity" / "V1" / "outputs" / self.runtime_tag

    @property
    def log_dir(self) -> Path:
        return self.project_root / "index_validity" / "V1" / "logs" / self.runtime_tag

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def yearwise_figures_dir(self) -> Path:
        return self.figures_dir / "yearwise"

    @property
    def physical_figures_dir(self) -> Path:
        return self.figures_dir / "physical"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"
