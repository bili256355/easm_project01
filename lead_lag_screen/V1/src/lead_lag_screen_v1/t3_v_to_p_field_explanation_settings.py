# -*- coding: utf-8 -*-
"""Settings for T3 V→P field-explanation hard-evidence audit v1_a."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List


@dataclass(frozen=True)
class RegionSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class FieldExplanationAuditSettings:
    """Hard-coded defaults for the field-explanation audit.

    Scientific status
    -----------------
    This layer is a hard-evidence audit for T3 V→P contraction. It is not a
    pathway establishment model and does not change V1 temporal-eligibility
    labels. It uses V indices as source variables and the precipitation field
    as target to diagnose field-level weakening, tau0 replacement, V-component
    replacement, P-target shift, transition-window mixing, and local V1 design
    limitations.
    """

    project_root: Path = Path(r"D:\easm_project01")
    output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a"
    )

    # Foundation data. These are intentionally the smooth5 foundation products.
    foundation_preprocess_rel: str = "foundation/V1/outputs/baseline_smooth5_a/preprocess"
    foundation_indices_rel: str = "foundation/V1/outputs/baseline_smooth5_a/indices"

    # Window basis. Default follows V6_1/V1 main-screen basis, not the older
    # physical-hypothesis audit's wider T3 window.
    use_legacy_t3_window: bool = False
    windows_main: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "S3": (87, 106),
        "T3_early": (107, 112),
        "T3_full": (107, 117),
        "T3_late": (113, 117),
        "S4": (118, 154),
    })
    windows_legacy: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "S3": (90, 107),
        "T3_early": (106, 113),
        "T3_full": (106, 120),
        "T3_late": (114, 120),
        "S4": (120, 158),
    })

    v_indices: List[str] = field(default_factory=lambda: [
        "V_strength",
        "V_NS_diff",
        "V_pos_centroid_lat",
    ])

    # Lag design: lag 0 is tau0/same-day. Positive lags are V(t-lag) -> P(t).
    max_lag: int = 5

    # Bootstrap is applied to regional response / similarity diagnostics by
    # year-block resampling. Full gridpoint bootstrap maps are intentionally not
    # generated to avoid an unmanageable computation.
    n_bootstrap: int = 1000
    bootstrap_seed: int = 20260429

    make_figures: bool = True
    # Map figures default to cartopy. Use --no-cartopy only as an emergency
    # fallback when cartopy is not available in the runtime environment.
    use_cartopy: bool = True

    # P-field aliases and coordinates. The resolver also accepts *_smoothed.
    precip_aliases: Tuple[str, ...] = (
        "precip", "precipitation", "pr", "rain", "P",
    )

    # Pre-registered regions. Do not edit boundaries based on output maps.
    regions: Dict[str, RegionSpec] = field(default_factory=lambda: {
        "main_meiyu": RegionSpec(24.0, 35.0, 100.0, 125.0),
        "south_china": RegionSpec(18.0, 25.0, 105.0, 120.0),
        "scs": RegionSpec(10.0, 20.0, 105.0, 130.0),
        "south_china_scs": RegionSpec(10.0, 25.0, 105.0, 130.0),
        "north_northeast": RegionSpec(35.0, 50.0, 110.0, 135.0),
        "main_easm_domain": RegionSpec(10.0, 50.0, 100.0, 135.0),
    })

    @property
    def windows(self) -> Dict[str, Tuple[int, int]]:
        return self.windows_legacy if self.use_legacy_t3_window else self.windows_main

    @property
    def output_tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def output_maps_dir(self) -> Path:
        return self.output_dir / "maps"

    @property
    def output_figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def output_summary_dir(self) -> Path:
        return self.output_dir / "summary"
