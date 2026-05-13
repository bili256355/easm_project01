# -*- coding: utf-8 -*-
"""Settings for P/V850 latitudinal object-change audit v1_a.

This layer intentionally stays at the object level. It does not read or compute
V->P support/R2/pathway results. It diagnoses how precipitation and v850 fields
change across S3 -> T3 -> S4, with special care for multi-band precipitation
structures and possible higher-latitude escape beyond the fixed north box.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BoxSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LonSectorSpec:
    lon_min: float
    lon_max: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PVLatitudinalObjectChangeSettings:
    project_root: Path = Path(r"D:\easm_project01")

    output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_p_v_latitudinal_object_change_audit_v1_a"
    )

    foundation_preprocess_rel: str = "foundation/V1/outputs/baseline_smooth5_a/preprocess"

    windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "S3": (87, 106),
        "T3_early": (107, 112),
        "T3_full": (107, 117),
        "T3_late": (113, 117),
        "S4": (118, 154),
    })
    window_order: List[str] = field(default_factory=lambda: ["S3", "T3_early", "T3_full", "T3_late", "S4"])

    comparisons: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "T3_full_minus_S3": ("T3_full", "S3"),
        "T3_late_minus_T3_early": ("T3_late", "T3_early"),
        "S4_minus_T3_full": ("S4", "T3_full"),
    })

    regions: Dict[str, BoxSpec] = field(default_factory=lambda: {
        "north_northeast": BoxSpec(35.0, 50.0, 110.0, 135.0),
        "main_meiyu": BoxSpec(24.0, 35.0, 100.0, 125.0),
        "south_china": BoxSpec(18.0, 25.0, 105.0, 120.0),
        "scs": BoxSpec(10.0, 20.0, 105.0, 130.0),
        "south_china_scs": BoxSpec(10.0, 25.0, 105.0, 130.0),
        "main_easm_domain": BoxSpec(10.0, 50.0, 100.0, 135.0),
    })

    lon_sectors: Dict[str, LonSectorSpec] = field(default_factory=lambda: {
        "full_easm_lon": LonSectorSpec(100.0, 135.0),
        "west_sector": LonSectorSpec(100.0, 112.0),
        "mid_sector": LonSectorSpec(112.0, 123.0),
        "east_sector": LonSectorSpec(123.0, 135.0),
        "south_scs_ref": LonSectorSpec(105.0, 130.0),
    })

    # P multi-band detection defaults. They are conservative object diagnostics,
    # not physical mechanism gates.
    p_peak_relative_threshold: float = 0.15
    p_peak_absolute_threshold: float = 0.5
    p_min_peak_distance_deg: float = 3.0
    p_min_band_width_deg: float = 3.0
    p_edge_fraction_of_peak: float = 0.50
    profile_smooth_points: int = 3

    # Latband audit uses the actual data latitude extent. Bins are clipped to data.
    latband_step_deg: float = 5.0

    # Direction thresholds for tabular wording only.
    p_delta_epsilon: float = 1.0e-6
    v_delta_epsilon: float = 1.0e-6

    make_figures: bool = True
    use_cartopy: bool = True
    no_show: bool = True

    precip_aliases: Tuple[str, ...] = ("precip_smoothed", "precip", "precipitation", "pr", "rain", "P")
    v850_aliases: Tuple[str, ...] = ("v850_smoothed", "v850", "V850", "v", "meridional_wind")

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
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"


def settings_to_jsonable(settings: PVLatitudinalObjectChangeSettings) -> Dict[str, object]:
    out = dict(settings.__dict__)
    for key in ["project_root", "output_dir"]:
        out[key] = str(out[key])
    out["regions"] = {k: v.as_dict() for k, v in settings.regions.items()}
    out["lon_sectors"] = {k: v.as_dict() for k, v in settings.lon_sectors.items()}
    return out
