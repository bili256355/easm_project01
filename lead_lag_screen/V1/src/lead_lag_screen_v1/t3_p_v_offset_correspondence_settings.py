# -*- coding: utf-8 -*-
"""Settings for P/V850 offset-correspondence audit v1_b.

This layer is intentionally object-level only. It separates precipitation
climatological bands from precipitation change peaks/bands, and compares them
against V850 climatological structures and V850 change structures. It does not
compute V->P support, R2, lag/tau0, pathway, or causal evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LonSectorSpec:
    lon_min: float
    lon_max: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PVOffsetCorrespondenceSettings:
    project_root: Path = Path(r"D:\easm_project01")
    output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_p_v_offset_correspondence_audit_v1_b"
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

    lon_sectors: Dict[str, LonSectorSpec] = field(default_factory=lambda: {
        "full_easm_lon": LonSectorSpec(100.0, 135.0),
        "west_sector": LonSectorSpec(100.0, 112.0),
        "mid_sector": LonSectorSpec(112.0, 123.0),
        "east_sector": LonSectorSpec(123.0, 135.0),
        "south_scs_ref": LonSectorSpec(105.0, 130.0),
    })

    # Fixed offset candidates, in degrees latitude, used for pre-registered
    # offset checks. They are not free-fit locations.
    offset_degrees: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)

    # P climatological-band / change-band detection defaults.
    p_peak_relative_threshold: float = 0.15
    p_peak_absolute_threshold: float = 0.5
    p_delta_peak_relative_threshold: float = 0.20
    p_delta_peak_absolute_threshold: float = 0.25
    min_peak_distance_deg: float = 3.0
    min_band_width_deg: float = 3.0
    edge_fraction_of_peak: float = 0.50
    profile_smooth_points: int = 3

    # V change peak defaults.
    v_change_relative_threshold: float = 0.15
    v_change_absolute_threshold: float = 0.05
    gradient_change_relative_threshold: float = 0.15
    gradient_change_absolute_threshold: float = 0.02

    latband_step_deg: float = 5.0

    make_figures: bool = True
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


def settings_to_jsonable(settings: PVOffsetCorrespondenceSettings) -> Dict[str, object]:
    out = dict(settings.__dict__)
    for key in ["project_root", "output_dir"]:
        out[key] = str(out[key])
    out["lon_sectors"] = {k: v.as_dict() for k, v in settings.lon_sectors.items()}
    return out
