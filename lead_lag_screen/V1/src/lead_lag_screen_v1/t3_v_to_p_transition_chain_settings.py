# -*- coding: utf-8 -*-
"""Settings for T3 V->P transition-chain report v1_b.

This layer is a reporting / evidence-chain layer. It does not rerun the V1
lead-lag screen and does not change previous hard-evidence calculations. It
reads the previous field-explanation audit tables, recomputes observed support
maps only in memory for figures, and adds object-state / object-change context
for precipitation and v850.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RegionSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TransitionChainReportSettings:
    project_root: Path = Path(r"D:\easm_project01")

    output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_transition_chain_report_v1_b"
    )

    previous_field_explanation_output_dir: Path = Path(
        r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a"
    )

    foundation_preprocess_rel: str = "foundation/V1/outputs/baseline_smooth5_a/preprocess"
    foundation_indices_rel: str = "foundation/V1/outputs/baseline_smooth5_a/indices"

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

    v_components: List[str] = field(default_factory=lambda: [
        "V_strength",
        "V_NS_diff",
        "V_pos_centroid_lat",
    ])

    regions: Dict[str, RegionSpec] = field(default_factory=lambda: {
        "north_northeast": RegionSpec(35.0, 50.0, 110.0, 135.0),
        "main_meiyu": RegionSpec(24.0, 35.0, 100.0, 125.0),
        "south_china": RegionSpec(18.0, 25.0, 105.0, 120.0),
        "scs": RegionSpec(10.0, 20.0, 105.0, 130.0),
        "south_china_scs": RegionSpec(10.0, 25.0, 105.0, 130.0),
        "main_easm_domain": RegionSpec(10.0, 50.0, 100.0, 135.0),
    })

    north_region: str = "north_northeast"
    main_region: str = "main_meiyu"
    south_region: str = "south_china_scs"

    max_lag: int = 5
    support_metric_column: str = "region_mean_R2_map"
    support_map_lag_label: str = "positive_lag_max"

    # Thresholds are deliberately small and transparent. They control wording
    # such as increase / decrease / near_zero; they do not create physical claims.
    r2_delta_epsilon: float = 0.005
    object_delta_epsilon: float = 1.0e-10

    make_figures: bool = True
    use_cartopy: bool = True
    no_show: bool = True

    precip_aliases: Tuple[str, ...] = ("precip", "precipitation", "pr", "rain", "P")
    v850_aliases: Tuple[str, ...] = ("v850", "V850", "v", "meridional_wind")

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

    @property
    def previous_tables_dir(self) -> Path:
        return self.previous_field_explanation_output_dir / "tables"


def settings_to_jsonable(settings: TransitionChainReportSettings) -> Dict[str, object]:
    out = dict(settings.__dict__)
    for key in ["project_root", "output_dir", "previous_field_explanation_output_dir"]:
        out[key] = str(out[key])
    out["regions"] = {k: v.as_dict() for k, v in settings.regions.items()}
    return out
