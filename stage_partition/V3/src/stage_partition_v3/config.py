from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Literal
import json


@dataclass
class FoundationInputConfig:
    project_root: Path = Path(r"D:\easm_project01")
    foundation_layer: str = "foundation"
    foundation_version: str = "V1"
    preprocess_output_tag: str = "baseline_a"

    def foundation_root(self) -> Path:
        return self.project_root / self.foundation_layer / self.foundation_version

    def preprocess_dir(self) -> Path:
        return self.foundation_root() / "outputs" / self.preprocess_output_tag / "preprocess"

    def smoothed_fields_path(self) -> Path:
        return self.preprocess_dir() / "smoothed_fields.npz"


@dataclass
class ProfileGridConfig:
    lat_step_deg: float = 2.0
    p_lon_range: tuple[float, float] = (105.0, 125.0)
    p_lat_range: tuple[float, float] = (15.0, 39.0)
    v_lon_range: tuple[float, float] = (105.0, 125.0)
    v_lat_range: tuple[float, float] = (10.0, 30.0)
    h_lon_range: tuple[float, float] = (110.0, 140.0)
    h_lat_range: tuple[float, float] = (15.0, 35.0)
    je_lon_range: tuple[float, float] = (120.0, 150.0)
    je_lat_range: tuple[float, float] = (25.0, 45.0)
    jw_lon_range: tuple[float, float] = (80.0, 110.0)
    jw_lat_range: tuple[float, float] = (25.0, 45.0)


@dataclass
class StateVectorConfig:
    standardize: bool = True
    block_equal_contribution: bool = True


@dataclass
class RupturesWindowConfig:
    enabled: bool = True
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: Literal["fixed_n_bkps", "pen", "epsilon"] = "pen"
    fixed_n_bkps: Optional[int] = None
    pen: Optional[float] = 4.0
    epsilon: Optional[float] = None


@dataclass
class EdgeConfig:
    min_distance_days: int = 3
    prominence_quantile: float = 0.80
    nearest_peak_search_radius: int = 10


@dataclass
class WindowConstructionConfig:
    nearest_peak_search_radius: int = 10
    support_rel_prominence: float = 0.35
    support_floor_quantile: float = 0.60
    support_floor_ratio: float = 0.15
    merge_gap_days: int = 3
    min_band_width_days: int = 1
    valley_relief_tolerance: float = 0.10


@dataclass
class SupportAuditConfig:
    enable_param_path: bool = True
    enable_bootstrap: bool = True
    enable_permutation: bool = True
    bootstrap_reps: int = 100
    random_seed: int = 42
    ruptures_width_grid: list[int] = field(default_factory=lambda: [12, 16, 20, 24, 28])
    ruptures_pen_grid: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    permutation_reps: int = 200
    overlap_ratio_threshold: float = 0.50
    center_tolerance_days: int = 3
    match_mode: Literal["strict_window_overlap", "legacy_overlap_or_center"] = "strict_window_overlap"
    support_overlap_ratio_min: float = 0.50
    support_overlap_days_min: int = 1
    support_center_tolerance_is_auxiliary_only: bool = True


@dataclass
class RetentionAuditConfig:
    min_support_score: float = 0.30
    min_bootstrap_effective: int = 20
    min_param_path_hit_fraction: float = 0.50


@dataclass
class OutputConfig:
    output_tag: str = "mainline_v3_j"
    write_plots: bool = True
    emit_yearwise_outputs: bool = True


@dataclass
class YearwiseSupportConfig:
    near_peak_tolerance_days: int = 3
    overlap_days_min: int = 1
    overlap_ratio_min: float = 0.50
    headline_support_mode: str = "strict_overlap"
    emit_relation_audit: bool = True
    emit_support_summary: bool = True


@dataclass
class PointSignificanceConfig:
    enabled: bool = True
    null_reps: int = 500
    null_shift_mode: Literal["yearwise_circular_shift"] = "yearwise_circular_shift"
    local_window_days: int = 2
    bootstrap_presence_tolerance_days: int = 2
    neighbor_competition_radius_days: int = 5
    global_alpha: float = 0.05
    local_alpha: float = 0.10
    bootstrap_stable_min: float = 0.60
    param_stable_min: float = 0.60
    yearwise_near_min: float = 0.40
    null_compare_mode: str = "matched_peak_same_scale"
    null_global_peak_mode: str = "formal_point_ranked_peak"
    null_local_match_tolerance_days: int = 2
    peak_prominence_min: float = 0.0
    peak_distance_min_days: int = 3
    competition_tie_tolerance_score: float = 0.03
    competition_tie_tolerance_winfrac: float = 0.08
    competition_tie_tolerance_bootstrap: float = 0.08
    competition_tie_tolerance_yearwise: float = 0.08
    competition_decision_mode: str = "pairwise_closed_verbose"
    local_peak_exists_min_score: float = 0.0
    emit_local_null_joint_score: bool = True
    local_exist_alpha: float = 0.10
    local_use_conditional_test: bool = True
    matched_peak_max_center_offset_days: int = 2
    matched_peak_min_prominence: float = 0.0
    matched_peak_select_mode: str = 'closest_then_prominence_then_score'
    local_matched_stat_mode: str = 'zscore_vs_local_background'
    local_background_window_days: int = 7
    local_background_exclude_peak_core_days: int = 1
    use_matched_local_test: bool = True
    emit_local_match_quality: bool = True
    strict_tier_use_global_fwer_as_headline: bool = False
    strict_tier_use_local_matched_test_as_headline: bool = True


@dataclass
class BTrackPointConfig:
    enabled: bool = True
    strict_match_radius_days: int = 2
    match_radius_days: int = 4
    near_radius_days: int = 8
    use_existing_point_stability_outputs: bool = True


@dataclass
class BTrackPointJudgementConfig:
    formal_robust_match_rate_min: float = 0.95
    formal_supported_match_rate_min: float = 0.85
    formal_caution_match_rate_min: float = 0.75
    formal_robust_ambiguous_rate_max: float = 0.10
    formal_supported_ambiguous_rate_max: float = 0.20
    formal_robust_offset_iqr_max: float = 3.0
    formal_supported_offset_iqr_max: float = 6.0
    caution_yearwise_exact_min: float = 0.20
    caution_path_presence_min: float = 0.80
    neighbor_ambiguous_rate_min: float = 0.30
    neighbor_weak_match_rate_max: float = 0.60
    neighbor_high_tie_rate_min: float = 0.30


@dataclass
class StagePartitionV3Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateVectorConfig = field(default_factory=StateVectorConfig)
    ruptures_window: RupturesWindowConfig = field(default_factory=RupturesWindowConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    window_construction: WindowConstructionConfig = field(default_factory=WindowConstructionConfig)
    support: SupportAuditConfig = field(default_factory=SupportAuditConfig)
    retention: RetentionAuditConfig = field(default_factory=RetentionAuditConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    yearwise_support: YearwiseSupportConfig = field(default_factory=YearwiseSupportConfig)
    point_significance: PointSignificanceConfig = field(default_factory=PointSignificanceConfig)
    btrack_point: BTrackPointConfig = field(default_factory=BTrackPointConfig)
    btrack_point_judgement: BTrackPointJudgementConfig = field(default_factory=BTrackPointJudgementConfig)

    def layer_root(self) -> Path:
        return self.foundation.project_root / "stage_partition" / "V3"

    def output_root(self) -> Path:
        return self.layer_root() / "outputs" / self.output.output_tag

    def log_root(self) -> Path:
        return self.layer_root() / "logs" / self.output.output_tag

    def to_dict(self) -> dict:
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return [convert(x) for x in obj]
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if hasattr(obj, '__dataclass_fields__'):
                return convert(asdict(obj))
            return obj
        return convert(self)

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
