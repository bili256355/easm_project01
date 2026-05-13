from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
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
class StateBuilderConfig:
    standardize: bool = True
    block_equal_contribution: bool = True
    trim_invalid_days: bool = True

@dataclass
class RupturesWindowConfig:
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: str = "pen"
    fixed_n_bkps: int | None = None
    pen: float | None = 4.0
    epsilon: float | None = None
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10

@dataclass
class PointMatchingConfig:
    strict_match_radius_days: int = 2
    match_radius_days: int = 4
    near_radius_days: int = 8
    ambiguous_score_tie_tol: float = 1e-8
    ambiguous_day_tie_tol: int = 0

@dataclass
class CompetitionConfig:
    neighbor_radius_days: int = 5
    score_tie_tolerance: float = 0.03
    bootstrap_tie_tolerance: float = 0.10
    yearwise_tie_tolerance: float = 0.10
    require_support_for_non_tie: bool = True

@dataclass
class PointAuditUniverseConfig:
    enabled: bool = True
    radius_days: int = 4
    weak_peak_min_prominence: float = 0.0
    dedup_same_day_candidates: bool = True
    audit_universe_mode: str = "headline_centered_local_neighbor_only"

@dataclass
class BootstrapConfig:
    n_bootstrap: int = 200
    random_seed: int = 42
    progress: bool = True

@dataclass
class YearwiseConfig:
    progress: bool = True
    strict_exact_equivalent: bool = True
    exact_hit_max_abs_offset_days: int = 2
    pair_competition_mode: str = "conservative_comparable_years_only"
    pair_requires_both_detected: bool = True

@dataclass
class ParameterPathConfig:
    progress: bool = True
    bandwidth_values: list[int] = field(default_factory=lambda: [18, 20, 22])
    pen_values: list[float] = field(default_factory=lambda: [3.5, 4.0, 4.5])
    parameter_path_scope: str = "detector_effective_only"

@dataclass
class BTrackJudgementConfig:
    formal_robust_match_rate_min: float = 0.95
    formal_supported_match_rate_min: float = 0.85
    formal_caution_match_rate_min: float = 0.75
    formal_robust_ambiguous_rate_max: float = 0.10
    formal_supported_ambiguous_rate_max: float = 0.20
    formal_robust_offset_iqr_max: float = 3.0
    formal_supported_offset_iqr_max: float = 6.0
    formal_robust_yearwise_strict_min: float = 0.20
    formal_supported_yearwise_total_support_min: float = 0.45
    formal_caution_yearwise_total_support_min: float = 0.20
    caution_path_presence_min: float = 0.80
    neighbor_ambiguous_rate_min: float = 0.30
    neighbor_weak_match_rate_max: float = 0.60
    neighbor_high_tie_rate_min: float = 0.30

@dataclass
class OutputConfig:
    output_tag: str = "mainline_v4_d"
    write_plots: bool = False

@dataclass
class StagePartitionV4Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    detector: RupturesWindowConfig = field(default_factory=RupturesWindowConfig)
    matching: PointMatchingConfig = field(default_factory=PointMatchingConfig)
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    point_audit_universe: PointAuditUniverseConfig = field(default_factory=PointAuditUniverseConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    yearwise: YearwiseConfig = field(default_factory=YearwiseConfig)
    parameter_path: ParameterPathConfig = field(default_factory=ParameterPathConfig)
    btrack_judgement: BTrackJudgementConfig = field(default_factory=BTrackJudgementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    def layer_root(self) -> Path:
        return self.foundation.project_root / "stage_partition" / "V4"
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
