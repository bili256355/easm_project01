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
class SourceStageConfig:
    project_root: Path = Path(r"D:\easm_project01")
    source_v6_output_tag: str = "mainline_v6_a"
    source_v6_1_output_tag: str = "mainline_v6_1_a"

    def stage_root(self) -> Path:
        return self.project_root / "stage_partition"

    def v6_root(self) -> Path:
        return self.stage_root() / "V6"

    def v6_1_root(self) -> Path:
        return self.stage_root() / "V6_1"

    def v6_output_root(self) -> Path:
        return self.v6_root() / "outputs" / self.source_v6_output_tag

    def v6_1_output_root(self) -> Path:
        return self.v6_1_root() / "outputs" / self.source_v6_1_output_tag

    def v6_update_log(self) -> Path:
        return self.v6_root() / "UPDATE_LOG_V6.md"

    def v6_1_update_log(self) -> Path:
        return self.v6_1_root() / "UPDATE_LOG_V6_1.md"

    def v6_bootstrap_summary_path(self) -> Path:
        return self.v6_output_root() / "candidate_points_bootstrap_summary.csv"

    def v6_1_windows_path(self) -> Path:
        return self.v6_1_output_root() / "derived_windows_registry.csv"


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
    # Required by V6 state_builder when V7 inherits the joint valid-day index.
    block_equal_contribution: bool = True
    trim_invalid_days: bool = True
    use_joint_valid_day_index: bool = True


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
class AcceptedWindowConfig:
    accepted_peak_days: tuple[int, ...] = (45, 81, 113, 160)
    excluded_candidate_days: tuple[int, ...] = (18, 96, 132, 135)
    bootstrap_match_threshold: float = 0.95
    require_logs: bool = True
    require_bootstrap_support: bool = True
    require_windows_from_v6_1: bool = True


@dataclass
class PeakLabelConfig:
    edge_margin_days: int = 1
    high_plateau_ratio: float = 0.90
    broad_peak_min_high_days: int = 3
    multi_peak_second_ratio: float = 0.90
    weak_peak_sharpness_threshold: float = 1.10


@dataclass
class AnalysisWindowConfig:
    buffer_days: int = 5
    edge_margin_days: int = 1


@dataclass
class BootstrapTimingConfig:
    n_bootstrap: int = 1000
    random_seed: int = 20260430
    use_same_resample_for_all_fields: bool = True
    progress_every: int = 50
    write_sample_long_tables: bool = True
    debug_n_bootstrap: int | None = None

    def effective_n_bootstrap(self) -> int:
        return int(self.debug_n_bootstrap) if self.debug_n_bootstrap is not None else int(self.n_bootstrap)


@dataclass
class TimingConfidenceConfig:
    support_radius_days: int = 2
    stable_min_support_modal_r2: float = 0.80
    stable_max_q95_width_days: float = 6.0
    stable_max_loyo_iqr_days: float = 3.0
    moderate_min_support_modal_r2: float = 0.60
    moderate_max_q95_width_days: float = 10.0
    fdr_alpha: float = 0.05
    boundary_support_threshold: float = 0.30


@dataclass
class PairwiseOrderConfig:
    # V7-c reads V7-b outputs and audits pairwise relative order.
    source_v7_b_output_tag: str = "field_transition_timing_v7_b"
    output_tag: str = "field_transition_pairwise_order_v7_c"

    sync_radius_days: int = 2
    order_1d_lag_days: int = 1
    order_2d_lag_days: int = 2

    robust_min_prob: float = 0.85
    robust_min_prob_by_1d: float = 0.75
    robust_min_median_abs_lag_days: float = 2.0
    robust_max_q_value: float = 0.05
    robust_min_loyo_prob: float = 0.75

    robust_censored_min_prob: float = 0.85
    robust_censored_min_prob_by_1d: float = 0.70
    robust_censored_min_median_abs_lag_days: float = 1.0
    robust_censored_max_q_value: float = 0.05
    robust_censored_min_loyo_prob: float = 0.70

    moderate_min_prob: float = 0.70
    moderate_min_prob_by_1d: float = 0.60
    moderate_min_median_abs_lag_days: float = 1.0
    moderate_max_q_value: float = 0.10
    moderate_min_loyo_prob: float = 0.60

    sync_min_prob_within_2d: float = 0.60
    sync_max_abs_median_lag_days: float = 1.0
    sync_max_prob_by_2d: float = 0.60

    conflict_bootstrap_prob_threshold: float = 0.70
    conflict_loyo_prob_threshold: float = 0.50

    # Used when deriving left/right censoring from V7-b observed/modal peaks.
    boundary_edge_margin_days: int = 1

    write_plots: bool = True

@dataclass
class IntervalTimingConfig:
    # V7-d changes the observation target from a single argmax peak day
    # to a transition active interval around each accepted window.
    output_tag: str = "field_transition_interval_timing_v7_d"
    analysis_radius_days: int = 15
    prepost_k_days: int = 5
    min_prepost_days: int = 3

    # Active interval extraction on locally normalized curves.
    active_min_norm_threshold: float = 0.50
    active_quantile_threshold: float = 0.75
    no_signal_epsilon: float = 1e-12

    # Candidate peak influence flags. These candidates are NOT reintroduced as
    # main windows; they are only flagged if the widened analysis window touches them.
    excluded_candidate_margin_days: int = 3
    boundary_margin_days: int = 1

    # Detector-vs-contrast agreement labels.
    metric_good_agreement_days: float = 3.0
    metric_moderate_agreement_days: float = 6.0

    # Interval relation thresholds.
    sync_center_radius_days: float = 2.0
    sync_overlap_ratio: float = 0.60
    shifted_overlap_max_ratio: float = 0.50

    separated_min_center_prob: float = 0.80
    separated_min_sep_prob: float = 0.50
    separated_min_median_lag_days: float = 2.0
    separated_min_loyo_center_prob: float = 0.70

    shifted_min_center_prob: float = 0.80
    shifted_min_shifted_prob: float = 0.65
    shifted_min_median_lag_days: float = 2.0
    shifted_min_loyo_center_prob: float = 0.70

    moderate_min_center_prob: float = 0.70
    moderate_min_shifted_prob: float = 0.50
    moderate_min_median_lag_days: float = 1.0
    moderate_min_loyo_center_prob: float = 0.60

    boundary_support_threshold: float = 0.30
    ambiguous_fraction_threshold: float = 0.50
    sync_window_fraction_threshold: float = 0.50
    partial_order_min_edges: int = 3

    # Reuse V7-b bootstrap year samples when available, so V7-b/V7-d are comparable.
    source_v7_b_output_tag: str = "field_transition_timing_v7_b"
    reuse_v7_b_resamples_if_available: bool = True
    write_sample_long_tables: bool = True
    write_plots: bool = True



@dataclass
class ProgressTimingConfig:
    # V7-e changes the observation target from change intensity to transition progress.
    output_tag: str = "field_transition_progress_timing_v7_e"
    analysis_radius_days: int = 15
    pre_offset_start_days: int = 15
    pre_offset_end_days: int = 8
    post_offset_start_days: int = 8
    post_offset_end_days: int = 15
    min_period_days: int = 3

    progress_clip_min: float = 0.0
    progress_clip_max: float = 1.0
    threshold_onset: float = 0.25
    threshold_midpoint: float = 0.50
    threshold_finish: float = 0.75
    stable_crossing_days: int = 2

    separation_clear_ratio: float = 1.5
    separation_moderate_ratio: float = 1.0
    separation_weak_ratio: float = 0.6
    min_transition_norm: float = 1e-10

    monotonic_corr_threshold: float = 0.60
    nonmonotonic_crossing_threshold: int = 3
    boundary_margin_days: int = 1
    excluded_candidate_margin_days: int = 3

    # Pairwise progress-order thresholds.
    sync_midpoint_radius_days: float = 2.0
    separated_min_finish_before_onset_prob: float = 0.50
    separated_min_midpoint_prob: float = 0.80
    separated_min_median_lag_days: float = 2.0
    separated_min_loyo_midpoint_prob: float = 0.70

    shifted_min_midpoint_prob: float = 0.80
    shifted_min_median_lag_days: float = 2.0
    shifted_min_loyo_midpoint_prob: float = 0.70

    moderate_min_midpoint_prob: float = 0.70
    moderate_min_median_lag_days: float = 1.0
    moderate_min_loyo_midpoint_prob: float = 0.60

    sync_min_prob: float = 0.60
    ambiguous_fraction_threshold: float = 0.50
    sync_window_fraction_threshold: float = 0.50
    partial_order_min_edges: int = 3
    boundary_support_threshold: float = 0.30

    # Reuse V7-b bootstrap year samples when available, so V7-b/V7-e are comparable.
    source_v7_b_output_tag: str = "field_transition_timing_v7_b"
    reuse_v7_b_resamples_if_available: bool = True
    write_sample_long_tables: bool = True
    write_plots: bool = True
@dataclass
class OutputConfig:
    output_tag: str = "field_transition_timing_v7_a"
    write_plots: bool = True


@dataclass
class StagePartitionV7Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    source: SourceStageConfig = field(default_factory=SourceStageConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    detector: RupturesWindowConfig = field(default_factory=RupturesWindowConfig)
    accepted_windows: AcceptedWindowConfig = field(default_factory=AcceptedWindowConfig)
    peak_labels: PeakLabelConfig = field(default_factory=PeakLabelConfig)
    analysis_window: AnalysisWindowConfig = field(default_factory=AnalysisWindowConfig)
    bootstrap: BootstrapTimingConfig = field(default_factory=BootstrapTimingConfig)
    timing_confidence: TimingConfidenceConfig = field(default_factory=TimingConfidenceConfig)
    pairwise_order: PairwiseOrderConfig = field(default_factory=PairwiseOrderConfig)
    interval_timing: IntervalTimingConfig = field(default_factory=IntervalTimingConfig)
    progress_timing: ProgressTimingConfig = field(default_factory=ProgressTimingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def layer_root(self) -> Path:
        return self.foundation.project_root / "stage_partition" / "V7"

    def output_root(self) -> Path:
        return self.layer_root() / "outputs" / self.output.output_tag

    def log_root(self) -> Path:
        return self.layer_root() / "logs" / self.output.output_tag

    def pairwise_output_root(self) -> Path:
        return self.layer_root() / "outputs" / self.pairwise_order.output_tag

    def pairwise_log_root(self) -> Path:
        return self.layer_root() / "logs" / self.pairwise_order.output_tag

    def v7_b_output_root(self) -> Path:
        return self.layer_root() / "outputs" / self.pairwise_order.source_v7_b_output_tag

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
            if hasattr(obj, "__dataclass_fields__"):
                return convert(asdict(obj))
            return obj

        return convert(asdict(self))

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
