from __future__ import annotations

"""
V10.3 peak discovery sensitivity test.

Scope:
  - joint_all free-season peak discovery sensitivity
  - object-native free-season peak discovery sensitivity for P/V/H/Je/Jw
  - candidate peak, bootstrap support, candidate band, derived window, and lineage mapping changes

Base layers:
  - V10.1 joint main-window reproduction output is used as the joint lineage baseline/reference.
  - V10.2 object-native peak discovery output is used as object-native baseline/reference.

Important boundary:
  This module does not perform physical interpretation, pair-order analysis, or accepted-window re-decision.
  It does not import V6/V6_1/V7/V9 modules. It dynamically loads the local V10.2 implementation
  as the already verified free-discovery semantic base.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime, timezone
import copy
import importlib.util
import json
import os
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Basic utilities
# =============================================================================

OBJECT_ORDER = ["P", "V", "H", "Je", "Jw"]
SCOPE_SPECS = [("joint_all", OBJECT_ORDER)] + [(f"{obj}_only", [obj]) for obj in OBJECT_ORDER]
OUTPUT_TAG = "peak_discovery_sensitivity_v10_3"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def clean_output_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for sub in ["by_scope", "cross_scope", "lineage_mapping", "audit", "figures"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def day_index_to_month_day(day_index: int) -> str:
    mdays = [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]
    d = int(day_index)
    for month, nday in mdays:
        if d < nday:
            return f"{month:02d}-{d + 1:02d}"
        d -= nday
    return f"overflow+{d}"


def _as_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _as_int_or_nan(x: Any):
    try:
        if pd.isna(x):
            return np.nan
        return int(round(float(x)))
    except Exception:
        return np.nan


def _normalize_boolish(x: Any):
    """Return a pandas-nullable boolean scalar without object-dtype fillna.

    This helper avoids pandas FutureWarning messages triggered by
    object-dtype Series.fillna(False).astype(bool).  For the current V10.3
    outputs the source column is already boolean/NaN, so this preserves the
    existing count semantics while making the dtype conversion explicit.
    """
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isnan(float(x)):
            return pd.NA
        return bool(float(x))
    if isinstance(x, str):
        z = x.strip().lower()
        if z in {"true", "t", "1", "yes", "y"}:
            return True
        if z in {"false", "f", "0", "no", "n", ""}:
            return False
    return bool(x)


def _safe_bool_true_count(series: pd.Series | None) -> int:
    if series is None:
        return 0
    s = pd.Series(series)
    if s.empty:
        return 0
    b = s.map(_normalize_boolish).astype("boolean").fillna(False)
    return int(b.sum())


# =============================================================================
# Dynamic loading of the verified V10.2 semantic base
# =============================================================================


def load_v10_2_module(bundle_root: Path):
    v10_root = bundle_root.parent
    module_path = v10_root / "v10.2" / "src" / "object_native_peak_discovery_v10_2.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing V10.2 semantic base: {module_path}. Run/install V10.2 before V10.3."
        )
    spec = importlib.util.spec_from_file_location("object_native_peak_discovery_v10_2_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V10.2 module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses expects dynamically loaded modules to be registered
    # in sys.modules before @dataclass classes are processed. Without this,
    # dataclasses can fail with: AttributeError: 'NoneType' object has no attribute '__dict__'.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SensitivityConfigSpec:
    config_id: str
    sensitivity_group: str
    changed_factor: str
    changed_value: str
    input_source: str = "baseline_smooth9"
    detector_width: int | None = None
    detector_pen: float | None = None
    local_peak_min_distance_days: int | None = None
    strict_match_days: int | None = None
    match_days: int | None = None
    near_days: int | None = None
    band_floor_quantile: float | None = None
    band_prominence_ratio: float | None = None
    band_max_half_width: int | None = None
    merge_gap_days: int | None = None
    close_neighbor_exemption_days: int | None = None
    significant_peak_threshold: float | None = None
    bootstrap_required: bool = True


@dataclass
class V10_3Settings:
    output_tag: str = OUTPUT_TAG
    bootstrap_mode: str = "baseline_and_match"  # baseline_and_match | all | none
    n_bootstrap: int = 1000
    progress: bool = True
    include_smoothing_group: bool = True
    include_smooth5_internal_group: bool = True
    smooth5_output_tag: str = "baseline_smooth5_a"

    def resolved_n_bootstrap(self) -> int:
        if os.environ.get("V10_3_DEBUG_N_BOOTSTRAP"):
            return int(os.environ["V10_3_DEBUG_N_BOOTSTRAP"])
        if os.environ.get("V10_3_N_BOOTSTRAP"):
            return int(os.environ["V10_3_N_BOOTSTRAP"])
        return int(self.n_bootstrap)

    def resolved_bootstrap_mode(self) -> str:
        return os.environ.get("V10_3_BOOTSTRAP_MODE", self.bootstrap_mode).strip().lower()

    def resolved_include_smoothing_group(self) -> bool:
        env = os.environ.get("V10_3_INCLUDE_SMOOTHING_GROUP")
        if env is None:
            return bool(self.include_smoothing_group)
        return env.strip().lower() in {"1", "true", "yes", "y", "on"}

    def resolved_smooth5_output_tag(self) -> str:
        return os.environ.get("V10_3_SMOOTH5_OUTPUT_TAG", self.smooth5_output_tag).strip()

    def resolved_include_smooth5_internal_group(self) -> bool:
        env = os.environ.get("V10_3_INCLUDE_SMOOTH5_INTERNAL_GROUP")
        if env is None:
            return bool(self.include_smooth5_internal_group)
        return env.strip().lower() in {"1", "true", "yes", "y", "on"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sensitivity_specs(settings: V10_3Settings) -> list[SensitivityConfigSpec]:
    # This is deliberately single-factor, not a mixed factorial grid.
    base = SensitivityConfigSpec("BASELINE", "baseline", "none", "baseline")
    specs = [
        base,
        SensitivityConfigSpec("DET_WIDTH_16", "detector_width", "detector_width", "16", detector_width=16),
        SensitivityConfigSpec("DET_WIDTH_24", "detector_width", "detector_width", "24", detector_width=24),
        SensitivityConfigSpec("DET_PEN_3", "detector_penalty", "detector_penalty", "3.0", detector_pen=3.0),
        SensitivityConfigSpec("DET_PEN_5", "detector_penalty", "detector_penalty", "5.0", detector_pen=5.0),
        SensitivityConfigSpec("PEAK_DISTANCE_5", "local_peak_distance", "local_peak_min_distance_days", "5", local_peak_min_distance_days=5),
        SensitivityConfigSpec("PEAK_DISTANCE_7", "local_peak_distance", "local_peak_min_distance_days", "7", local_peak_min_distance_days=7),
        SensitivityConfigSpec("MATCH_RADIUS_1_4_7", "bootstrap_match_radius", "strict_match/match/near", "1/4/7", strict_match_days=1, match_days=4, near_days=7),
        SensitivityConfigSpec("MATCH_RADIUS_3_6_9", "bootstrap_match_radius", "strict_match/match/near", "3/6/9", strict_match_days=3, match_days=6, near_days=9),
        SensitivityConfigSpec("BAND_Q30", "candidate_band", "band_floor_quantile", "0.30", band_floor_quantile=0.30),
        SensitivityConfigSpec("BAND_Q40", "candidate_band", "band_floor_quantile", "0.40", band_floor_quantile=0.40),
        SensitivityConfigSpec("BAND_PROM_045", "candidate_band", "band_prominence_ratio", "0.45", band_prominence_ratio=0.45),
        SensitivityConfigSpec("BAND_PROM_055", "candidate_band", "band_prominence_ratio", "0.55", band_prominence_ratio=0.55),
        SensitivityConfigSpec("BAND_MAX_HW_8", "candidate_band", "band_max_half_width", "8", band_max_half_width=8),
        SensitivityConfigSpec("BAND_MAX_HW_12", "candidate_band", "band_max_half_width", "12", band_max_half_width=12),
        SensitivityConfigSpec("MERGE_GAP_0", "window_merge", "merge_gap_days", "0", merge_gap_days=0),
        SensitivityConfigSpec("MERGE_GAP_2", "window_merge", "merge_gap_days", "2", merge_gap_days=2),
        SensitivityConfigSpec("CLOSE_NEIGHBOR_3", "window_merge", "close_neighbor_exemption_days", "3", close_neighbor_exemption_days=3),
        SensitivityConfigSpec("CLOSE_NEIGHBOR_5", "window_merge", "close_neighbor_exemption_days", "5", close_neighbor_exemption_days=5),
        SensitivityConfigSpec("PROTECT_THR_090", "window_merge", "significant_peak_threshold", "0.90", significant_peak_threshold=0.90),
    ]
    if settings.resolved_include_smoothing_group():
        # SMOOTH_INPUT_5D is the smooth5 baseline compared against the smooth9 BASELINE.
        # Smooth5-internal perturbations below are compared against this config, not against BASELINE,
        # so they answer: within 5-day smoothed input, which algorithm parameter is most sensitive?
        specs.append(
            SensitivityConfigSpec(
                "SMOOTH_INPUT_5D",
                "smooth_input",
                "smoothed_fields_input",
                "smooth5",
                input_source="smooth5",
            )
        )
        if settings.resolved_include_smooth5_internal_group():
            specs.extend([
                SensitivityConfigSpec("SMOOTH5_DET_WIDTH_16", "smooth5_detector_width", "detector_width", "16", input_source="smooth5", detector_width=16),
                SensitivityConfigSpec("SMOOTH5_DET_WIDTH_24", "smooth5_detector_width", "detector_width", "24", input_source="smooth5", detector_width=24),
                SensitivityConfigSpec("SMOOTH5_DET_PEN_3", "smooth5_detector_penalty", "detector_penalty", "3.0", input_source="smooth5", detector_pen=3.0),
                SensitivityConfigSpec("SMOOTH5_DET_PEN_5", "smooth5_detector_penalty", "detector_penalty", "5.0", input_source="smooth5", detector_pen=5.0),
                SensitivityConfigSpec("SMOOTH5_PEAK_DISTANCE_5", "smooth5_local_peak_distance", "local_peak_min_distance_days", "5", input_source="smooth5", local_peak_min_distance_days=5),
                SensitivityConfigSpec("SMOOTH5_PEAK_DISTANCE_7", "smooth5_local_peak_distance", "local_peak_min_distance_days", "7", input_source="smooth5", local_peak_min_distance_days=7),
                SensitivityConfigSpec("SMOOTH5_MATCH_RADIUS_1_4_7", "smooth5_bootstrap_match_radius", "strict_match/match/near", "1/4/7", input_source="smooth5", strict_match_days=1, match_days=4, near_days=7),
                SensitivityConfigSpec("SMOOTH5_MATCH_RADIUS_3_6_9", "smooth5_bootstrap_match_radius", "strict_match/match/near", "3/6/9", input_source="smooth5", strict_match_days=3, match_days=6, near_days=9),
                SensitivityConfigSpec("SMOOTH5_BAND_Q30", "smooth5_candidate_band", "band_floor_quantile", "0.30", input_source="smooth5", band_floor_quantile=0.30),
                SensitivityConfigSpec("SMOOTH5_BAND_Q40", "smooth5_candidate_band", "band_floor_quantile", "0.40", input_source="smooth5", band_floor_quantile=0.40),
                SensitivityConfigSpec("SMOOTH5_BAND_PROM_045", "smooth5_candidate_band", "band_prominence_ratio", "0.45", input_source="smooth5", band_prominence_ratio=0.45),
                SensitivityConfigSpec("SMOOTH5_BAND_PROM_055", "smooth5_candidate_band", "band_prominence_ratio", "0.55", input_source="smooth5", band_prominence_ratio=0.55),
                SensitivityConfigSpec("SMOOTH5_BAND_MAX_HW_8", "smooth5_candidate_band", "band_max_half_width", "8", input_source="smooth5", band_max_half_width=8),
                SensitivityConfigSpec("SMOOTH5_BAND_MAX_HW_12", "smooth5_candidate_band", "band_max_half_width", "12", input_source="smooth5", band_max_half_width=12),
                SensitivityConfigSpec("SMOOTH5_MERGE_GAP_0", "smooth5_window_merge", "merge_gap_days", "0", input_source="smooth5", merge_gap_days=0),
                SensitivityConfigSpec("SMOOTH5_MERGE_GAP_2", "smooth5_window_merge", "merge_gap_days", "2", input_source="smooth5", merge_gap_days=2),
                SensitivityConfigSpec("SMOOTH5_CLOSE_NEIGHBOR_3", "smooth5_window_merge", "close_neighbor_exemption_days", "3", input_source="smooth5", close_neighbor_exemption_days=3),
                SensitivityConfigSpec("SMOOTH5_CLOSE_NEIGHBOR_5", "smooth5_window_merge", "close_neighbor_exemption_days", "5", input_source="smooth5", close_neighbor_exemption_days=5),
                SensitivityConfigSpec("SMOOTH5_PROTECT_THR_090", "smooth5_window_merge", "significant_peak_threshold", "0.90", input_source="smooth5", significant_peak_threshold=0.90),
            ])
    return specs


def should_run_bootstrap(spec: SensitivityConfigSpec, mode: str) -> bool:
    mode = mode.lower()
    if mode == "none":
        return False
    if mode == "all":
        return True
    # Default V10.3 closure mode: run bootstrap for baseline, match-radius,
    # smooth-input baseline, and the minimal detector-width configs that are
    # required to audit the dominant candidate-peak sensitivity source.
    minimal_detector_width = {"detector_width", "smooth5_detector_width"}
    match_groups = {"bootstrap_match_radius", "smooth_input", "smooth5_bootstrap_match_radius"}
    if mode in {"baseline_and_match", "baseline_match_width", "baseline_match_detector_width"}:
        return (
            spec.config_id == "BASELINE"
            or spec.sensitivity_group in match_groups
            or spec.sensitivity_group in minimal_detector_width
        )
    if mode == "baseline_only":
        return spec.config_id == "BASELINE"
    # Safe default: baseline + match-radius + minimal detector-width bootstrap.
    return spec.config_id == "BASELINE" or spec.sensitivity_group in match_groups or spec.sensitivity_group in minimal_detector_width


def make_v10_2_settings(v10_2, base_settings, spec: SensitivityConfigSpec, v10_3_settings: V10_3Settings, do_bootstrap: bool):
    s = copy.deepcopy(base_settings)
    s.output_tag = OUTPUT_TAG
    # detector changes
    if spec.detector_width is not None:
        s.detector.width = int(spec.detector_width)
    if spec.detector_pen is not None:
        s.detector.pen = float(spec.detector_pen)
    if spec.local_peak_min_distance_days is not None:
        s.detector.local_peak_min_distance_days = int(spec.local_peak_min_distance_days)
    # bootstrap match radius changes
    s.bootstrap.n_bootstrap = int(v10_3_settings.resolved_n_bootstrap()) if do_bootstrap else 0
    s.bootstrap.progress = bool(v10_3_settings.progress)
    if spec.strict_match_days is not None:
        s.bootstrap.strict_match_max_abs_offset_days = int(spec.strict_match_days)
    if spec.match_days is not None:
        s.bootstrap.match_max_abs_offset_days = int(spec.match_days)
    if spec.near_days is not None:
        s.bootstrap.near_max_abs_offset_days = int(spec.near_days)
    # band/window changes
    if spec.band_floor_quantile is not None:
        s.band.peak_floor_quantile = float(spec.band_floor_quantile)
    if spec.band_prominence_ratio is not None:
        s.band.prominence_ratio_threshold = float(spec.band_prominence_ratio)
    if spec.band_max_half_width is not None:
        s.band.max_band_half_width_days = int(spec.band_max_half_width)
    if spec.merge_gap_days is not None:
        s.band.merge_gap_days = int(spec.merge_gap_days)
    if spec.close_neighbor_exemption_days is not None:
        s.band.close_neighbor_exemption_days = int(spec.close_neighbor_exemption_days)
    if spec.significant_peak_threshold is not None:
        s.band.significant_peak_threshold = float(spec.significant_peak_threshold)
    return s


# =============================================================================
# Baseline/reference loading
# =============================================================================


def read_v10_1_lineage(v10_root: Path) -> pd.DataFrame:
    p = v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "lineage" / "joint_main_window_lineage_v10_1.csv"
    return safe_read_csv(p)


def read_v10_2_object_catalog(v10_root: Path) -> pd.DataFrame:
    p = v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "cross_object" / "object_native_candidate_catalog_all_objects_v10_2.csv"
    return safe_read_csv(p)


def read_v10_2_object_mapping(v10_root: Path) -> pd.DataFrame:
    p = v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "lineage_mapping" / "object_candidate_to_joint_lineage_v10_2.csv"
    return safe_read_csv(p)


def read_v10_window_conditioned_selection(v10_root: Path) -> pd.DataFrame:
    p = v10_root / "outputs" / "peak_subpeak_reproduce_v10_a" / "cross_window" / "main_window_selection_all_windows.csv"
    return safe_read_csv(p)


def resolve_smooth5_fields_path(base_settings, settings: V10_3Settings) -> Path:
    """Resolve the 5-day smoothed input path for smooth-input sensitivity.

    The historical project has used slightly different output-tag names across
    iterations.  Prefer an explicit environment override, then the configured
    tag, then a short list of common fallbacks.  This function only resolves the
    path; it does not silently reinterpret a missing smooth5 file as baseline.
    """
    env = os.environ.get("V10_3_SMOOTH5_FIELDS")
    if env:
        return Path(env)
    f = base_settings.foundation
    candidates = []
    for tag in [settings.resolved_smooth5_output_tag(), "baseline_smooth5_a", "baseline_smooth5", "smooth5_a"]:
        if not tag:
            continue
        path = (
            f.project_root
            / f.foundation_layer
            / f.foundation_version
            / "outputs"
            / tag
            / "preprocess"
            / "smoothed_fields.npz"
        )
        if path not in candidates:
            candidates.append(path)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_input_profile_sources(v10_2, base_settings, settings: V10_3Settings) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Load baseline smooth9 and optional smooth5 profile inputs.

    Output source names are intentionally semantic rather than path-derived:
    - baseline_smooth9: V10/V10.1/V10.2 baseline input
    - smooth5: 5-day smoothed input for input-source sensitivity
    """
    sources: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def _load_source(name: str, path: Path, required: bool) -> None:
        row = {
            "input_source": name,
            "smoothed_fields_path": str(path),
            "required": bool(required),
            "exists": bool(path.exists()),
            "status": "missing",
            "n_years": np.nan,
            "n_days": np.nan,
            "error": "",
        }
        if not path.exists():
            msg = (
                f"Missing {name} smoothed_fields.npz: {path}. "
                "Set V10_3_SMOOTH5_FIELDS to the correct smooth5 npz path if needed."
            )
            row["error"] = msg
            rows.append(row)
            if required:
                raise FileNotFoundError(msg)
            return
        try:
            smoothed = v10_2.load_smoothed_fields(path)
            profiles = v10_2.build_profiles(smoothed, base_settings.profile)
            years = smoothed.get("years", np.arange(next(iter(profiles.values())).raw_cube.shape[0]))
            sample = next(iter(profiles.values())).raw_cube
            row.update({
                "status": "loaded",
                "n_years": int(sample.shape[0]),
                "n_days": int(sample.shape[1]),
            })
            sources[name] = {
                "input_source": name,
                "smoothed_fields_path": str(path),
                "smoothed": smoothed,
                "profiles": profiles,
                "years": years,
                "status": "loaded",
            }
            rows.append(row)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            rows.append(row)
            if required:
                raise

    baseline_path = base_settings.foundation.smoothed_fields_path()
    _load_source("baseline_smooth9", baseline_path, required=True)
    if settings.resolved_include_smoothing_group():
        _load_source("smooth5", resolve_smooth5_fields_path(base_settings, settings), required=True)
    return sources, pd.DataFrame(rows)


# =============================================================================
# Discovery run per scope/config
# =============================================================================


def _scope_object_label(scope: str, object_names: list[str]) -> str:
    return "joint_all" if scope == "joint_all" else object_names[0]


def run_scope_config(v10_2, profiles, years, source_info: dict[str, Any], scope: str, object_names: list[str], spec: SensitivityConfigSpec, base_settings, v10_3_settings: V10_3Settings, bootstrap_mode: str) -> dict[str, pd.DataFrame | dict[str, Any]]:
    do_boot = should_run_bootstrap(spec, bootstrap_mode)
    settings = make_v10_2_settings(v10_2, base_settings, spec, v10_3_settings, do_boot)
    state = v10_2.build_scope_state_matrix(profiles, object_names, settings.state)
    det = v10_2.run_point_detector(state["state_matrix"][state["valid_day_mask"], :], state["valid_day_index"], settings.detector)
    registry = v10_2.build_candidate_registry(det["local_peaks_df"], det["primary_points_df"], source_run_tag=f"v10_3_{scope}_{spec.config_id}")

    # Bootstrap support can be very expensive. It is run according to V10_3_BOOTSTRAP_MODE.
    if do_boot:
        boot = v10_2.run_bootstrap_support_for_scope(
            profiles,
            object_names,
            years,
            registry,
            settings,
            desc=f"V10.3 {scope} {spec.config_id} bootstrap",
        )
        boot_summary = boot["summary_df"].copy()
        if scope == "joint_all":
            # Keep joint support class separate from object support class.
            frac = pd.to_numeric(boot_summary.get("bootstrap_match_fraction", np.nan), errors="coerce")
            boot_summary["joint_support_class"] = np.where(frac >= 0.95, "joint_strong_candidate",
                                                    np.where(frac >= 0.80, "joint_candidate",
                                                    np.where(frac >= 0.50, "joint_weak_candidate", "joint_unstable_candidate")))
        else:
            boot_summary = v10_2.add_object_support_class(boot_summary)
        boot_records = boot["records_df"].copy()
        boot_meta = boot["meta_df"].copy()
        bootstrap_status = "run"
    else:
        cols = [
            "candidate_id", "point_day", "bootstrap_strict_fraction", "bootstrap_match_fraction",
            "bootstrap_near_fraction", "bootstrap_no_match_fraction", "bootstrap_n", "support_class"
        ]
        boot_summary = pd.DataFrame(columns=cols)
        boot_records = pd.DataFrame()
        boot_meta = pd.DataFrame()
        bootstrap_status = "not_run_by_mode"

    bands = v10_2.build_candidate_point_bands(registry, det["profile"], settings.band)
    windows, membership = v10_2.merge_candidate_bands_into_windows(bands, boot_summary, settings.band)
    try:
        unc_summary, ret_dist = v10_2.summarize_window_uncertainty(windows, membership, boot_records, boot_summary, settings.uncertainty)
    except Exception:
        unc_summary, ret_dist = pd.DataFrame(), pd.DataFrame()

    detector_scores = det["profile"].rename("detector_score").reset_index().rename(columns={"index": "day"})
    for df in [registry, bands, windows, membership, detector_scores, boot_summary, boot_records, boot_meta, unc_summary, ret_dist]:
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            df.insert(0, "config_id", spec.config_id)
            df.insert(0, "scope", scope)
            df.insert(1, "object", _scope_object_label(scope, object_names))
            df["sensitivity_group"] = spec.sensitivity_group
            df["changed_factor"] = spec.changed_factor
            df["changed_value"] = spec.changed_value
            df["input_source"] = spec.input_source
            df["smoothed_fields_path"] = str(source_info.get("smoothed_fields_path", ""))
            df["bootstrap_status"] = bootstrap_status

    meta = {
        "scope": scope,
        "object_names": object_names,
        "config_id": spec.config_id,
        "sensitivity_group": spec.sensitivity_group,
        "changed_factor": spec.changed_factor,
        "changed_value": spec.changed_value,
        "input_source": spec.input_source,
        "smoothed_fields_path": str(source_info.get("smoothed_fields_path", "")),
        "n_candidates": int(len(registry)),
        "candidate_days": registry["point_day"].astype(int).tolist() if not registry.empty else [],
        "n_derived_windows": int(len(windows)),
        "derived_window_main_peak_days": windows["main_peak_day"].dropna().astype(int).tolist() if not windows.empty else [],
        "bootstrap_status": bootstrap_status,
        "n_bootstrap": int(settings.bootstrap.n_bootstrap) if do_boot else 0,
        "detector_width": int(settings.detector.width),
        "detector_pen": float(settings.detector.pen) if settings.detector.pen is not None else None,
        "local_peak_min_distance_days": int(settings.detector.local_peak_min_distance_days),
        "strict_match_days": int(settings.bootstrap.strict_match_max_abs_offset_days),
        "match_days": int(settings.bootstrap.match_max_abs_offset_days),
        "near_days": int(settings.bootstrap.near_max_abs_offset_days),
        "band_floor_quantile": float(settings.band.peak_floor_quantile),
        "band_prominence_ratio": float(settings.band.prominence_ratio_threshold),
        "band_max_half_width": int(settings.band.max_band_half_width_days),
        "merge_gap_days": int(settings.band.merge_gap_days),
        "close_neighbor_exemption_days": int(settings.band.close_neighbor_exemption_days),
        "significant_peak_threshold": float(settings.band.significant_peak_threshold),
    }
    return {
        "registry": registry,
        "detector_scores": detector_scores,
        "boot_summary": boot_summary,
        "boot_records": boot_records,
        "boot_meta": boot_meta,
        "bands": bands,
        "windows": windows,
        "membership": membership,
        "uncertainty_summary": unc_summary,
        "return_day_distribution": ret_dist,
        "meta": meta,
    }


# =============================================================================
# Comparison helpers
# =============================================================================


def candidate_match_class(delta: float) -> str:
    if not np.isfinite(delta):
        return "MISSING_BASELINE_CANDIDATE"
    ad = abs(float(delta))
    if ad == 0:
        return "SAME_DAY"
    if ad <= 2:
        return "STRICT_SHIFT_LE_2D"
    if ad <= 5:
        return "MATCH_SHIFT_LE_5D"
    if ad <= 8:
        return "NEAR_SHIFT_LE_8D"
    return "SHIFT_GT_8D"


def nearest_row_by_day(df: pd.DataFrame, day_col: str, day: int) -> tuple[pd.Series | None, float]:
    if df is None or df.empty or day_col not in df.columns:
        return None, np.nan
    vals = pd.to_numeric(df[day_col], errors="coerce")
    if vals.notna().sum() == 0:
        return None, np.nan
    idx = (vals - int(day)).abs().idxmin()
    return df.loc[idx], float(vals.loc[idx] - int(day))


def compare_candidates(baseline: pd.DataFrame, perturbed: pd.DataFrame, scope: str, spec: SensitivityConfigSpec, reference_config_id: str = "BASELINE") -> pd.DataFrame:
    rows = []
    b = baseline.copy() if baseline is not None else pd.DataFrame()
    p = perturbed.copy() if perturbed is not None else pd.DataFrame()
    matched_perturbed_ids = set()
    if b.empty:
        return pd.DataFrame()
    for _, br in b.sort_values("point_day").iterrows():
        bday = int(br["point_day"])
        pr, delta = nearest_row_by_day(p, "point_day", bday)
        if pr is not None:
            matched_perturbed_ids.add(str(pr.get("candidate_id", "")))
            pday = int(pr["point_day"])
            pscore = _as_float(pr.get("peak_score", np.nan))
            pcid = str(pr.get("candidate_id", ""))
        else:
            pday, pscore, pcid = np.nan, np.nan, ""
        bscore = _as_float(br.get("peak_score", np.nan))
        rows.append({
            "scope": scope,
            "object": "joint_all" if scope == "joint_all" else scope.replace("_only", ""),
            "config_id": spec.config_id,
            "reference_config_id": reference_config_id,
            "sensitivity_group": spec.sensitivity_group,
            "changed_factor": spec.changed_factor,
            "changed_value": spec.changed_value,
            "baseline_candidate_id": str(br.get("candidate_id", "")),
            "baseline_candidate_day": bday,
            "baseline_candidate_date": day_index_to_month_day(bday),
            "perturbed_candidate_id": pcid,
            "perturbed_candidate_day": pday,
            "perturbed_candidate_date": day_index_to_month_day(int(pday)) if pd.notna(pday) else "",
            "day_delta": pday - bday if pd.notna(pday) else np.nan,
            "candidate_match_class": candidate_match_class(pday - bday if pd.notna(pday) else np.nan),
            "same_as_baseline_day": bool(pd.notna(pday) and int(pday) == bday),
            "baseline_score": bscore,
            "perturbed_score": pscore,
            "score_delta": pscore - bscore if np.isfinite(pscore) and np.isfinite(bscore) else np.nan,
            "candidate_status_change": "matched_to_perturbed" if pr is not None else "baseline_missing_in_perturbed",
        })
    # New perturbed candidates not near any baseline candidate within 8d.
    if not p.empty:
        bdays = pd.to_numeric(b["point_day"], errors="coerce").dropna().astype(int).tolist()
        for _, pr in p.sort_values("point_day").iterrows():
            pcid = str(pr.get("candidate_id", ""))
            pday = int(pr["point_day"])
            nearest = min([abs(pday - bd) for bd in bdays], default=999)
            if nearest > 8:
                rows.append({
                    "scope": scope,
                    "object": "joint_all" if scope == "joint_all" else scope.replace("_only", ""),
                    "config_id": spec.config_id,
                    "sensitivity_group": spec.sensitivity_group,
                    "changed_factor": spec.changed_factor,
                    "changed_value": spec.changed_value,
                    "baseline_candidate_id": "",
                    "baseline_candidate_day": np.nan,
                    "baseline_candidate_date": "",
                    "perturbed_candidate_id": pcid,
                    "perturbed_candidate_day": pday,
                    "perturbed_candidate_date": day_index_to_month_day(pday),
                    "day_delta": np.nan,
                    "candidate_match_class": "NEW_PERTURBED_CANDIDATE",
                    "same_as_baseline_day": False,
                    "baseline_score": np.nan,
                    "perturbed_score": _as_float(pr.get("peak_score", np.nan)),
                    "score_delta": np.nan,
                    "candidate_status_change": "new_perturbed_candidate",
                })
    return pd.DataFrame(rows)


def compare_support(baseline_boot: pd.DataFrame, pert_boot: pd.DataFrame, scope: str, spec: SensitivityConfigSpec, reference_config_id: str = "BASELINE") -> pd.DataFrame:
    rows = []
    b = baseline_boot.copy() if baseline_boot is not None else pd.DataFrame()
    p = pert_boot.copy() if pert_boot is not None else pd.DataFrame()
    if b.empty:
        return pd.DataFrame()
    # Compare by nearest point day.
    for _, br in b.sort_values("point_day").iterrows():
        bday = int(br["point_day"])
        pr, delta = nearest_row_by_day(p, "point_day", bday)
        def getfrac(row, col):
            return _as_float(row.get(col, np.nan)) if row is not None else np.nan
        bmatch = getfrac(br, "bootstrap_match_fraction")
        pmatch = getfrac(pr, "bootstrap_match_fraction") if pr is not None else np.nan
        # Support class names differ by scope.
        bcls = str(br.get("joint_support_class", br.get("object_support_class", "")))
        pcls = str(pr.get("joint_support_class", pr.get("object_support_class", ""))) if pr is not None else ""
        rows.append({
            "scope": scope,
            "object": "joint_all" if scope == "joint_all" else scope.replace("_only", ""),
            "config_id": spec.config_id,
            "reference_config_id": reference_config_id,
            "sensitivity_group": spec.sensitivity_group,
            "changed_factor": spec.changed_factor,
            "changed_value": spec.changed_value,
            "baseline_candidate_day": bday,
            "perturbed_candidate_day": int(pr["point_day"]) if pr is not None else np.nan,
            "candidate_match_class": candidate_match_class(delta if pr is not None else np.nan),
            "baseline_strict_fraction": getfrac(br, "bootstrap_strict_fraction"),
            "perturbed_strict_fraction": getfrac(pr, "bootstrap_strict_fraction") if pr is not None else np.nan,
            "baseline_match_fraction": bmatch,
            "perturbed_match_fraction": pmatch,
            "match_fraction_delta": pmatch - bmatch if np.isfinite(pmatch) and np.isfinite(bmatch) else np.nan,
            "baseline_near_fraction": getfrac(br, "bootstrap_near_fraction"),
            "perturbed_near_fraction": getfrac(pr, "bootstrap_near_fraction") if pr is not None else np.nan,
            "baseline_support_class": bcls,
            "perturbed_support_class": pcls,
            "support_class_changed": bool(bcls != pcls) if pcls else np.nan,
            "bootstrap_status": str(pr.get("bootstrap_status", "")) if pr is not None else "missing_or_not_run",
        })
    return pd.DataFrame(rows)


def compare_bands(baseline_bands: pd.DataFrame, pert_bands: pd.DataFrame, scope: str, spec: SensitivityConfigSpec, reference_config_id: str = "BASELINE") -> pd.DataFrame:
    rows = []
    b = baseline_bands.copy() if baseline_bands is not None else pd.DataFrame()
    p = pert_bands.copy() if pert_bands is not None else pd.DataFrame()
    if b.empty:
        return pd.DataFrame()
    for _, br in b.sort_values("point_day").iterrows():
        bday = int(br["point_day"])
        pr, delta = nearest_row_by_day(p, "point_day", bday)
        rows.append({
            "scope": scope,
            "object": "joint_all" if scope == "joint_all" else scope.replace("_only", ""),
            "config_id": spec.config_id,
            "reference_config_id": reference_config_id,
            "sensitivity_group": spec.sensitivity_group,
            "changed_factor": spec.changed_factor,
            "changed_value": spec.changed_value,
            "baseline_candidate_day": bday,
            "perturbed_candidate_day": int(pr["point_day"]) if pr is not None else np.nan,
            "candidate_match_class": candidate_match_class(delta if pr is not None else np.nan),
            "baseline_band_start": _as_int_or_nan(br.get("band_start_day", np.nan)),
            "baseline_band_end": _as_int_or_nan(br.get("band_end_day", np.nan)),
            "perturbed_band_start": _as_int_or_nan(pr.get("band_start_day", np.nan)) if pr is not None else np.nan,
            "perturbed_band_end": _as_int_or_nan(pr.get("band_end_day", np.nan)) if pr is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["band_start_delta"] = df["perturbed_band_start"] - df["baseline_band_start"]
        df["band_end_delta"] = df["perturbed_band_end"] - df["baseline_band_end"]
        df["baseline_band_width"] = df["baseline_band_end"] - df["baseline_band_start"] + 1
        df["perturbed_band_width"] = df["perturbed_band_end"] - df["perturbed_band_start"] + 1
        df["band_width_delta"] = df["perturbed_band_width"] - df["baseline_band_width"]
    return df


def compare_windows(baseline_windows: pd.DataFrame, pert_windows: pd.DataFrame, scope: str, spec: SensitivityConfigSpec, reference_config_id: str = "BASELINE") -> pd.DataFrame:
    rows = []
    b = baseline_windows.copy() if baseline_windows is not None else pd.DataFrame()
    p = pert_windows.copy() if pert_windows is not None else pd.DataFrame()
    if b.empty:
        return pd.DataFrame()
    for _, br in b.sort_values("main_peak_day").iterrows():
        bday = int(br["main_peak_day"])
        pr, delta = nearest_row_by_day(p, "main_peak_day", bday)
        if pr is None:
            status = "MISSING_WINDOW"
        else:
            if abs(delta) == 0:
                status = "UNCHANGED" if (_as_int_or_nan(pr.get("start_day")) == _as_int_or_nan(br.get("start_day")) and _as_int_or_nan(pr.get("end_day")) == _as_int_or_nan(br.get("end_day"))) else "BOUNDARY_SHIFT_ONLY"
            elif abs(delta) <= 8:
                status = "MAIN_PEAK_CHANGED"
            else:
                status = "MISSING_WINDOW"
        rows.append({
            "scope": scope,
            "object": "joint_all" if scope == "joint_all" else scope.replace("_only", ""),
            "config_id": spec.config_id,
            "reference_config_id": reference_config_id,
            "sensitivity_group": spec.sensitivity_group,
            "changed_factor": spec.changed_factor,
            "changed_value": spec.changed_value,
            "baseline_window_id": str(br.get("window_id", "")),
            "baseline_main_peak_day": bday,
            "perturbed_window_id": str(pr.get("window_id", "")) if pr is not None else "",
            "perturbed_main_peak_day": int(pr["main_peak_day"]) if pr is not None else np.nan,
            "main_peak_delta": int(pr["main_peak_day"]) - bday if pr is not None else np.nan,
            "baseline_window_start": _as_int_or_nan(br.get("start_day", np.nan)),
            "baseline_window_end": _as_int_or_nan(br.get("end_day", np.nan)),
            "perturbed_window_start": _as_int_or_nan(pr.get("start_day", np.nan)) if pr is not None else np.nan,
            "perturbed_window_end": _as_int_or_nan(pr.get("end_day", np.nan)) if pr is not None else np.nan,
            "member_candidate_change": bool(str(br.get("member_candidate_ids", "")) != str(pr.get("member_candidate_ids", ""))) if pr is not None else np.nan,
            "merge_split_status": status,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["window_start_delta"] = df["perturbed_window_start"] - df["baseline_window_start"]
        df["window_end_delta"] = df["perturbed_window_end"] - df["baseline_window_end"]
    return df


# =============================================================================
# Lineage-aware mapping
# =============================================================================


def classify_provenance(scope: str, day: int, object_name: str, joint_lineage: pd.DataFrame, obj_catalog: pd.DataFrame, obj_mapping: pd.DataFrame, v10_selection: pd.DataFrame) -> dict[str, Any]:
    out = {
        "nearest_joint_candidate_day": np.nan,
        "distance_to_nearest_joint_candidate": np.nan,
        "nearest_joint_lineage_status": "",
        "nearest_joint_derived_window_id": "",
        "nearest_strict_accepted_window_id": "",
        "distance_to_nearest_strict_anchor": np.nan,
        "matched_object_native_baseline_candidate_id": "",
        "matched_object_native_baseline_candidate_day": np.nan,
        "object_support_class": "",
        "was_v10_window_conditioned_main_peak": False,
        "v10_window_id_if_main": "",
        "candidate_provenance_class": "NEW_CANDIDATE_NO_KNOWN_LINEAGE",
    }
    # Joint lineage mapping.
    if joint_lineage is not None and not joint_lineage.empty and "candidate_day" in joint_lineage.columns:
        jd = pd.to_numeric(joint_lineage["candidate_day"], errors="coerce")
        if jd.notna().any():
            idx = (jd - int(day)).abs().idxmin()
            jr = joint_lineage.loc[idx]
            out["nearest_joint_candidate_day"] = int(jd.loc[idx])
            out["distance_to_nearest_joint_candidate"] = int(abs(int(jd.loc[idx]) - int(day)))
            out["nearest_joint_lineage_status"] = str(jr.get("lineage_status", ""))
            out["nearest_joint_derived_window_id"] = str(jr.get("v6_1_window_id", ""))
            out["nearest_strict_accepted_window_id"] = str(jr.get("strict_accepted_window_id", ""))
            try:
                out["distance_to_nearest_strict_anchor"] = abs(int(jr.get("strict_accepted_anchor_day", np.nan)) - int(day))
            except Exception:
                out["distance_to_nearest_strict_anchor"] = np.nan
    # Object-native baseline mapping for object scopes only.
    obj_match_dist = np.nan
    if scope != "joint_all" and obj_catalog is not None and not obj_catalog.empty:
        sub = obj_catalog[obj_catalog.get("object", "").astype(str) == object_name] if "object" in obj_catalog.columns else pd.DataFrame()
        day_col = "point_day" if "point_day" in sub.columns else "candidate_day"
        if not sub.empty and day_col in sub.columns:
            vals = pd.to_numeric(sub[day_col], errors="coerce")
            if vals.notna().any():
                idx = (vals - int(day)).abs().idxmin()
                rr = sub.loc[idx]
                obj_match_dist = abs(float(vals.loc[idx]) - int(day))
                if obj_match_dist <= 8:
                    out["matched_object_native_baseline_candidate_id"] = str(rr.get("candidate_id", ""))
                    out["matched_object_native_baseline_candidate_day"] = int(vals.loc[idx])
                    out["object_support_class"] = str(rr.get("object_support_class", ""))
    # Window-conditioned V10 main selection mapping.
    if scope != "joint_all" and v10_selection is not None and not v10_selection.empty:
        if "object" in v10_selection.columns and "selected_peak_day" in v10_selection.columns:
            sub = v10_selection[v10_selection["object"].astype(str) == object_name].copy()
            vals = pd.to_numeric(sub["selected_peak_day"], errors="coerce")
            if not sub.empty and vals.notna().any():
                close = sub[(vals - int(day)).abs() <= 0]
                if not close.empty:
                    out["was_v10_window_conditioned_main_peak"] = True
                    out["v10_window_id_if_main"] = ";".join(sorted(close["window_id"].astype(str).unique().tolist())) if "window_id" in close.columns else ""
    # Provenance class.
    if out["was_v10_window_conditioned_main_peak"]:
        prov = "BASELINE_MAIN_CANDIDATE"
    elif scope != "joint_all" and pd.notna(out["matched_object_native_baseline_candidate_day"]):
        prov = "BASELINE_OBJECT_NATIVE_CANDIDATE"
    elif "strict_accepted" in str(out["nearest_joint_lineage_status"]):
        prov = "JOINT_STRICT_ACCEPTED_LINEAGE"
    elif pd.notna(out["distance_to_nearest_joint_candidate"]) and float(out["distance_to_nearest_joint_candidate"]) <= 8:
        prov = "JOINT_NON_STRICT_OR_KNOWN_LINEAGE"
    else:
        prov = "NEW_CANDIDATE_NO_KNOWN_LINEAGE"
    out["candidate_provenance_class"] = prov
    return out


def build_lineage_mapping_for_candidates(all_registries: pd.DataFrame, joint_lineage: pd.DataFrame, obj_catalog: pd.DataFrame, obj_mapping: pd.DataFrame, v10_selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if all_registries is None or all_registries.empty:
        return pd.DataFrame()
    for _, r in all_registries.iterrows():
        scope = str(r["scope"])
        obj = str(r.get("object", "joint_all"))
        day = int(r["point_day"])
        m = classify_provenance(scope, day, obj, joint_lineage, obj_catalog, obj_mapping, v10_selection)
        row = {
            "scope": scope,
            "object": obj,
            "config_id": str(r.get("config_id", "")),
            "sensitivity_group": str(r.get("sensitivity_group", "")),
            "changed_factor": str(r.get("changed_factor", "")),
            "changed_value": str(r.get("changed_value", "")),
            "candidate_id": str(r.get("candidate_id", "")),
            "perturbed_candidate_day": day,
            "perturbed_candidate_date": day_index_to_month_day(day),
            "perturbed_peak_score": _as_float(r.get("peak_score", np.nan)),
        }
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Summary helpers
# =============================================================================


def build_scope_config_summary(meta_rows: list[dict[str, Any]], cand_cmp: pd.DataFrame, support_cmp: pd.DataFrame, win_cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta = pd.DataFrame(meta_rows)
    if meta.empty:
        return pd.DataFrame()
    for _, mr in meta.iterrows():
        scope = str(mr["scope"]); cid = str(mr["config_id"])
        csub = cand_cmp[(cand_cmp["scope"] == scope) & (cand_cmp["config_id"] == cid)] if cand_cmp is not None and not cand_cmp.empty else pd.DataFrame()
        ssub = support_cmp[(support_cmp["scope"] == scope) & (support_cmp["config_id"] == cid)] if support_cmp is not None and not support_cmp.empty else pd.DataFrame()
        wsub = win_cmp[(win_cmp["scope"] == scope) & (win_cmp["config_id"] == cid)] if win_cmp is not None and not win_cmp.empty else pd.DataFrame()
        rows.append({
            "scope": scope,
            "object": "joint_all" if scope == "joint_all" else scope.replace("_only", ""),
            "config_id": cid,
            "sensitivity_group": mr.get("sensitivity_group"),
            "changed_factor": mr.get("changed_factor"),
            "changed_value": mr.get("changed_value"),
            "n_candidates": int(mr.get("n_candidates", 0)),
            "candidate_days": ";".join(map(str, mr.get("candidate_days", []))),
            "n_derived_windows": int(mr.get("n_derived_windows", 0)),
            "derived_window_main_peak_days": ";".join(map(str, mr.get("derived_window_main_peak_days", []))),
            "bootstrap_status": mr.get("bootstrap_status"),
            "n_candidate_same_day": int((csub.get("candidate_match_class", pd.Series(dtype=str)) == "SAME_DAY").sum()) if not csub.empty else 0,
            "n_candidate_shift_or_missing": int((csub.get("candidate_match_class", pd.Series(dtype=str)) != "SAME_DAY").sum()) if not csub.empty else 0,
            "n_support_class_changed": _safe_bool_true_count(ssub.get("support_class_changed")) if not ssub.empty else 0,
            "n_windows_changed": int((wsub.get("merge_split_status", pd.Series(dtype=str)) != "UNCHANGED").sum()) if not wsub.empty else 0,
        })
    return pd.DataFrame(rows)




def comparison_reference_config_id(spec: SensitivityConfigSpec) -> str:
    """Choose the appropriate baseline for a sensitivity config.

    Smooth5-internal perturbations must be compared against the smooth5 baseline
    (SMOOTH_INPUT_5D), not against the smooth9 BASELINE.  The plain
    SMOOTH_INPUT_5D config itself remains compared against BASELINE to preserve
    the original smooth9-vs-smooth5 input sensitivity question.
    """
    if spec.input_source == "smooth5" and spec.config_id != "SMOOTH_INPUT_5D":
        return "SMOOTH_INPUT_5D"
    return "BASELINE"


def build_smooth5_internal_summary(scope_summary: pd.DataFrame) -> pd.DataFrame:
    if scope_summary is None or scope_summary.empty:
        return pd.DataFrame()
    df = scope_summary.copy()
    df = df[df["config_id"].astype(str).str.startswith("SMOOTH5_")].copy()
    if df.empty:
        return pd.DataFrame()
    group_cols = ["scope", "object", "sensitivity_group", "changed_factor"]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        scope, obj, group, factor = keys
        rows.append({
            "scope": scope,
            "object": obj,
            "sensitivity_group": group,
            "changed_factor": factor,
            "n_configs": int(len(sub)),
            "total_candidate_shift_or_missing": int(pd.to_numeric(sub["n_candidate_shift_or_missing"], errors="coerce").fillna(0).sum()),
            "max_candidate_shift_or_missing": int(pd.to_numeric(sub["n_candidate_shift_or_missing"], errors="coerce").fillna(0).max()),
            "total_support_class_changed": int(pd.to_numeric(sub["n_support_class_changed"], errors="coerce").fillna(0).sum()),
            "total_windows_changed": int(pd.to_numeric(sub["n_windows_changed"], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows)

def build_candidate_shift_type_summary(cand_cmp: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate changes by explicit shift type, not one mixed count.

    This audit separates same-day, <=2d, <=5d, <=8d, >8d/missing, and new
    candidates so that detector-width/input sensitivity is not overstated by
    mixing small timing drift with true candidate replacement.
    """
    if cand_cmp is None or cand_cmp.empty:
        return pd.DataFrame()
    df = cand_cmp.copy()
    rows = []
    group_cols = ["scope", "object", "config_id", "reference_config_id", "sensitivity_group", "changed_factor", "changed_value"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        scope, obj, cid, ref, group, factor, value = keys
        cls = sub.get("candidate_match_class", pd.Series(dtype=str)).astype(str)
        # Baseline rows with no baseline_candidate_id are new perturbed candidates.
        has_base = sub.get("baseline_candidate_id", pd.Series(index=sub.index, dtype=str)).astype(str).str.len() > 0
        base_rows = sub[has_base].copy()
        new_rows = sub[~has_base].copy()
        deltas = pd.to_numeric(base_rows.get("day_delta", pd.Series(dtype=float)), errors="coerce").abs()
        rows.append({
            "scope": scope,
            "object": obj,
            "config_id": cid,
            "reference_config_id": ref,
            "sensitivity_group": group,
            "changed_factor": factor,
            "changed_value": value,
            "n_baseline_candidates_compared": int(len(base_rows)),
            "n_same_day": int((base_rows.get("candidate_match_class", pd.Series(dtype=str)).astype(str) == "SAME_DAY").sum()) if not base_rows.empty else 0,
            "n_strict_shift_le_2d": int((base_rows.get("candidate_match_class", pd.Series(dtype=str)).astype(str) == "STRICT_SHIFT_LE_2D").sum()) if not base_rows.empty else 0,
            "n_match_shift_le_5d": int((base_rows.get("candidate_match_class", pd.Series(dtype=str)).astype(str) == "MATCH_SHIFT_LE_5D").sum()) if not base_rows.empty else 0,
            "n_near_shift_le_8d": int((base_rows.get("candidate_match_class", pd.Series(dtype=str)).astype(str) == "NEAR_SHIFT_LE_8D").sum()) if not base_rows.empty else 0,
            "n_shift_gt_8d_or_missing": int((base_rows.get("candidate_match_class", pd.Series(dtype=str)).astype(str).isin(["SHIFT_GT_8D", "MISSING_BASELINE_CANDIDATE"]).sum())) if not base_rows.empty else 0,
            "n_missing_baseline_candidate": int((base_rows.get("candidate_match_class", pd.Series(dtype=str)).astype(str) == "MISSING_BASELINE_CANDIDATE").sum()) if not base_rows.empty else 0,
            "n_new_perturbed_candidate": int(len(new_rows)),
            "median_abs_day_delta": float(deltas.median()) if deltas.notna().any() else np.nan,
            "max_abs_day_delta": float(deltas.max()) if deltas.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def build_candidate_order_inversion_summary(cand_cmp: pd.DataFrame) -> pd.DataFrame:
    """Audit whether matched candidate sequences invert their temporal order.

    The audit only compares baseline candidates that have finite perturbed-day
    matches. New candidates and missing baseline candidates are counted but not
    used for pairwise inversion because they change sequence composition rather
    than invert matched candidate order.
    """
    if cand_cmp is None or cand_cmp.empty:
        return pd.DataFrame()
    df = cand_cmp.copy()
    rows = []
    group_cols = ["scope", "object", "config_id", "reference_config_id", "sensitivity_group", "changed_factor", "changed_value"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        scope, obj, cid, ref, group, factor, value = keys
        bday = pd.to_numeric(sub.get("baseline_candidate_day"), errors="coerce")
        pday = pd.to_numeric(sub.get("perturbed_candidate_day"), errors="coerce")
        cls = sub.get("candidate_match_class", pd.Series(index=sub.index, dtype=str)).astype(str)
        matched = sub[bday.notna() & pday.notna() & ~cls.isin(["MISSING_BASELINE_CANDIDATE", "NEW_PERTURBED_CANDIDATE"])].copy()
        matched["_bday"] = pd.to_numeric(matched["baseline_candidate_day"], errors="coerce")
        matched["_pday"] = pd.to_numeric(matched["perturbed_candidate_day"], errors="coerce")
        matched = matched.sort_values("_bday")
        b = matched["_bday"].to_numpy(dtype=float)
        p = matched["_pday"].to_numpy(dtype=float)
        n_inv = 0
        n_pairs = 0
        n_ties_after = 0
        n_order_preserved = 0
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                if not (np.isfinite(b[i]) and np.isfinite(b[j]) and np.isfinite(p[i]) and np.isfinite(p[j])):
                    continue
                if b[i] == b[j]:
                    continue
                n_pairs += 1
                base_sign = np.sign(b[j] - b[i])
                pert_sign = np.sign(p[j] - p[i])
                if pert_sign == 0:
                    n_ties_after += 1
                elif base_sign != pert_sign:
                    n_inv += 1
                else:
                    n_order_preserved += 1
        new_count = int((cls == "NEW_PERTURBED_CANDIDATE").sum())
        missing_count = int((cls == "MISSING_BASELINE_CANDIDATE").sum())
        if n_pairs == 0:
            status = "INSUFFICIENT_MATCHED_CANDIDATES"
        elif n_inv == 0:
            status = "NO_ORDER_INVERSION"
        else:
            status = "ORDER_INVERSION_PRESENT"
        rows.append({
            "scope": scope,
            "object": obj,
            "config_id": cid,
            "reference_config_id": ref,
            "sensitivity_group": group,
            "changed_factor": factor,
            "changed_value": value,
            "n_matched_candidates_for_order": int(len(matched)),
            "n_pairwise_orders": int(n_pairs),
            "n_order_preserved_pairs": int(n_order_preserved),
            "n_inversions": int(n_inv),
            "n_ties_after_perturbation": int(n_ties_after),
            "n_new_perturbed_candidates": new_count,
            "n_missing_baseline_candidates": missing_count,
            "order_stability_status": status,
        })
    return pd.DataFrame(rows)


def build_new_candidate_inventory(cand_cmp: pd.DataFrame, lineage_mapping: pd.DataFrame) -> pd.DataFrame:
    """List all new perturbed candidates and attach lineage/provenance fields."""
    if cand_cmp is None or cand_cmp.empty:
        return pd.DataFrame()
    df = cand_cmp.copy()
    is_new = df.get("candidate_match_class", pd.Series(index=df.index, dtype=str)).astype(str) == "NEW_PERTURBED_CANDIDATE"
    new = df[is_new].copy()
    if new.empty:
        return pd.DataFrame()
    # Attach candidate provenance from the full lineage mapping by scope/config/day.
    if lineage_mapping is not None and not lineage_mapping.empty:
        lm = lineage_mapping.copy()
        for col in ["scope", "object", "config_id"]:
            if col in lm.columns:
                lm[col] = lm[col].astype(str)
        lm["_day_key"] = pd.to_numeric(lm.get("perturbed_candidate_day"), errors="coerce").round().astype("Int64")
        new["_day_key"] = pd.to_numeric(new.get("perturbed_candidate_day"), errors="coerce").round().astype("Int64")
        join_cols = ["scope", "object", "config_id", "_day_key"]
        keep_cols = join_cols + [c for c in [
            "candidate_id", "nearest_joint_candidate_day", "distance_to_nearest_joint_candidate",
            "nearest_joint_lineage_status", "nearest_joint_derived_window_id",
            "nearest_strict_accepted_window_id", "distance_to_nearest_strict_anchor",
            "matched_object_native_baseline_candidate_id", "matched_object_native_baseline_candidate_day",
            "object_support_class", "was_v10_window_conditioned_main_peak",
            "v10_window_id_if_main", "candidate_provenance_class"
        ] if c in lm.columns]
        new = new.merge(lm[keep_cols], on=join_cols, how="left", suffixes=("", "_lineage"))
        new = new.drop(columns=["_day_key"], errors="ignore")
    cols = [c for c in [
        "scope", "object", "config_id", "reference_config_id", "sensitivity_group",
        "changed_factor", "changed_value", "perturbed_candidate_id", "perturbed_candidate_day",
        "perturbed_candidate_date", "perturbed_score", "nearest_joint_candidate_day",
        "distance_to_nearest_joint_candidate", "nearest_joint_lineage_status",
        "nearest_joint_derived_window_id", "nearest_strict_accepted_window_id",
        "distance_to_nearest_strict_anchor", "matched_object_native_baseline_candidate_id",
        "matched_object_native_baseline_candidate_day", "object_support_class",
        "was_v10_window_conditioned_main_peak", "v10_window_id_if_main",
        "candidate_provenance_class"
    ] if c in new.columns]
    return new[cols].copy()


def build_detector_width_bootstrap_summary(support_cmp: pd.DataFrame) -> pd.DataFrame:
    """Summarize support changes for the minimal detector-width bootstrap configs."""
    if support_cmp is None or support_cmp.empty:
        return pd.DataFrame()
    df = support_cmp.copy()
    df = df[df.get("sensitivity_group", pd.Series(dtype=str)).astype(str).isin(["detector_width", "smooth5_detector_width"])].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["scope", "object", "config_id", "reference_config_id", "sensitivity_group", "changed_factor", "changed_value"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        scope, obj, cid, ref, group, factor, value = keys
        status = sub.get("bootstrap_status", pd.Series(dtype=str)).astype(str)
        ran = ~(status.isin(["not_run_by_mode", "missing_or_not_run", ""]))
        md = pd.to_numeric(sub.get("match_fraction_delta", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "scope": scope,
            "object": obj,
            "config_id": cid,
            "reference_config_id": ref,
            "sensitivity_group": group,
            "changed_factor": factor,
            "changed_value": value,
            "n_candidates_compared": int(len(sub)),
            "n_bootstrap_rows_with_run": int(ran.sum()),
            "n_support_class_changed": _safe_bool_true_count(sub.get("support_class_changed")),
            "mean_match_fraction_delta": float(md.mean()) if md.notna().any() else np.nan,
            "median_abs_match_fraction_delta": float(md.abs().median()) if md.notna().any() else np.nan,
            "max_abs_match_fraction_delta": float(md.abs().max()) if md.notna().any() else np.nan,
            "bootstrap_audit_status": "BOOTSTRAP_AVAILABLE" if int(ran.sum()) > 0 else "BOOTSTRAP_NOT_RUN",
        })
    return pd.DataFrame(rows)

def write_markdown_summary(output_root: Path, run_meta: dict[str, Any], scope_summary: pd.DataFrame, settings: V10_3Settings) -> None:
    lines = [
        "# V10.3 peak discovery sensitivity summary",
        "",
        "This run performs single-factor sensitivity tests for the free-season peak discovery workflow.",
        "It covers joint_all and P/V/H/Je/Jw object-native scopes.",
        "",
        "## Boundary",
        "- Does not perform physical interpretation.",
        "- Does not perform pair-order analysis.",
        "- Does not re-decide strict accepted windows.",
        "- Uses V10.1/V10.2 outputs as baseline lineage/catalog references.",
        "",
        "## Run status",
        f"- status: {run_meta.get('status')}",
        f"- bootstrap_mode: {run_meta.get('bootstrap_mode')}",
        f"- n_bootstrap_if_run: {settings.resolved_n_bootstrap()}",
        f"- n_scopes: {run_meta.get('n_scopes')}",
        f"- n_configs: {run_meta.get('n_configs')}",
        "",
        "## Configs",
    ]
    for item in run_meta.get("configs", []):
        lines.append(f"- {item.get('config_id')}: {item.get('sensitivity_group')} {item.get('changed_factor')}={item.get('changed_value')} bootstrap={item.get('bootstrap_planned')}")
    lines += ["", "## Scope/config summary"]
    if scope_summary is not None and not scope_summary.empty:
        for _, r in scope_summary.iterrows():
            lines.append(
                f"- {r['scope']} / {r['config_id']}: candidates={r['n_candidates']} days={r['candidate_days']} "
                f"windows={r['n_derived_windows']} main_days={r['derived_window_main_peak_days']} "
                f"candidate_shift_or_missing={r['n_candidate_shift_or_missing']} windows_changed={r['n_windows_changed']}"
            )
    lines += [
        "",
        "## Added closure audits",
        "- cross_scope/candidate_order_inversion_summary_v10_3.csv",
        "- cross_scope/candidate_shift_type_summary_v10_3.csv",
        "- cross_scope/detector_width_bootstrap_support_summary_v10_3.csv",
        "- lineage_mapping/new_candidate_inventory_v10_3.csv",
        "",
        "## Interpretation limits",
        "Changed candidates are candidate-discovery sensitivity evidence only. A changed candidate should be mapped to V10.1 joint lineage and V10.2 object-native baseline before any physical audit.",
    ]
    (output_root / "PEAK_DISCOVERY_SENSITIVITY_V10_3_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main runner
# =============================================================================


def run_peak_discovery_sensitivity_v10_3(bundle_root: Path | str | None = None, settings: V10_3Settings | None = None) -> dict[str, Any]:
    settings = settings or V10_3Settings()
    bundle_root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parents[1]
    v10_root = bundle_root.parent
    output_root = bundle_root / "outputs" / OUTPUT_TAG
    log_root = bundle_root / "logs"
    started = now_utc()
    clean_output_dirs(output_root)
    log_root.mkdir(parents=True, exist_ok=True)
    write_json(settings.to_dict(), output_root / "config_used.json")

    v10_2 = load_v10_2_module(bundle_root)
    base_settings = v10_2.Settings()
    base_settings.output_tag = OUTPUT_TAG
    # Ensure V10_3 env controls bootstrap count through the copied V10.2 settings object.
    base_settings.bootstrap.n_bootstrap = int(settings.resolved_n_bootstrap())
    base_settings.bootstrap.progress = bool(settings.progress)

    input_sources, input_source_inventory = load_input_profile_sources(v10_2, base_settings, settings)
    write_dataframe(input_source_inventory, output_root / "cross_scope" / "input_source_inventory_v10_3.csv")

    joint_lineage = read_v10_1_lineage(v10_root)
    obj_catalog = read_v10_2_object_catalog(v10_root)
    obj_mapping = read_v10_2_object_mapping(v10_root)
    v10_selection = read_v10_window_conditioned_selection(v10_root)
    write_dataframe(joint_lineage, output_root / "lineage_mapping" / "joint_lineage_reference_used_v10_3.csv")
    write_dataframe(obj_catalog, output_root / "lineage_mapping" / "object_native_catalog_reference_used_v10_3.csv")
    write_dataframe(obj_mapping, output_root / "lineage_mapping" / "object_native_lineage_mapping_reference_used_v10_3.csv")
    write_dataframe(v10_selection, output_root / "lineage_mapping" / "v10_window_conditioned_selection_reference_used_v10_3.csv")

    specs = build_sensitivity_specs(settings)
    bootstrap_mode = settings.resolved_bootstrap_mode()
    config_rows = []
    for spec in specs:
        config_rows.append({
            **spec.__dict__,
            "bootstrap_planned": should_run_bootstrap(spec, bootstrap_mode),
        })
    config_df = pd.DataFrame(config_rows)
    write_dataframe(config_df, output_root / "cross_scope" / "sensitivity_config_grid_v10_3.csv")

    results: dict[tuple[str, str], dict[str, Any]] = {}
    meta_rows: list[dict[str, Any]] = []
    all_registries: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_boot_summaries: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    all_windows: list[pd.DataFrame] = []
    all_memberships: list[pd.DataFrame] = []

    for scope, object_names in SCOPE_SPECS:
        scope_dir = output_root / "by_scope" / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            cfg_dir = scope_dir / spec.config_id
            cfg_dir.mkdir(parents=True, exist_ok=True)
            src = input_sources.get(spec.input_source)
            if src is None or src.get("status") != "loaded":
                raise RuntimeError(f"Input source {spec.input_source!r} is not loaded; check input_source_inventory_v10_3.csv")
            res = run_scope_config(
                v10_2,
                src["profiles"],
                src["years"],
                src,
                scope,
                object_names,
                spec,
                base_settings,
                settings,
                bootstrap_mode,
            )
            results[(scope, spec.config_id)] = res
            meta_rows.append(res["meta"])
            # Write per-scope/config outputs.
            write_dataframe(res["registry"], cfg_dir / f"{scope}_{spec.config_id}_candidate_registry_v10_3.csv")
            write_dataframe(res["detector_scores"], cfg_dir / f"{scope}_{spec.config_id}_detector_scores_v10_3.csv")
            write_dataframe(res["boot_summary"], cfg_dir / f"{scope}_{spec.config_id}_bootstrap_summary_v10_3.csv")
            write_dataframe(res["bands"], cfg_dir / f"{scope}_{spec.config_id}_candidate_bands_v10_3.csv")
            write_dataframe(res["windows"], cfg_dir / f"{scope}_{spec.config_id}_derived_windows_v10_3.csv")
            write_dataframe(res["membership"], cfg_dir / f"{scope}_{spec.config_id}_window_membership_v10_3.csv")
            write_json(res["meta"], cfg_dir / f"{scope}_{spec.config_id}_meta_v10_3.json")
            all_registries.append(res["registry"])
            all_scores.append(res["detector_scores"])
            all_boot_summaries.append(res["boot_summary"])
            all_bands.append(res["bands"])
            all_windows.append(res["windows"])
            all_memberships.append(res["membership"])

    all_registry_df = pd.concat([x for x in all_registries if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in all_registries) else pd.DataFrame()
    all_scores_df = pd.concat([x for x in all_scores if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in all_scores) else pd.DataFrame()
    all_boot_df = pd.concat([x for x in all_boot_summaries if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in all_boot_summaries) else pd.DataFrame()
    all_bands_df = pd.concat([x for x in all_bands if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in all_bands) else pd.DataFrame()
    all_windows_df = pd.concat([x for x in all_windows if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in all_windows) else pd.DataFrame()
    all_membership_df = pd.concat([x for x in all_memberships if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in all_memberships) else pd.DataFrame()

    write_dataframe(all_registry_df, output_root / "cross_scope" / "candidate_registry_all_scopes_configs_v10_3.csv")
    write_dataframe(all_scores_df, output_root / "cross_scope" / "detector_scores_all_scopes_configs_v10_3.csv")
    write_dataframe(all_boot_df, output_root / "cross_scope" / "bootstrap_support_all_scopes_configs_v10_3.csv")
    write_dataframe(all_bands_df, output_root / "cross_scope" / "candidate_bands_all_scopes_configs_v10_3.csv")
    write_dataframe(all_windows_df, output_root / "cross_scope" / "derived_windows_all_scopes_configs_v10_3.csv")
    write_dataframe(all_membership_df, output_root / "cross_scope" / "window_membership_all_scopes_configs_v10_3.csv")

    # Compare every config to the scope baseline.
    cand_cmp_rows = []
    support_cmp_rows = []
    band_cmp_rows = []
    win_cmp_rows = []
    for scope, object_names in SCOPE_SPECS:
        for spec in specs:
            reference_config_id = comparison_reference_config_id(spec)
            if (scope, reference_config_id) not in results:
                raise RuntimeError(f"Missing comparison reference {reference_config_id} for {scope}/{spec.config_id}")
            base = results[(scope, reference_config_id)]
            res = results[(scope, spec.config_id)]
            cand_cmp_rows.append(compare_candidates(base["registry"], res["registry"], scope, spec, reference_config_id))
            support_cmp_rows.append(compare_support(base["boot_summary"], res["boot_summary"], scope, spec, reference_config_id))
            band_cmp_rows.append(compare_bands(base["bands"], res["bands"], scope, spec, reference_config_id))
            win_cmp_rows.append(compare_windows(base["windows"], res["windows"], scope, spec, reference_config_id))
    cand_cmp = pd.concat([x for x in cand_cmp_rows if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in cand_cmp_rows) else pd.DataFrame()
    support_cmp = pd.concat([x for x in support_cmp_rows if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in support_cmp_rows) else pd.DataFrame()
    band_cmp = pd.concat([x for x in band_cmp_rows if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in band_cmp_rows) else pd.DataFrame()
    win_cmp = pd.concat([x for x in win_cmp_rows if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in win_cmp_rows) else pd.DataFrame()

    write_dataframe(cand_cmp, output_root / "cross_scope" / "candidate_peak_sensitivity_by_scope_config_v10_3.csv")
    write_dataframe(support_cmp, output_root / "cross_scope" / "bootstrap_support_sensitivity_by_scope_config_v10_3.csv")
    write_dataframe(band_cmp, output_root / "cross_scope" / "candidate_band_sensitivity_by_scope_config_v10_3.csv")
    write_dataframe(win_cmp, output_root / "cross_scope" / "derived_window_sensitivity_by_scope_config_v10_3.csv")

    lineage_mapping = build_lineage_mapping_for_candidates(all_registry_df, joint_lineage, obj_catalog, obj_mapping, v10_selection)
    write_dataframe(lineage_mapping, output_root / "lineage_mapping" / "candidate_lineage_sensitivity_mapping_v10_3.csv")

    scope_summary = build_scope_config_summary(meta_rows, cand_cmp, support_cmp, win_cmp)
    write_dataframe(scope_summary, output_root / "cross_scope" / "peak_discovery_sensitivity_summary_by_scope_config_v10_3.csv")
    smooth5_internal_summary = build_smooth5_internal_summary(scope_summary)
    write_dataframe(smooth5_internal_summary, output_root / "cross_scope" / "smooth5_internal_sensitivity_summary_by_scope_factor_v10_3.csv")

    shift_type_summary = build_candidate_shift_type_summary(cand_cmp)
    write_dataframe(shift_type_summary, output_root / "cross_scope" / "candidate_shift_type_summary_v10_3.csv")
    order_inversion_summary = build_candidate_order_inversion_summary(cand_cmp)
    write_dataframe(order_inversion_summary, output_root / "cross_scope" / "candidate_order_inversion_summary_v10_3.csv")
    new_candidate_inventory = build_new_candidate_inventory(cand_cmp, lineage_mapping)
    write_dataframe(new_candidate_inventory, output_root / "lineage_mapping" / "new_candidate_inventory_v10_3.csv")
    detector_width_bootstrap_summary = build_detector_width_bootstrap_summary(support_cmp)
    write_dataframe(detector_width_bootstrap_summary, output_root / "cross_scope" / "detector_width_bootstrap_support_summary_v10_3.csv")

    run_meta = {
        "status": "success",
        "started_at": started,
        "finished_at": now_utc(),
        "output_tag": OUTPUT_TAG,
        "bundle_root": str(bundle_root),
        "v10_root": str(v10_root),
        "bootstrap_mode": bootstrap_mode,
        "n_bootstrap_if_run": int(settings.resolved_n_bootstrap()),
        "n_scopes": len(SCOPE_SPECS),
        "scopes": [x[0] for x in SCOPE_SPECS],
        "n_configs": len(specs),
        "configs": config_rows,
        "include_smoothing_group": bool(settings.resolved_include_smoothing_group()),
        "include_smooth5_internal_group": bool(settings.resolved_include_smooth5_internal_group()),
        "smooth5_internal_comparison_reference": "SMOOTH_INPUT_5D",
        "input_source_inventory": input_source_inventory.to_dict(orient="records"),
        "does_not_perform_physical_interpretation": True,
        "does_not_perform_pair_order_analysis": True,
        "does_not_redefine_strict_accepted_windows": True,
        "uses_v10_1_joint_lineage_reference": bool(joint_lineage is not None and not joint_lineage.empty),
        "uses_v10_2_object_native_reference": bool(obj_catalog is not None and not obj_catalog.empty),
        "calls_v6_v6_1_v7_v9_modules": False,
        "loads_v10_2_semantic_base_by_path": True,
        "closure_audits_added": [
            "candidate_order_inversion_summary",
            "detector_width_bootstrap_support_summary",
            "new_candidate_inventory",
            "candidate_shift_type_summary",
        ],
        "default_baseline_and_match_bootstrap_includes_detector_width": True,
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json({
        "n_scope_config_rows": int(len(scope_summary)),
        "n_candidate_comparison_rows": int(len(cand_cmp)),
        "n_support_comparison_rows": int(len(support_cmp)),
        "n_band_comparison_rows": int(len(band_cmp)),
        "n_window_comparison_rows": int(len(win_cmp)),
        "n_lineage_mapping_rows": int(len(lineage_mapping)),
        "n_smooth5_internal_summary_rows": int(len(smooth5_internal_summary)),
        "n_shift_type_summary_rows": int(len(shift_type_summary)),
        "n_order_inversion_summary_rows": int(len(order_inversion_summary)),
        "n_new_candidate_inventory_rows": int(len(new_candidate_inventory)),
        "n_detector_width_bootstrap_summary_rows": int(len(detector_width_bootstrap_summary)),
    }, output_root / "summary.json")
    write_markdown_summary(output_root, run_meta, scope_summary, settings)
    (log_root / "last_run.txt").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_meta


if __name__ == "__main__":
    run_peak_discovery_sensitivity_v10_3()
