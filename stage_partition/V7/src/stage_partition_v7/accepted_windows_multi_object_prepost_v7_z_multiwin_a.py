"""
V7-z-multiwin-a: accepted-window multi-object profile-mainline + 2D mirror.

This module generalizes the stabilized W45 clean mainline to every accepted/significant
system window.  It does NOT rediscover system windows.  It first builds a formal
window_scope_registry, then all subsequent profile detector, pre/post extraction,
2D mirror, bootstrap, and evidence-gate outputs are derived from that table.

Key scope separation
--------------------
- system_window: accepted core band from the stage-partition registry.
- detector_search_range: wide range between neighboring accepted windows, with a
  right neighbor buffer, used only by raw/profile object-window detector.
- analysis_range: C0 pre + system window + C0 post, used for daily pre/post curves.
- early/core/late: fixed comparison segments around system_window.

Dependencies
------------
This hotfix keeps the profile pre/post and 2D metric helpers from prior patches,
but restores the original V7-z ruptures.Window detector for profile object-window
detection.  By default it uses the hard-coded accepted/significant windows confirmed in the
current V7 workflow (W045, W081, W113, W160), processes only W45/anchor-45, and
runs profile-only metrics. Set V7_MULTI_WINDOW_MODE=all and V7_MULTI_RUN_2D=1
to expand. External accepted-window registries are optional overrides only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd

# Reuse stabilized W45 clean and 2D helper implementations.
try:
    from stage_partition_v7 import W45_multi_object_prepost_clean_mainline_v7_z_clean as clean
    from stage_partition_v7 import W45_2d_prepost_metric_mirror_v7_z_2d_a as mirror2d
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "V7-z-multiwin-a requires previous patches: "
        "W45_multi_object_prepost_clean_mainline_v7_z_clean.py and "
        "W45_2d_prepost_metric_mirror_v7_z_2d_a.py. "
        "Apply V7-z-clean and V7-z-2d-a hotfix patches first."
    ) from exc

VERSION = "v7_z_multiwin_a_hotfix06_w45_profile_order"
OUTPUT_TAG = "accepted_windows_multi_object_prepost_v7_z_multiwin_a_hotfix06_w45_profile_order"
EPS = 1.0e-12


@dataclass(frozen=True)
class AcceptedWindow:
    window_id: str
    anchor_day: int
    system_window_start: int
    system_window_end: int
    bootstrap_support: float
    accepted_status: str
    source_file: str


@dataclass(frozen=True)
class BaselineScope:
    name: str
    pre_start: int
    pre_end: int
    post_start: int
    post_end: int
    role: str
    is_valid: bool
    invalid_reason: str = ""

    def to_clean(self) -> clean.BaselineConfig:
        return clean.BaselineConfig(self.name, self.pre_start, self.pre_end, self.post_start, self.post_end, self.role)

    def to_2d(self) -> mirror2d.BaselineConfig:
        return mirror2d.BaselineConfig(self.name, self.pre_start, self.pre_end, self.post_start, self.post_end, self.role)


@dataclass(frozen=True)
class WindowScope:
    window_id: str
    anchor_day: int
    system_window_start: int
    system_window_end: int
    bootstrap_support: float
    accepted_status: str
    source_file: str
    detector_search_start: int
    detector_search_end: int
    detector_neighbor_buffer: int
    analysis_start: int
    analysis_end: int
    C0_pre_start: int
    C0_pre_end: int
    C0_post_start: int
    C0_post_end: int
    C1_pre_start: int
    C1_pre_end: int
    C1_post_start: int
    C1_post_end: int
    C2_pre_start: int
    C2_pre_end: int
    C2_post_start: int
    C2_post_end: int
    early_start: int
    early_end: int
    core_start: int
    core_end: int
    late_start: int
    late_end: int
    is_valid_for_prepost: bool
    invalid_reason: str = ""

    def baselines(self) -> List[BaselineScope]:
        bases = [
            BaselineScope("C0_full_stage", self.C0_pre_start, self.C0_pre_end, self.C0_post_start, self.C0_post_end, "main_full_stage", True),
            BaselineScope("C1_buffered_stage", self.C1_pre_start, self.C1_pre_end, self.C1_post_start, self.C1_post_end, "buffered_sensitivity", True),
            BaselineScope("C2_immediate_pre", self.C2_pre_start, self.C2_pre_end, self.C2_post_start, self.C2_post_end, "immediate_pre_sensitivity", True),
        ]
        out: List[BaselineScope] = []
        for b in bases:
            pre_len = b.pre_end - b.pre_start + 1
            post_len = b.post_end - b.post_start + 1
            min_pre = 8 if b.name in ["C0_full_stage", "C2_immediate_pre"] else 5
            min_post = 8 if b.name == "C0_full_stage" else 5
            if b.pre_start > b.pre_end or b.post_start > b.post_end:
                out.append(BaselineScope(b.name, b.pre_start, b.pre_end, b.post_start, b.post_end, b.role, False, "empty_range"))
            elif pre_len < min_pre:
                out.append(BaselineScope(b.name, b.pre_start, b.pre_end, b.post_start, b.post_end, b.role, False, f"short_pre_{pre_len}"))
            elif post_len < min_post:
                out.append(BaselineScope(b.name, b.pre_start, b.pre_end, b.post_start, b.post_end, b.role, False, f"short_post_{post_len}"))
            else:
                out.append(b)
        return out


@dataclass
class MultiWinConfig:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    season_start: int = 0
    season_end: int = 182
    detector_neighbor_buffer: int = 5
    baseline_buffer_days: int = 5
    immediate_pre_days: int = 10
    segment_pre_days: int = 10
    segment_late_extra_days: int = 5
    min_detector_search_days: int = 25
    detector_width: int = 20
    detector_min_size: int = 2
    peak_min_distance: int = 3
    max_peaks_per_object: int = 5
    band_max_half_width: int = 10
    band_min_half_width: int = 2
    band_score_ratio: float = 0.50
    band_floor_quantile: float = 0.35
    bootstrap_n: int = 1000
    random_seed: int = 42
    peak_match_days: int = 5
    low_dynamic_range_eps: float = 1e-10
    # Conservative default: run only W45/anchor-45 and profile-only until W45 regression is verified.
    run_2d: bool = False
    save_daily_curves: bool = True
    save_bootstrap_samples: bool = False
    skip_figures: bool = True
    window_mode: str = "w45"  # w45 | all | list
    target_windows: str = "W045,45"
    # Default is hardcoded because this project has not used an accepted_windows table.
    # Optional modes: hardcoded | registry | auto.  Registry/auto are opt-in only.
    window_source: str = "hardcoded"
    accepted_window_registry: Optional[str] = None
    smoothed_fields_path: Optional[str] = None
    log_every_bootstrap: int = 50
    run_w45_profile_order_tests: bool = True
    save_bootstrap_curves: bool = False
    tau_sync_quantile_primary: float = 0.75
    tau_sync_quantile_low: float = 0.50
    tau_sync_quantile_high: float = 0.90
    merge_short_gap_days: int = 1

    @staticmethod
    def from_env() -> "MultiWinConfig":
        cfg = MultiWinConfig()
        cfg.accepted_window_registry = os.environ.get("V7_MULTI_ACCEPTED_WINDOW_REGISTRY")
        cfg.window_source = os.environ.get("V7_MULTI_WINDOW_SOURCE", cfg.window_source).strip().lower()
        # If the user explicitly provides a registry path, treat it as an override.
        if cfg.accepted_window_registry:
            cfg.window_source = "registry"
        cfg.smoothed_fields_path = os.environ.get("V7_MULTI_SMOOTHED_FIELDS") or os.environ.get("V7Z_SMOOTHED_FIELDS")
        if os.environ.get("V7_MULTI_N_BOOTSTRAP"):
            cfg.bootstrap_n = int(os.environ["V7_MULTI_N_BOOTSTRAP"])
        if os.environ.get("V7_MULTI_DEBUG_N_BOOTSTRAP"):
            cfg.bootstrap_n = int(os.environ["V7_MULTI_DEBUG_N_BOOTSTRAP"])
        if os.environ.get("V7_MULTI_RUN_2D") is not None:
            cfg.run_2d = os.environ.get("V7_MULTI_RUN_2D") == "1"
        if os.environ.get("V7_MULTI_WINDOW_MODE"):
            cfg.window_mode = os.environ["V7_MULTI_WINDOW_MODE"].strip().lower()
        if os.environ.get("V7_MULTI_TARGET_WINDOWS"):
            cfg.target_windows = os.environ["V7_MULTI_TARGET_WINDOWS"].strip()
        if os.environ.get("V7_MULTI_SAVE_DAILY_CURVES") is not None:
            cfg.save_daily_curves = os.environ.get("V7_MULTI_SAVE_DAILY_CURVES") == "1"
        if os.environ.get("V7_MULTI_SAVE_BOOTSTRAP_SAMPLES") == "1":
            cfg.save_bootstrap_samples = True
        if os.environ.get("V7_MULTI_LOG_EVERY_BOOTSTRAP"):
            cfg.log_every_bootstrap = int(os.environ["V7_MULTI_LOG_EVERY_BOOTSTRAP"])
        if os.environ.get("V7_MULTI_RUN_W45_PROFILE_ORDER_TESTS") is not None:
            cfg.run_w45_profile_order_tests = os.environ.get("V7_MULTI_RUN_W45_PROFILE_ORDER_TESTS") == "1"
        if os.environ.get("V7_MULTI_SAVE_BOOTSTRAP_CURVES") == "1":
            cfg.save_bootstrap_curves = True
        return cfg


# -----------------------------------------------------------------------------
# Logging and helpers
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _safe_quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _prob_positive(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr > 0)) if arr.size else float("nan")


def _prob_negative(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr < 0)) if arr.size else float("nan")


def _summarize_samples(samples: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": np.nan, "q025": np.nan, "q975": np.nan, "P_positive": np.nan, "P_negative": np.nan}
    return {
        "median": float(np.median(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q975": float(np.quantile(arr, 0.975)),
        "P_positive": float(np.mean(arr > 0)),
        "P_negative": float(np.mean(arr < 0)),
    }


def _decision_from_samples(samples: Sequence[float], positive_name: str, negative_name: str) -> str:
    s = _summarize_samples(samples)
    q025, q975 = s["q025"], s["q975"]
    pp, pn = s["P_positive"], s["P_negative"]
    if np.isfinite(q025) and q025 > 0:
        return positive_name + "_supported"
    if np.isfinite(q975) and q975 < 0:
        return negative_name + "_supported"
    if np.isfinite(pp) and pp >= 0.80:
        return positive_name + "_tendency"
    if np.isfinite(pn) and pn >= 0.80:
        return negative_name + "_tendency"
    return "unresolved"


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> Tuple[int, float]:
    lo, hi = max(a0, b0), min(a1, b1)
    overlap = max(0, hi - lo + 1)
    denom = max(1, min(a1 - a0 + 1, b1 - b0 + 1))
    return int(overlap), float(overlap / denom)


# -----------------------------------------------------------------------------
# Accepted-window registry handling
# -----------------------------------------------------------------------------

ANCHOR_ALIASES = ["anchor_day", "center_day", "peak_day", "day_center", "window_center", "center", "day"]
START_ALIASES = ["system_window_start", "window_start", "start_day", "band_start", "accepted_start", "support_start", "start"]
END_ALIASES = ["system_window_end", "window_end", "end_day", "band_end", "accepted_end", "support_end", "end"]
SUPPORT_ALIASES = ["bootstrap_support", "support", "bootstrap_match_fraction", "match_fraction", "retention_support", "support_fraction"]
STATUS_ALIASES = ["accepted_status", "status", "decision", "support_class", "accepted_flag", "retained", "is_accepted"]
ID_ALIASES = ["window_id", "id", "name", "candidate_id", "window_name"]


def _find_col(cols: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a in cols:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _accepted_like(df: pd.DataFrame) -> pd.Series:
    support_col = _find_col(df.columns, SUPPORT_ALIASES)
    status_col = _find_col(df.columns, STATUS_ALIASES)
    ok = pd.Series(False, index=df.index)
    if support_col:
        sup = pd.to_numeric(df[support_col], errors="coerce")
        ok = ok | (sup >= 0.95)
    if status_col:
        txt = df[status_col].astype(str).str.lower()
        ok = ok | txt.str.contains("accepted|accepted_95|significant|retained|pass|true|1", regex=True, na=False)
    # If there is no status/support, do not auto accept; user should specify a registry.
    return ok


def _registry_signature(df: pd.DataFrame) -> Optional[Tuple[Tuple[int, int, int], ...]]:
    ac, sc, ec = _find_col(df.columns, ANCHOR_ALIASES), _find_col(df.columns, START_ALIASES), _find_col(df.columns, END_ALIASES)
    if not (ac and sc and ec):
        return None
    ok = _accepted_like(df)
    rows = df.loc[ok].copy()
    if rows.empty:
        return None
    triples = []
    for _, r in rows.iterrows():
        try:
            triples.append((int(round(float(r[ac]))), int(round(float(r[sc]))), int(round(float(r[ec])))))
        except Exception:
            continue
    if not triples:
        return None
    return tuple(sorted(set(triples)))


def _scan_window_sources(v7_root: Path, cfg: MultiWinConfig) -> Tuple[pd.DataFrame, Optional[Path]]:
    outputs = v7_root / "outputs"
    patterns = ["**/*accepted*window*.csv", "**/*window*registry*.csv", "**/*bootstrap*window*.csv", "**/*stage_partition_main_windows*.csv", "**/*window_support_audit*.csv"]
    paths: List[Path] = []
    if outputs.exists():
        for pat in patterns:
            paths.extend(outputs.glob(pat))
    paths = sorted(set(paths))
    rows = []
    sig_to_paths: Dict[Tuple[Tuple[int, int, int], ...], List[Path]] = {}
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            rows.append({"candidate_path": str(p), "read_status": f"error:{exc}", "selection_status": "rejected", "selection_reason": "read_error"})
            continue
        ac, sc, ec = _find_col(df.columns, ANCHOR_ALIASES), _find_col(df.columns, START_ALIASES), _find_col(df.columns, END_ALIASES)
        supc, statc = _find_col(df.columns, SUPPORT_ALIASES), _find_col(df.columns, STATUS_ALIASES)
        ok = _accepted_like(df)
        sig = _registry_signature(df)
        row = {
            "candidate_path": str(p),
            "read_status": "ok",
            "has_anchor_day": bool(ac),
            "has_start_end": bool(sc and ec),
            "has_support": bool(supc),
            "has_status": bool(statc),
            "n_rows": int(len(df)),
            "n_accepted_like_rows": int(ok.sum()),
            "signature": repr(sig) if sig else "",
            "selection_status": "candidate" if sig else "rejected",
            "selection_reason": "has accepted-like rows with anchor/start/end" if sig else "missing accepted-like rows or required columns",
        }
        rows.append(row)
        if sig:
            sig_to_paths.setdefault(sig, []).append(p)
    audit = pd.DataFrame(rows)
    if not sig_to_paths:
        return audit, None
    # If several files produce exactly the same accepted windows, it is safe to choose one.
    if len(sig_to_paths) == 1:
        sig = next(iter(sig_to_paths))
        chosen = sorted(sig_to_paths[sig], key=lambda x: (len(str(x)), str(x)))[0]
        audit.loc[audit["candidate_path"] == str(chosen), "selection_status"] = "selected"
        audit.loc[audit["candidate_path"] == str(chosen), "selection_reason"] = "unique accepted-window signature"
        return audit, chosen
    # If more than one distinct signature exists, do not guess.
    return audit, None



def _hardcoded_accepted_windows() -> List[AcceptedWindow]:
    """Return the accepted/significant windows used in the current V7 workflow.

    The user has not used an accepted_windows table in this project.  Therefore the
    default path must not scan or require a registry.  These four windows match the
    V7 accepted/significant windows already used by the preceding W45 and multiwin
    discussions.  They are deliberately centralized here so all scope derivation
    remains transparent in window_scope_registry.
    """
    source = "hardcoded_current_V7_significant_windows"
    return [
        AcceptedWindow("W045", 45, 40, 48, 0.95, "hardcoded_accepted", source),
        AcceptedWindow("W081", 81, 75, 87, 0.95, "hardcoded_accepted", source),
        AcceptedWindow("W113", 113, 108, 118, 0.95, "hardcoded_accepted", source),
        AcceptedWindow("W160", 160, 155, 165, 0.95, "hardcoded_accepted", source),
    ]


def _write_hardcoded_window_audit(out_cross: Path, cfg: MultiWinConfig, selected: bool = True) -> None:
    rows = []
    for w in _hardcoded_accepted_windows():
        rows.append({
            "candidate_path": w.source_file,
            "read_status": "not_applicable_hardcoded",
            "has_anchor_day": True,
            "has_start_end": True,
            "has_support": True,
            "has_status": True,
            "n_rows": 4,
            "n_accepted_like_rows": 4,
            "selection_status": "selected" if selected else "available",
            "selection_reason": "default hardcoded windows; no accepted_windows registry required",
            "window_id": w.window_id,
            "anchor_day": w.anchor_day,
            "system_window_start": w.system_window_start,
            "system_window_end": w.system_window_end,
        })
    _safe_to_csv(pd.DataFrame(rows), out_cross / "accepted_window_source_candidates_v7_z_multiwin_a.csv")

def _load_accepted_windows(v7_root: Path, out_cross: Path, cfg: MultiWinConfig) -> List[AcceptedWindow]:
    # Default: do NOT require or scan an accepted_windows table.  The current
    # project workflow has been using the four confirmed V7 significant windows
    # directly, and the conservative W45 run needs those neighboring windows to
    # define detector/baseline scope.  Registry/auto modes are optional overrides.
    source_mode = (cfg.window_source or "hardcoded").strip().lower()
    if source_mode == "hardcoded" and not cfg.accepted_window_registry:
        wins = _hardcoded_accepted_windows()
        _write_hardcoded_window_audit(out_cross, cfg, selected=True)
        return sorted(wins, key=lambda w: (w.system_window_start, w.anchor_day))
    if cfg.accepted_window_registry:
        source = Path(cfg.accepted_window_registry)
        source_audit = pd.DataFrame([{"candidate_path": str(source), "selection_status": "selected", "selection_reason": "V7_MULTI_ACCEPTED_WINDOW_REGISTRY override"}])
    elif source_mode == "auto":
        source_audit, source = _scan_window_sources(v7_root, cfg)
    else:
        raise RuntimeError(
            f"Unsupported V7_MULTI_WINDOW_SOURCE={cfg.window_source!r}. Use hardcoded, registry, or auto. "
            "Default hardcoded mode does not require an accepted_windows table."
        )
    _safe_to_csv(source_audit, out_cross / "accepted_window_source_candidates_v7_z_multiwin_a.csv")
    if source is None or not Path(source).exists():
        raise RuntimeError(
            "Could not uniquely locate an accepted-window registry. This should only occur in "
            "V7_MULTI_WINDOW_SOURCE=auto or registry mode. For the default project workflow, leave "
            "V7_MULTI_WINDOW_SOURCE unset so the hardcoded W045/W081/W113/W160 windows are used."
        )
    df = pd.read_csv(source)
    ac, sc, ec = _find_col(df.columns, ANCHOR_ALIASES), _find_col(df.columns, START_ALIASES), _find_col(df.columns, END_ALIASES)
    supc, statc, idc = _find_col(df.columns, SUPPORT_ALIASES), _find_col(df.columns, STATUS_ALIASES), _find_col(df.columns, ID_ALIASES)
    if not (ac and sc and ec):
        raise ValueError(f"accepted-window registry lacks required anchor/start/end columns: {source}")
    ok = _accepted_like(df)
    rows = df.loc[ok].copy()
    if rows.empty:
        raise ValueError(f"accepted-window registry has no accepted/significant rows: {source}")
    wins: List[AcceptedWindow] = []
    for idx, r in rows.iterrows():
        anchor = int(round(float(r[ac])))
        start = int(round(float(r[sc])))
        end = int(round(float(r[ec])))
        support = float(pd.to_numeric(pd.Series([r[supc]]), errors="coerce").iloc[0]) if supc else float("nan")
        status = str(r[statc]) if statc else ("accepted_by_support" if np.isfinite(support) and support >= 0.95 else "accepted")
        wid_raw = str(r[idc]) if idc else f"W{anchor:03d}"
        # Make stable compact window id.
        wid = wid_raw if wid_raw and wid_raw.lower() != "nan" else f"W{anchor:03d}"
        wid = wid.replace(" ", "_")
        wins.append(AcceptedWindow(wid, anchor, start, end, support, status, str(source)))
    wins = sorted(wins, key=lambda w: (w.system_window_start, w.anchor_day))
    return wins


def _build_window_scopes(wins: List[AcceptedWindow], cfg: MultiWinConfig) -> Tuple[List[WindowScope], pd.DataFrame]:
    scopes: List[WindowScope] = []
    validity_rows = []
    for i, w in enumerate(wins):
        prev_w = wins[i - 1] if i > 0 else None
        next_w = wins[i + 1] if i < len(wins) - 1 else None
        c0_pre_start = cfg.season_start if prev_w is None else prev_w.system_window_end + 1
        c0_pre_end = w.system_window_start - 1
        c0_post_start = w.system_window_end + 1
        c0_post_end = cfg.season_end if next_w is None else next_w.system_window_start - 1
        c1_pre_start = c0_pre_start if prev_w is None else c0_pre_start + cfg.baseline_buffer_days
        c1_pre_end = w.system_window_start - cfg.baseline_buffer_days - 1
        c1_post_start = w.system_window_end + cfg.baseline_buffer_days + 1
        c1_post_end = c0_post_end if next_w is None else c0_post_end - cfg.baseline_buffer_days
        c2_pre_end = w.system_window_start - cfg.baseline_buffer_days - 1
        c2_pre_start = max(c2_pre_end - cfg.immediate_pre_days + 1, c1_pre_start)
        c2_post_start, c2_post_end = c1_post_start, c1_post_end
        analysis_start, analysis_end = c0_pre_start, c0_post_end
        det_start = cfg.season_start if prev_w is None else prev_w.system_window_end + 1
        det_end = cfg.season_end if next_w is None else next_w.system_window_start - 1 - cfg.detector_neighbor_buffer
        early_start = max(analysis_start, w.system_window_start - cfg.segment_pre_days)
        early_end = w.system_window_start - 1
        core_start = w.system_window_start
        core_end = w.anchor_day
        late_start = w.anchor_day + 1
        late_end = min(analysis_end, w.system_window_end + cfg.segment_late_extra_days)
        # preliminary validity; detailed rows below.
        invalid_reasons = []
        if det_end - det_start + 1 < cfg.min_detector_search_days:
            invalid_reasons.append("short_detector_search")
        if c0_pre_end - c0_pre_start + 1 < 8:
            invalid_reasons.append("short_C0_pre")
        if c0_post_end - c0_post_start + 1 < 8:
            invalid_reasons.append("short_C0_post")
        scope = WindowScope(
            window_id=w.window_id,
            anchor_day=w.anchor_day,
            system_window_start=w.system_window_start,
            system_window_end=w.system_window_end,
            bootstrap_support=w.bootstrap_support,
            accepted_status=w.accepted_status,
            source_file=w.source_file,
            detector_search_start=det_start,
            detector_search_end=det_end,
            detector_neighbor_buffer=cfg.detector_neighbor_buffer,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            C0_pre_start=c0_pre_start,
            C0_pre_end=c0_pre_end,
            C0_post_start=c0_post_start,
            C0_post_end=c0_post_end,
            C1_pre_start=c1_pre_start,
            C1_pre_end=c1_pre_end,
            C1_post_start=c1_post_start,
            C1_post_end=c1_post_end,
            C2_pre_start=c2_pre_start,
            C2_pre_end=c2_pre_end,
            C2_post_start=c2_post_start,
            C2_post_end=c2_post_end,
            early_start=early_start,
            early_end=early_end,
            core_start=core_start,
            core_end=core_end,
            late_start=late_start,
            late_end=late_end,
            is_valid_for_prepost=("short_C0_pre" not in invalid_reasons and "short_C0_post" not in invalid_reasons),
            invalid_reason=";".join(invalid_reasons),
        )
        scopes.append(scope)
        # Validity audit rows.
        checks = [
            ("detector_search", det_start, det_end, cfg.min_detector_search_days),
            ("analysis_range", analysis_start, analysis_end, 20),
            ("C0_pre", c0_pre_start, c0_pre_end, 8),
            ("C0_post", c0_post_start, c0_post_end, 8),
            ("C1_pre", c1_pre_start, c1_pre_end, 5),
            ("C1_post", c1_post_start, c1_post_end, 5),
            ("C2_pre", c2_pre_start, c2_pre_end, 8),
            ("C2_post", c2_post_start, c2_post_end, 5),
            ("early", early_start, early_end, 5),
            ("core", core_start, core_end, 3),
            ("late", late_start, late_end, 5),
        ]
        for typ, s, e, mn in checks:
            length = e - s + 1
            validity_rows.append({
                "window_id": w.window_id,
                "scope_type": typ,
                "start_day": s,
                "end_day": e,
                "length": length,
                "min_required_length": mn,
                "is_valid": bool(length >= mn),
                "invalid_reason": "" if length >= mn else f"short_{typ}_{length}",
            })
    return scopes, pd.DataFrame(validity_rows)


# -----------------------------------------------------------------------------
# Dynamic configs, curve metrics, and detector wrappers
# -----------------------------------------------------------------------------


def _filter_scopes_for_run(scopes: List[WindowScope], cfg: MultiWinConfig) -> Tuple[List[WindowScope], pd.DataFrame]:
    """Return scopes selected for this run while preserving full scope registry upstream.

    Default is W45/anchor-45 only.  Set V7_MULTI_WINDOW_MODE=all to run all windows,
    or V7_MULTI_WINDOW_MODE=list with V7_MULTI_TARGET_WINDOWS to run selected IDs/anchors.
    """
    mode = (cfg.window_mode or "w45").strip().lower()
    rows = []
    if mode == "all":
        for s in scopes:
            rows.append({"window_id": s.window_id, "anchor_day": s.anchor_day, "run_selected": True, "reason": "V7_MULTI_WINDOW_MODE=all"})
        return scopes, pd.DataFrame(rows)
    tokens = [x.strip() for x in (cfg.target_windows or "").split(",") if x.strip()]
    if mode == "w45" and not tokens:
        tokens = ["W045", "45"]
    selected: List[WindowScope] = []
    for s in scopes:
        id_norm = str(s.window_id).lower().replace("w", "").lstrip("0")
        match = False
        for tok in tokens:
            t = tok.lower().replace("w", "").lstrip("0")
            if str(s.window_id).lower() == tok.lower() or id_norm == t:
                match = True
            try:
                if int(round(float(tok.replace("W", "").replace("w", "")))) == int(s.anchor_day):
                    match = True
            except Exception:
                pass
            # Also match windows whose system band contains the target day.
            try:
                tv = int(round(float(tok.replace("W", "").replace("w", ""))))
                if s.system_window_start <= tv <= s.system_window_end:
                    match = True
            except Exception:
                pass
        rows.append({
            "window_id": s.window_id,
            "anchor_day": s.anchor_day,
            "system_window_start": s.system_window_start,
            "system_window_end": s.system_window_end,
            "run_selected": bool(match),
            "reason": f"mode={mode}; targets={cfg.target_windows}",
        })
        if match:
            selected.append(s)
    if not selected:
        raise RuntimeError(
            f"No windows selected for run. mode={mode}, targets={cfg.target_windows}. "
            "Set V7_MULTI_WINDOW_MODE=all or V7_MULTI_TARGET_WINDOWS to a valid window id/anchor."
        )
    return selected, pd.DataFrame(rows)

def _clean_cfg_for_window(scope: WindowScope, cfg: MultiWinConfig, for_detector: bool = False) -> clean.CleanConfig:
    ccfg = clean.CleanConfig()
    ccfg.w45_start = scope.system_window_start
    ccfg.w45_end = scope.system_window_end
    ccfg.anchor_day = scope.anchor_day
    ccfg.detection_start = scope.detector_search_start
    ccfg.detection_end = scope.detector_search_end
    ccfg.curve_start = scope.analysis_start
    ccfg.curve_end = scope.analysis_end
    ccfg.compare_start = scope.early_start
    ccfg.compare_end = scope.late_end
    ccfg.early_start = scope.early_start
    ccfg.early_end = scope.early_end
    ccfg.core_start = scope.core_start
    ccfg.core_end = scope.core_end
    ccfg.late_start = scope.late_start
    ccfg.late_end = scope.late_end
    ccfg.detector_width = cfg.detector_width
    ccfg.detector_min_size = cfg.detector_min_size
    ccfg.peak_min_distance = cfg.peak_min_distance
    ccfg.max_peaks_per_object = cfg.max_peaks_per_object
    ccfg.band_max_half_width = cfg.band_max_half_width
    ccfg.band_min_half_width = cfg.band_min_half_width
    ccfg.band_score_ratio = cfg.band_score_ratio
    ccfg.band_floor_quantile = cfg.band_floor_quantile
    ccfg.bootstrap_n = cfg.bootstrap_n
    ccfg.random_seed = cfg.random_seed
    ccfg.peak_match_days = cfg.peak_match_days
    ccfg.low_dynamic_range_eps = cfg.low_dynamic_range_eps
    ccfg.skip_figures = True
    return ccfg


def _mirror_cfg_for_window(scope: WindowScope, cfg: MultiWinConfig) -> mirror2d.Mirror2DConfig:
    mcfg = mirror2d.Mirror2DConfig()
    mcfg.w45_start = scope.system_window_start
    mcfg.w45_end = scope.system_window_end
    mcfg.anchor_day = scope.anchor_day
    mcfg.curve_start = scope.analysis_start
    mcfg.curve_end = scope.analysis_end
    mcfg.compare_start = scope.early_start
    mcfg.compare_end = scope.late_end
    mcfg.early_start = scope.early_start
    mcfg.early_end = scope.early_end
    mcfg.core_start = scope.core_start
    mcfg.core_end = scope.core_end
    mcfg.late_start = scope.late_start
    mcfg.late_end = scope.late_end
    mcfg.bootstrap_n = cfg.bootstrap_n
    mcfg.random_seed = cfg.random_seed
    mcfg.low_dynamic_range_eps = cfg.low_dynamic_range_eps
    mcfg.skip_figures = True
    return mcfg


def _window_support_class(support: float) -> str:
    if np.isfinite(support) and support >= 0.95:
        return "accepted_window"
    if np.isfinite(support) and support >= 0.80:
        return "candidate_window"
    if np.isfinite(support) and support >= 0.50:
        return "weak_window"
    return "unstable_window"



# -----------------------------------------------------------------------------
# Original V7-z ruptures.Window detector helpers
# -----------------------------------------------------------------------------

def _import_ruptures():
    try:
        import ruptures as rpt  # type: ignore
        return rpt
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "V7-z-multiwin-a hotfix requires ruptures for the original V7-z detector. "
            "Install ruptures in the active environment or use the previously generated outputs only."
        ) from exc


def _extract_local_peaks_from_profile(profile: pd.Series, min_distance_days: int) -> pd.DataFrame:
    cols = ["peak_id", "peak_day", "peak_score", "peak_prominence", "peak_rank"]
    try:
        from scipy.signal import find_peaks, peak_prominences  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("V7-z-multiwin-a hotfix requires scipy.signal for peak extraction.") from exc
    if profile is None or profile.empty:
        return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
    if peaks.size == 0:
        return pd.DataFrame(columns=cols)
    prominences, _, _ = peak_prominences(values, peaks)
    rows = []
    for pidx, prom in zip(peaks, prominences):
        rows.append({
            "peak_id": "LP000",
            "peak_day": int(s.index[int(pidx)]),
            "peak_score": float(values[int(pidx)]),
            "peak_prominence": float(prom),
        })
    df = pd.DataFrame(rows).sort_values(["peak_score", "peak_prominence", "peak_day"], ascending=[False, False, True]).reset_index(drop=True)
    df["peak_rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["peak_id"] = [f"CP{i:03d}" for i in range(1, len(df) + 1)]
    return df[cols]


def _finite_day_subset_matrix(matrix: np.ndarray, start_day: int, end_day: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(matrix, dtype=float)
    n_days = X.shape[0]
    lo = max(0, int(start_day))
    hi = min(n_days - 1, int(end_day))
    if lo > hi:
        return X[0:0], np.asarray([], dtype=int)
    days = np.arange(lo, hi + 1, dtype=int)
    sub = X[days]
    valid = np.any(np.isfinite(sub), axis=1)
    return sub[valid], days[valid]



def _zscore_features_v7z(matrix: np.ndarray) -> np.ndarray:
    """Feature-wise z-score used by original V7-z raw/profile detector.

    Original V7-z did not feed raw climatological profiles directly into
    ruptures.Window.  It first averaged across sampled years to obtain a
    day x feature climatological matrix, then standardized each feature along
    day.  This helper restores that exact detector-input contract for both
    observed and bootstrap detector passes.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected day x feature matrix, got shape={x.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
    sd = np.asarray(sd, dtype=float)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    return (x - mu) / sd


def _raw_state_matrix_v7z_from_year_cube(year_cube: np.ndarray, sampled_year_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Build original V7-z raw/profile detector state from year x day x feature cube.

    Parameters
    ----------
    year_cube:
        Profile cube with shape (year, day, feature).  This is the object
        lat-profile cube built from the smoothed field.
    sampled_year_indices:
        Optional paired year-bootstrap indices.  When provided, the selected
        years are averaged before feature-wise z-scoring, matching the original
        V7-z bootstrap semantics.
    """
    cube = np.asarray(year_cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected year x day x feature cube, got shape={cube.shape}")
    if sampled_year_indices is not None:
        cube = cube[np.asarray(sampled_year_indices, dtype=int), :, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clim = np.nanmean(cube, axis=0)
    return _zscore_features_v7z(clim)

def _map_profile_index(profile_raw: pd.Series, day_index: np.ndarray) -> pd.Series:
    if profile_raw is None or profile_raw.empty:
        return pd.Series(dtype=float, name="detector_score")
    out = {}
    n = len(day_index)
    for local_idx, val in profile_raw.items():
        try:
            li = int(local_idx)
        except Exception:
            continue
        if 0 <= li < n:
            out[int(day_index[li])] = float(val)
    return pd.Series(out, name="detector_score", dtype=float).sort_index()


def _run_original_ruptures_window(state_matrix: np.ndarray, cfg: MultiWinConfig, day_index: np.ndarray) -> pd.Series:
    rpt = _import_ruptures()
    signal = np.asarray(state_matrix, dtype=float)
    if signal.shape[0] < max(2 * int(cfg.detector_width), 3):
        return pd.Series(dtype=float, name="detector_score")
    algo = rpt.Window(width=int(cfg.detector_width), model="l2", min_size=int(cfg.detector_min_size), jump=1).fit(signal)
    # Keep the original V7-z semantics: run predict(pen=4.0) so algo.score is built the same way.
    try:
        _ = algo.predict(pen=4.0)
    except Exception:
        # In rare edge cases score may still be available after fit; if not, return empty.
        pass
    score = getattr(algo, "score", None)
    if score is None:
        return pd.Series(dtype=float, name="detector_score")
    arr = np.asarray(score, dtype=float).ravel()
    width_half = int(algo.width // 2)
    idx = np.arange(width_half, width_half + len(arr), dtype=int)
    profile_raw = pd.Series(arr, index=idx, name="detector_score")
    return _map_profile_index(profile_raw, day_index)


def _build_original_band(profile: pd.Series, peak_day: int, cfg: MultiWinConfig) -> Dict[str, object]:
    if profile is None or profile.empty or int(peak_day) not in profile.index:
        return {
            "band_start_day": int(peak_day),
            "band_end_day": int(peak_day),
            "support_floor": np.nan,
            "left_stop_reason": "missing_profile",
            "right_stop_reason": "missing_profile",
        }
    s = profile.sort_index().astype(float)
    peak_score = float(s.loc[int(peak_day)])
    finite = s[np.isfinite(s)]
    if finite.empty:
        floor = np.nan
    else:
        floor = float(max(np.nanquantile(finite.to_numpy(), cfg.band_floor_quantile), peak_score * cfg.band_score_ratio))
    min_lo = int(peak_day) - int(cfg.band_min_half_width)
    min_hi = int(peak_day) + int(cfg.band_min_half_width)
    lo = int(peak_day)
    while lo - 1 in s.index and (int(peak_day) - (lo - 1)) <= int(cfg.band_max_half_width):
        if lo - 1 <= min_lo or float(s.loc[lo - 1]) >= floor:
            lo -= 1
        else:
            break
    hi = int(peak_day)
    while hi + 1 in s.index and ((hi + 1) - int(peak_day)) <= int(cfg.band_max_half_width):
        if hi + 1 >= min_hi or float(s.loc[hi + 1]) >= floor:
            hi += 1
        else:
            break
    return {
        "band_start_day": int(lo),
        "band_end_day": int(hi),
        "support_floor": floor,
        "left_stop_reason": "floor_or_width",
        "right_stop_reason": "floor_or_width",
    }


def _run_original_v7z_detector_for_profile(X: np.ndarray, cfg: MultiWinConfig, scope: WindowScope, object_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the original V7-z ruptures.Window detector on one climatological profile matrix.

    X is day x feature.  The detector is restricted to scope.detector_search_range,
    but candidate relations are evaluated against scope.system_window.
    """
    sub, days = _finite_day_subset_matrix(X, scope.detector_search_start, scope.detector_search_end)
    profile = _run_original_ruptures_window(sub, cfg, days)
    # Full score table across detector search days, keeping NaNs for invalid/no-score days.
    all_days = np.arange(max(0, scope.detector_search_start), min(X.shape[0] - 1, scope.detector_search_end) + 1, dtype=int)
    score_df = pd.DataFrame({"day": all_days})
    if not profile.empty:
        score_df = score_df.merge(profile.rename("detector_score").reset_index().rename(columns={"index": "day"}), on="day", how="left")
    else:
        score_df["detector_score"] = np.nan
    score_df["score_valid"] = np.isfinite(score_df["detector_score"].to_numpy(dtype=float))
    score_df["object"] = object_name

    peaks = _extract_local_peaks_from_profile(profile, cfg.peak_min_distance)
    rows = []
    for _, p in peaks.head(int(cfg.max_peaks_per_object)).iterrows():
        band = _build_original_band(profile, int(p["peak_day"]), cfg)
        ov, ov_frac = _interval_overlap(int(band["band_start_day"]), int(band["band_end_day"]), scope.system_window_start, scope.system_window_end)
        rows.append({
            "object": object_name,
            "candidate_id": p["peak_id"],
            "peak_day": int(p["peak_day"]),
            "band_start_day": int(band["band_start_day"]),
            "band_end_day": int(band["band_end_day"]),
            "peak_score": float(p["peak_score"]),
            "peak_prominence": float(p["peak_prominence"]),
            "peak_rank": int(p["peak_rank"]),
            "overlap_days_with_W45": ov,
            "overlap_fraction_with_W45": ov_frac,
            "left_stop_reason": band.get("left_stop_reason", ""),
            "right_stop_reason": band.get("right_stop_reason", ""),
        })
    return score_df, pd.DataFrame(rows)


def _match_candidate_peak(bcand: pd.DataFrame, observed_day: int, radius: int) -> bool:
    if bcand is None or bcand.empty:
        return False
    return bool(np.any(np.abs(bcand["peak_day"].to_numpy(dtype=float) - int(observed_day)) <= int(radius)))


def _is_system_relevant_candidate(row: pd.Series, scope: WindowScope) -> bool:
    peak = int(row["peak_day"])
    # Current-window relevance is based on PEAK DAY, not broad candidate-band overlap.
    # Band overlap alone can incorrectly promote far-pre/far-post peaks whose wide
    # bands graze the system window.  Those peaks are retained as secondary signals
    # but cannot replace the current-window-relevant main candidate.
    return bool(scope.early_start <= peak <= scope.late_end)

def _relation_to_system(peak_day: int, scope: WindowScope) -> str:
    if peak_day < scope.early_start:
        return "pre_window"
    if scope.early_start <= peak_day <= scope.early_end:
        return "front_or_early"
    if scope.system_window_start <= peak_day <= scope.system_window_end:
        return "within_system_window"
    if scope.late_start <= peak_day <= scope.late_end:
        return "late_or_catchup"
    if peak_day > scope.late_end:
        return "post_window"
    return "near_boundary"


def _select_main_candidate(cand: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    """Select the current system-window-relevant main candidate.

    Observed candidates receive ``support_class`` after the paired-bootstrap support
    pass.  Bootstrap candidates are single-resample detector outputs and therefore
    do **not** have ``support_class``.  Selection must be robust to both tables.

    Main rule: choose from system-window-relevant candidates first.  Bootstrap
    candidates use the same relevance/proximity/score logic, but with neutral
    support tier rather than crashing or inventing support.
    """
    base_cols = {
        "object": None,
        "selected_candidate_id": None,
        "selected_peak_day": np.nan,
        "selected_window_start": np.nan,
        "selected_window_end": np.nan,
        "selected_role": "unresolved",
        "support_class": "unavailable",
        "selection_reason": "no_candidates",
        "excluded_candidates": "",
        "early_secondary_candidates": "",
        "late_secondary_candidates": "",
    }
    if cand is None or cand.empty:
        return pd.DataFrame([base_cols])
    c = cand.copy()

    # Bootstrap candidate tables do not carry observed bootstrap support yet.
    # Use a neutral support class instead of failing.
    if "support_class" not in c.columns:
        c["support_class"] = "bootstrap_unscored"
    c["support_tier"] = c["support_class"].map({
        "accepted_window": 3,
        "candidate_window": 2,
        "weak_window": 1,
        "unstable_window": 0,
    }).fillna(0)

    # Defensive defaults for candidate tables from different detector passes.
    if "candidate_id" not in c.columns:
        c["candidate_id"] = [f"CP{i+1:03d}" for i in range(len(c))]
    if "peak_score" not in c.columns:
        c["peak_score"] = np.nan
    if "band_start_day" not in c.columns:
        c["band_start_day"] = c["peak_day"]
    if "band_end_day" not in c.columns:
        c["band_end_day"] = c["peak_day"]
    if "object" not in c.columns:
        c["object"] = None

    c["relation"] = c["peak_day"].apply(lambda x: _relation_to_system(int(x), scope))
    c["system_relevant"] = c.apply(lambda r: _is_system_relevant_candidate(r, scope), axis=1)
    c["overlap_sys"] = c.apply(lambda r: _interval_overlap(int(r["band_start_day"]), int(r["band_end_day"]), scope.system_window_start, scope.system_window_end)[1], axis=1)
    c["distance_to_anchor"] = (c["peak_day"] - scope.anchor_day).abs()
    c["distance_to_system_band"] = c["peak_day"].apply(
        lambda d: 0 if scope.system_window_start <= int(d) <= scope.system_window_end else min(abs(int(d) - scope.system_window_start), abs(int(d) - scope.system_window_end))
    )

    main_pool = c[c["system_relevant"]].copy()
    reason = "system_relevant_first"
    if main_pool.empty:
        main_pool = c.copy()
        reason = "fallback_no_system_relevant_candidate"

    main_pool["selection_score"] = (
        main_pool["overlap_sys"] * 500
        + main_pool["support_tier"] * 100
        - main_pool["distance_to_anchor"] * 2
        - main_pool["distance_to_system_band"]
        + main_pool["peak_score"].rank(pct=True).fillna(0) * 10
    )
    row = main_pool.sort_values(["selection_score", "support_tier", "peak_score"], ascending=False).iloc[0]

    def _cand_label(r: pd.Series) -> str:
        return f"{r.get('candidate_id', '')}@{int(r['peak_day'])}:{r.get('support_class', 'unavailable')}"

    excluded = ";".join([_cand_label(r) for _, r in c.iterrows() if r.get("candidate_id") != row.get("candidate_id")])
    early = ";".join([_cand_label(r) for _, r in c.iterrows() if int(r["peak_day"]) < scope.early_start])
    late = ";".join([_cand_label(r) for _, r in c.iterrows() if int(r["peak_day"]) > scope.late_end])
    return pd.DataFrame([{
        "object": row.get("object", None),
        "selected_candidate_id": row.get("candidate_id", None),
        "selected_peak_day": int(row["peak_day"]),
        "selected_window_start": int(row["band_start_day"]),
        "selected_window_end": int(row["band_end_day"]),
        "selected_role": row["relation"],
        "support_class": row.get("support_class", "unavailable"),
        "selection_reason": reason,
        "excluded_candidates": excluded,
        "early_secondary_candidates": early,
        "late_secondary_candidates": late,
    }])


def _select_boot_candidate_day(cand: pd.DataFrame, scope: WindowScope) -> float:
    if cand is None or cand.empty:
        return float("nan")
    sel = _select_main_candidate(cand, scope)
    if sel.empty:
        return float("nan")
    return float(sel["selected_peak_day"].iloc[0])


def _make_bootstrap_indices(ny: int, scope: WindowScope, cfg: MultiWinConfig) -> List[np.ndarray]:
    """Shared paired-year bootstrap indices for W45 profile diagnostics.

    The same year-index resamples are used by the object-window bootstrap and the
    pre/post curve-order bootstrap so peak/order and curve/order diagnostics are
    paired at the resample level.
    """
    rng = np.random.default_rng(cfg.random_seed + int(scope.anchor_day))
    return [rng.integers(0, ny, size=ny) for _ in range(cfg.bootstrap_n)]


def _run_detector_and_bootstrap(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scope: WindowScope,
    cfg: MultiWinConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run profile object-window detector with the original V7-z ruptures.Window semantics.

    This replaces the previous multiwin-a approximation based on left/right mean distance.
    The goal is W45 regression: when run on W045 with the same data, candidate timing should
    be comparable to the original V7-z W45 detector before any cross-window interpretation.
    """
    score_rows = []
    cand_rows = []
    boot_days_by_obj: Dict[str, List[float]] = {obj: [] for obj in profiles}
    boot_peak_rows: List[dict] = []

    # Observed candidates.  IMPORTANT: original V7-z raw/profile detector used
    # feature-wise z-scored climatological profiles, not raw climatological
    # profiles.  Do not replace this with unstandardized raw profile input.
    for obj, (prof_by_year, _target_lat, _weights) in profiles.items():
        state = _raw_state_matrix_v7z_from_year_cube(prof_by_year)
        scores, cand = _run_original_v7z_detector_for_profile(state, cfg, scope, obj)
        cand_rows.append(cand)
        score_rows.append(scores)
    cand_df = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame()
    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()

    ny = next(iter(profiles.values()))[0].shape[0]
    boot_indices = _make_bootstrap_indices(ny, scope, cfg)
    candidate_keys = []
    for _, r in cand_df.iterrows():
        candidate_keys.append((r["object"], r["candidate_id"], int(r["peak_day"])))
    support_counts = {(obj, cid): 0 for obj, cid, _ in candidate_keys}

    # Bootstrap candidate support and selected-peak distributions using the SAME selector as observed.
    for ib, idx in enumerate(boot_indices):
        for obj, (prof_by_year, _target_lat, _weights) in profiles.items():
            state = _raw_state_matrix_v7z_from_year_cube(prof_by_year, idx)
            _bscores, bcand = _run_original_v7z_detector_for_profile(state, cfg, scope, obj)
            if bcand.empty:
                selected_boot_day = np.nan
                boot_days_by_obj[obj].append(np.nan)
                boot_peak_rows.append({"window_id": scope.window_id, "bootstrap_id": ib, "object": obj, "selected_peak_day": np.nan})
                continue
            selected_boot_day = _select_boot_candidate_day(bcand, scope)
            boot_days_by_obj[obj].append(selected_boot_day)
            boot_peak_rows.append({"window_id": scope.window_id, "bootstrap_id": ib, "object": obj, "selected_peak_day": selected_boot_day})
            for _, r in cand_df[cand_df["object"] == obj].iterrows():
                if _match_candidate_peak(bcand, int(r["peak_day"]), cfg.peak_match_days):
                    support_counts[(obj, r["candidate_id"])] += 1
        if cfg.log_every_bootstrap > 0 and (ib + 1) % cfg.log_every_bootstrap == 0:
            _log(f"  bootstrap detector {scope.window_id}: {ib + 1}/{cfg.bootstrap_n}")

    if not cand_df.empty:
        cand_df = cand_df.copy()
        cand_df["bootstrap_support"] = cand_df.apply(lambda r: support_counts.get((r["object"], r["candidate_id"]), 0) / max(1, cfg.bootstrap_n), axis=1)
        cand_df["support_class"] = cand_df["bootstrap_support"].apply(_window_support_class)
        cand_df["relation_to_system_window"] = cand_df["peak_day"].apply(lambda x: _relation_to_system(int(x), scope))
        cand_df.insert(0, "window_id", scope.window_id)

    # Current-window relevant main selections plus secondary candidate strings.
    selections = []
    for obj in profiles:
        sel = _select_main_candidate(cand_df[cand_df["object"] == obj], scope)
        sel.insert(0, "window_id", scope.window_id)
        selections.append(sel)
    selection_df = pd.concat(selections, ignore_index=True)

    # Selected peak deltas from bootstrap selected days.
    rows = []
    objs = sorted(profiles.keys())
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            va = np.asarray(boot_days_by_obj[a], dtype=float)
            vb = np.asarray(boot_days_by_obj[b], dtype=float)
            delta = vb - va  # positive => A earlier
            s = _summarize_samples(delta)
            obs_a = selection_df.loc[selection_df["object"] == a, "selected_peak_day"].iloc[0]
            obs_b = selection_df.loc[selection_df["object"] == b, "selected_peak_day"].iloc[0]
            obs_delta = float(obs_b - obs_a) if np.isfinite(obs_a) and np.isfinite(obs_b) else np.nan
            rows.append({
                "window_id": scope.window_id,
                "object_A": a,
                "object_B": b,
                "metric_family": "selected_raw_profile_peak_timing",
                "delta_definition": "B_peak_day - A_peak_day; positive means A earlier",
                "delta_observed": obs_delta,
                **{f"delta_{k}": v for k, v in s.items()},
                "decision": _decision_from_samples(delta, "A_earlier", "B_earlier"),
            })
    selected_delta_df = pd.DataFrame(rows)
    boot_peak_days_df = pd.DataFrame(boot_peak_rows)
    return score_df, cand_df, selection_df, selected_delta_df, boot_peak_days_df


def _profile_curves(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scope: WindowScope,
    cfg: MultiWinConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ccfg = _clean_cfg_for_window(scope, cfg)
    valid_bases = [b.to_clean() for b in scope.baselines() if b.is_valid]
    state_rows, growth_rows, branch_rows, single_rows = [], [], [], []
    for obj, (prof_by_year, _, weights) in profiles.items():
        clim = _nanmean(prof_by_year, axis=0)
        st, gr, br = clean._compute_state_growth_for_object(obj, clim, weights, valid_bases, ccfg)
        state_rows.append(st); growth_rows.append(gr); branch_rows.append(br)
        for base, g in st.groupby("baseline_config"):
            for branch, scol, vcol in [("dist", "S_dist", "V_dist"), ("pattern", "S_pattern", "V_pattern")]:
                single_rows.append({
                    "window_id": scope.window_id,
                    "object": obj,
                    "baseline_config": base,
                    "branch": branch,
                    "mean_early": clean._segment_mean(g, scol, ccfg.early_start, ccfg.early_end),
                    "mean_core": clean._segment_mean(g, scol, ccfg.core_start, ccfg.core_end),
                    "mean_late": clean._segment_mean(g, scol, ccfg.late_start, ccfg.late_end),
                    "growth_center": clean._positive_growth_center(g, branch, ccfg),
                    "positive_growth_area": float(np.nansum(np.where(g[vcol].to_numpy(float) > 0, g[vcol].to_numpy(float), 0.0))),
                })
    state = pd.concat(state_rows, ignore_index=True) if state_rows else pd.DataFrame()
    growth = pd.concat(growth_rows, ignore_index=True) if growth_rows else pd.DataFrame()
    branch = pd.concat(branch_rows, ignore_index=True) if branch_rows else pd.DataFrame()
    pair = clean._pairwise_curve_metrics(state, ccfg) if not state.empty else pd.DataFrame()
    if not pair.empty:
        pair.insert(0, "window_id", scope.window_id)
    return state, growth, branch, pd.DataFrame(single_rows), pair


def _build_2d_regions(fields: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    regions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spec in clean.OBJECT_SPECS:
        field = fields[spec.field_role]
        region, _lat_sel, _lon_sel, weights = mirror2d._extract_region_2d(field, lat, lon, spec)
        regions[spec.object_name] = (region, weights)
    return regions


def _compute_2d_for_window(
    regions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    scope: WindowScope,
    cfg: MultiWinConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not cfg.run_2d:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mcfg = _mirror_cfg_for_window(scope, cfg)
    bases = [b.to_2d() for b in scope.baselines() if b.is_valid]
    rows = []
    for obj, (region, weights) in regions.items():
        for base in bases:
            df = mirror2d._compute_2d_curves(region, weights, base, mcfg)
            df.insert(0, "object", obj)
            rows.append(df)
    curves = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if curves.empty:
        return curves, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    curves.insert(0, "window_id", scope.window_id)
    growth = curves[["window_id", "object", "baseline_config", "day", "V_dist_2d", "V_pattern_2d"]].copy()
    single_rows = []
    for (obj, base), g in curves.groupby(["object", "baseline_config"]):
        vals = mirror2d._single_object_metrics_from_curves(g, mcfg)
        for metric, val in vals.items():
            single_rows.append({"window_id": scope.window_id, "object": obj, "baseline_config": base, "metric": metric, "observed": val})
    metric_by_obj_base: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (obj, base), g in curves.groupby(["object", "baseline_config"]):
        metric_by_obj_base[(obj, base)] = mirror2d._single_object_metrics_from_curves(g, mcfg)
    pair_rows = []
    objs = sorted(curves["object"].unique())
    bases_present = sorted(curves["baseline_config"].unique())
    for base in bases_present:
        metric_by_obj = {o: metric_by_obj_base.get((o, base), {}) for o in objs}
        p = mirror2d._pairwise_metrics(metric_by_obj, objs)
        if not p.empty:
            p.insert(0, "window_id", scope.window_id)
            p.insert(1, "basis", "2d_field")
            p["baseline_config"] = base
            pair_rows.append(p)
    pair = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    return curves, growth, pd.DataFrame(single_rows), pair, pd.DataFrame()


def _bootstrap_pairwise_metrics(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    regions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    observed_profile_pair: pd.DataFrame,
    observed_2d_pair: pd.DataFrame,
    scope: WindowScope,
    cfg: MultiWinConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    # Use the same deterministic paired-year bootstrap index generator as the
    # detector bootstrap.  hotfix_06 accidentally referenced `boot_indices`
    # without defining it here, which caused a NameError before any order-test
    # outputs were written.  Keeping the generator seed tied to `scope` and
    # `cfg` preserves paired resampling across objects and aligns the curve
    # bootstrap with the detector bootstrap semantics.
    ny = next(iter(profiles.values()))[0].shape[0]
    boot_indices = _make_bootstrap_indices(ny, scope, cfg)
    ccfg = _clean_cfg_for_window(scope, cfg)
    mcfg = _mirror_cfg_for_window(scope, cfg)
    clean_bases = [b.to_clean() for b in scope.baselines() if b.is_valid]
    mirror_bases = [b.to_2d() for b in scope.baselines() if b.is_valid]
    sample_rows = []
    for ib, idx in enumerate(boot_indices):
        # profile state and pairwise
        st_rows = []
        for obj, (prof_by_year, _, weights) in profiles.items():
            clim = _nanmean(prof_by_year[idx], axis=0)
            st, _gr, _br = clean._compute_state_growth_for_object(obj, clim, weights, clean_bases, ccfg)
            st_rows.append(st)
        bst = pd.concat(st_rows, ignore_index=True) if st_rows else pd.DataFrame()
        bp = clean._pairwise_curve_metrics(bst, ccfg) if not bst.empty else pd.DataFrame()
        for _, r in bp.iterrows():
            sample_rows.append({
                "bootstrap_id": ib,
                "basis": "profile",
                "object_A": r["object_A"],
                "object_B": r["object_B"],
                "baseline_config": r["baseline_config"],
                "metric_family": r["metric_family"],
                "metric_name": r["metric_name"],
                "sample": r["observed"],
            })
        if cfg.run_2d and regions:
            curves_rows = []
            for obj, (region, weights) in regions.items():
                boot_region = region[idx]
                for base in mirror_bases:
                    df = mirror2d._compute_2d_curves(boot_region, weights, base, mcfg)
                    df.insert(0, "object", obj)
                    curves_rows.append(df)
            bcurves = pd.concat(curves_rows, ignore_index=True) if curves_rows else pd.DataFrame()
            if not bcurves.empty:
                objs = sorted(bcurves["object"].unique())
                for base, gg in bcurves.groupby("baseline_config"):
                    metric_by_obj = {obj: mirror2d._single_object_metrics_from_curves(g, mcfg) for obj, g in gg.groupby("object")}
                    bp2 = mirror2d._pairwise_metrics(metric_by_obj, objs)
                    for _, r in bp2.iterrows():
                        sample_rows.append({
                            "bootstrap_id": ib,
                            "basis": "2d_field",
                            "object_A": r["object_A"],
                            "object_B": r["object_B"],
                            "baseline_config": base,
                            "metric_family": r["metric_family"],
                            "metric_name": r.get("metric_name", r["metric_family"]),
                            "sample": r["delta_observed"],
                        })
        if cfg.log_every_bootstrap > 0 and (ib + 1) % cfg.log_every_bootstrap == 0:
            _log(f"  bootstrap curves {scope.window_id}: {ib + 1}/{cfg.bootstrap_n}")
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    summ_rows = []
    for keys, g in samples.groupby(["basis", "object_A", "object_B", "baseline_config", "metric_family", "metric_name"]):
        basis, a, b, base, fam, name = keys
        vals = g["sample"].to_numpy(dtype=float)
        s = _summarize_samples(vals)
        summ_rows.append({
            "window_id": scope.window_id,
            "basis": basis,
            "object_A": a,
            "object_B": b,
            "baseline_config": base,
            "metric_family": fam,
            "metric_name": name,
            **s,
            "decision": _decision_from_samples(vals, "A_direction", "B_direction"),
        })
    summ = pd.DataFrame(summ_rows)
    return summ[summ["basis"] == "profile"].copy(), summ[summ["basis"] == "2d_field"].copy(), samples if cfg.save_bootstrap_samples else None



# -----------------------------------------------------------------------------
# W45 profile-only order tests (hotfix06)
# -----------------------------------------------------------------------------

OBJECT_ORDER_PAIRS: List[Tuple[str, str]] = [
    ("P", "V"), ("P", "H"), ("P", "Je"), ("P", "Jw"),
    ("V", "H"), ("V", "Je"), ("V", "Jw"),
    ("H", "Je"), ("H", "Jw"),
    ("Je", "Jw"),
]


def _empty_order_outputs() -> Dict[str, pd.DataFrame]:
    names = [
        "timing_resolution_audit", "tau_sync_estimate", "pairwise_peak_order_test",
        "pairwise_synchrony_equivalence_test", "pairwise_window_overlap_test",
        "pairwise_state_progress_difference", "pairwise_state_catchup_reversal",
        "object_growth_sign_structure", "object_growth_pulse_structure",
        "pairwise_growth_process_difference", "pairwise_prepost_curve_interpretation",
        "pairwise_order_interpretation_summary",
    ]
    return {name: pd.DataFrame() for name in names}


def _state_for_bootstrap_indices(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scope: WindowScope,
    cfg: MultiWinConfig,
    idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute profile pre/post state and growth curves for one paired year resample."""
    ccfg = _clean_cfg_for_window(scope, cfg)
    valid_bases = [b.to_clean() for b in scope.baselines() if b.is_valid]
    st_rows, gr_rows = [], []
    for obj, (prof_by_year, _target_lat, weights) in profiles.items():
        clim = _nanmean(prof_by_year[idx], axis=0)
        st, gr, _br = clean._compute_state_growth_for_object(obj, clim, weights, valid_bases, ccfg)
        st_rows.append(st)
        gr_rows.append(gr)
    return (
        pd.concat(st_rows, ignore_index=True) if st_rows else pd.DataFrame(),
        pd.concat(gr_rows, ignore_index=True) if gr_rows else pd.DataFrame(),
    )


def _branch_col(branch: str) -> Tuple[str, str]:
    return ("S_dist", "V_dist") if branch == "dist" else ("S_pattern", "V_pattern")


def _range_defs(scope: WindowScope) -> Dict[str, Tuple[int, int]]:
    return {
        "full_analysis": (int(scope.analysis_start), int(scope.analysis_end)),
        "early": (int(scope.early_start), int(scope.early_end)),
        "core": (int(scope.core_start), int(scope.core_end)),
        "late": (int(scope.late_start), int(scope.late_end)),
        "system_window": (int(scope.system_window_start), int(scope.system_window_end)),
        "pre_anchor": (int(scope.early_start), int(scope.anchor_day)),
        "post_anchor": (int(scope.anchor_day + 1), int(scope.late_end)),
    }


def _curve_by_obj_base(state_df: pd.DataFrame, baseline: str, obj: str, col: str) -> Tuple[np.ndarray, np.ndarray]:
    g = state_df[(state_df["baseline_config"] == baseline) & (state_df["object"] == obj)].sort_values("day")
    return g["day"].to_numpy(dtype=int), g[col].to_numpy(dtype=float)


def _align_pair_curves(state_df: pd.DataFrame, baseline: str, a: str, b: str, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da, va = _curve_by_obj_base(state_df, baseline, a, col)
    db, vb = _curve_by_obj_base(state_df, baseline, b, col)
    if da.size == 0 or db.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    amap = dict(zip(da.tolist(), va.tolist()))
    bmap = dict(zip(db.tolist(), vb.tolist()))
    days = np.array(sorted(set(amap).intersection(bmap)), dtype=int)
    return days, np.array([amap[int(d)] for d in days], dtype=float), np.array([bmap[int(d)] for d in days], dtype=float)


def _sign_crossings(days: np.ndarray, delta: np.ndarray) -> Tuple[int, str, str, str]:
    valid = np.isfinite(delta)
    if not np.any(valid):
        return 0, "", "none", "none"
    d = days[valid]
    x = delta[valid]
    signs = np.sign(x)
    # carry non-zero signs across exact zeros for stable crossing count.
    nz = signs != 0
    if not np.any(nz):
        return 0, "", "near_equal", "near_equal"
    signs2 = signs.copy()
    last = 0.0
    for i in range(signs2.size):
        if signs2[i] == 0:
            signs2[i] = last
        else:
            last = signs2[i]
    for i in range(signs2.size - 1, -1, -1):
        if signs2[i] == 0 and i + 1 < signs2.size:
            signs2[i] = signs2[i + 1]
    idx = np.where(np.diff(signs2) != 0)[0]
    crossing_days = ";".join(str(int(d[i + 1])) for i in idx)
    initial = "A" if signs2[0] > 0 else "B" if signs2[0] < 0 else "near_equal"
    final = "A" if signs2[-1] > 0 else "B" if signs2[-1] < 0 else "near_equal"
    return int(len(idx)), crossing_days, initial, final


def _state_diff_metrics(state_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    if state_df is None or state_df.empty:
        return pd.DataFrame()
    baselines = sorted(state_df["baseline_config"].dropna().unique())
    ranges = _range_defs(scope)
    for base in baselines:
        for branch in ["dist", "pattern"]:
            scol, _ = _branch_col(branch)
            for a, b in OBJECT_ORDER_PAIRS:
                days, va, vb = _align_pair_curves(state_df, base, a, b, scol)
                if days.size == 0:
                    continue
                delta_all = va - vb
                crossing_count, crossing_days, initial_leader, final_leader = _sign_crossings(days, delta_all)
                for rname, (r0, r1) in ranges.items():
                    mask = (days >= r0) & (days <= r1)
                    delta = delta_all[mask]
                    dsub = days[mask]
                    if delta.size == 0:
                        continue
                    area_a = float(np.nansum(np.maximum(delta, 0)))
                    area_b = float(np.nansum(np.maximum(-delta, 0)))
                    rows.append({
                        "window_id": scope.window_id,
                        "object_A": a,
                        "object_B": b,
                        "baseline": base,
                        "branch": branch,
                        "range_name": rname,
                        "range_start": int(r0),
                        "range_end": int(r1),
                        "mean_delta_observed": float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan,
                        "area_A_ahead_observed": area_a,
                        "area_B_ahead_observed": area_b,
                        "area_balance_observed": area_a - area_b,
                        "duration_A_ahead": int(np.nansum(delta > 0)),
                        "duration_B_ahead": int(np.nansum(delta < 0)),
                        "crossing_count_full": crossing_count,
                        "crossing_days_full": crossing_days,
                        "initial_leader_full": initial_leader,
                        "final_leader_full": final_leader,
                    })
    return pd.DataFrame(rows)


def _state_catchup_reversal(state_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    if state_df is None or state_df.empty:
        return pd.DataFrame()
    for base in sorted(state_df["baseline_config"].dropna().unique()):
        for branch in ["dist", "pattern"]:
            scol, _ = _branch_col(branch)
            for a, b in OBJECT_ORDER_PAIRS:
                days, va, vb = _align_pair_curves(state_df, base, a, b, scol)
                if days.size == 0:
                    continue
                delta = va - vb
                crossing_count, crossing_days, initial, final = _sign_crossings(days, delta)
                if crossing_count == 0 and initial == "A":
                    rel = "persistent_A_ahead"
                elif crossing_count == 0 and initial == "B":
                    rel = "persistent_B_ahead"
                elif initial == "A" and final == "B":
                    rel = "A_ahead_then_B_catchup_or_reversal"
                elif initial == "B" and final == "A":
                    rel = "B_ahead_then_A_catchup_or_reversal"
                elif crossing_count > 1:
                    rel = "multiple_crossings"
                elif initial == "near_equal" and final == "near_equal":
                    rel = "mostly_parallel_or_equal"
                else:
                    rel = "state_relation_unresolved"
                rows.append({
                    "window_id": scope.window_id,
                    "object_A": a,
                    "object_B": b,
                    "baseline": base,
                    "branch": branch,
                    "initial_leader": initial,
                    "final_leader": final,
                    "crossing_count": crossing_count,
                    "crossing_days": crossing_days,
                    "A_ahead_then_B_catchup": bool(initial == "A" and final == "B"),
                    "B_ahead_then_A_catchup": bool(initial == "B" and final == "A"),
                    "persistent_A_ahead": bool(crossing_count == 0 and initial == "A"),
                    "persistent_B_ahead": bool(crossing_count == 0 and initial == "B"),
                    "state_relation_type": rel,
                })
    return pd.DataFrame(rows)


def _intervals_from_sign(days: np.ndarray, values: np.ndarray, positive: bool = True, merge_gap: int = 1) -> List[Tuple[int, int, float, int, float]]:
    mask = values > 0 if positive else values < 0
    valid = np.isfinite(values) & mask
    if not np.any(valid):
        return []
    ds = days[valid]
    vals = values[valid]
    intervals: List[List[float]] = []
    start = int(ds[0]); end = int(ds[0]); area = abs(float(vals[0])); peak_day = int(ds[0]); peak_val = abs(float(vals[0]))
    for d, v0 in zip(ds[1:], vals[1:]):
        d = int(d); v = abs(float(v0))
        if d - end <= merge_gap + 1:
            end = d; area += v
            if v > peak_val:
                peak_val = v; peak_day = d
        else:
            intervals.append([start, end, area, peak_day, peak_val])
            start = end = d; area = v; peak_day = d; peak_val = v
    intervals.append([start, end, area, peak_day, peak_val])
    return [(int(s), int(e), float(a), int(p), float(pv)) for s, e, a, p, pv in intervals]


def _growth_sign_structure(growth_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    if growth_df is None or growth_df.empty:
        return pd.DataFrame()
    for (obj, base), g in growth_df.groupby(["object", "baseline_config"]):
        g = g.sort_values("day")
        days = g["day"].to_numpy(dtype=int)
        for branch, vcol in [("dist", "V_dist"), ("pattern", "V_pattern")]:
            vals = g[vcol].to_numpy(dtype=float)
            pos = float(np.nansum(np.maximum(vals, 0)))
            neg = float(np.nansum(np.maximum(-vals, 0)))
            gross = pos + neg
            net = float(np.nansum(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else np.nan
            neg_intervals = _intervals_from_sign(days, vals, positive=False, merge_gap=1)
            rows.append({
                "window_id": scope.window_id,
                "object": obj,
                "baseline": base,
                "branch": branch,
                "positive_area_observed": pos,
                "negative_area_observed": neg,
                "negative_ratio_observed": neg / gross if gross > EPS else np.nan,
                "net_change_observed": net,
                "gross_change_observed": gross,
                "efficiency_observed": net / gross if gross > EPS and np.isfinite(net) else np.nan,
                "observed_negative_intervals": ";".join([f"{s}-{e}" for s, e, *_ in neg_intervals]),
            })
    return pd.DataFrame(rows)


def _growth_pulse_structure(growth_df: pd.DataFrame, scope: WindowScope, cfg: MultiWinConfig) -> pd.DataFrame:
    rows: List[dict] = []
    if growth_df is None or growth_df.empty:
        return pd.DataFrame()
    ranges = _range_defs(scope)
    for (obj, base), g in growth_df.groupby(["object", "baseline_config"]):
        g = g.sort_values("day")
        days = g["day"].to_numpy(dtype=int)
        for branch, vcol in [("dist", "V_dist"), ("pattern", "V_pattern")]:
            vals = g[vcol].to_numpy(dtype=float)
            pos = np.maximum(vals, 0)
            intervals = _intervals_from_sign(days, vals, positive=True, merge_gap=cfg.merge_short_gap_days)
            intervals_sorted = sorted(intervals, key=lambda x: x[2], reverse=True)
            primary = intervals_sorted[0] if intervals_sorted else (np.nan, np.nan, 0.0, np.nan, np.nan)
            secondary = intervals_sorted[1] if len(intervals_sorted) > 1 else (np.nan, np.nan, 0.0, np.nan, np.nan)
            def area_in(rname: str) -> float:
                r0, r1 = ranges[rname]
                m = (days >= r0) & (days <= r1)
                return float(np.nansum(pos[m]))
            early = area_in("early"); core = area_in("core"); late = area_in("late")
            pre_anchor = area_in("pre_anchor"); post_anchor = area_in("post_anchor")
            total = float(np.nansum(pos))
            if total <= EPS:
                structure = "weak_or_no_positive_growth"
            elif len(intervals_sorted) >= 2 and secondary[2] / max(primary[2], EPS) >= 0.25:
                structure = "multi_pulse_candidate"
            else:
                vals_area = {"early_dominant": early, "core_dominant": core, "late_dominant": late}
                structure = max(vals_area, key=vals_area.get)
            center = float(np.nansum(days * pos) / total) if total > EPS else np.nan
            rows.append({
                "window_id": scope.window_id,
                "object": obj,
                "baseline": base,
                "branch": branch,
                "n_positive_pulses": int(len(intervals_sorted)),
                "growth_center_positive": center,
                "primary_pulse_start": primary[0],
                "primary_pulse_end": primary[1],
                "primary_pulse_area": primary[2],
                "secondary_pulse_start": secondary[0],
                "secondary_pulse_end": secondary[1],
                "secondary_pulse_area": secondary[2],
                "secondary_primary_area_ratio": secondary[2] / max(primary[2], EPS) if primary[2] > EPS else np.nan,
                "early_positive_area": early,
                "core_positive_area": core,
                "late_positive_area": late,
                "pre_anchor_positive_area": pre_anchor,
                "post_anchor_positive_area": post_anchor,
                "growth_pulse_structure": structure,
            })
    return pd.DataFrame(rows)


def _pairwise_growth_process_from_pulses(pulse_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    if pulse_df is None or pulse_df.empty:
        return pd.DataFrame()
    metrics = [
        "early_positive_area", "core_positive_area", "late_positive_area",
        "pre_anchor_positive_area", "post_anchor_positive_area", "growth_center_positive",
    ]
    for (base, branch), g in pulse_df.groupby(["baseline", "branch"]):
        by_obj = {r["object"]: r for _, r in g.iterrows()}
        for a, b in OBJECT_ORDER_PAIRS:
            if a not in by_obj or b not in by_obj:
                continue
            ra, rb = by_obj[a], by_obj[b]
            for metric in metrics:
                av = float(ra.get(metric, np.nan)); bv = float(rb.get(metric, np.nan))
                if metric == "growth_center_positive":
                    delta = bv - av  # positive => A earlier
                    definition = "B_center - A_center; positive means A earlier"
                else:
                    delta = av - bv  # positive => A larger area
                    definition = "A_area - B_area; positive means A larger"
                rows.append({
                    "window_id": scope.window_id,
                    "object_A": a,
                    "object_B": b,
                    "baseline": base,
                    "branch": branch,
                    "metric": metric,
                    "delta_observed": delta,
                    "delta_definition": definition,
                    "A_growth_pulse_structure": ra.get("growth_pulse_structure", ""),
                    "B_growth_pulse_structure": rb.get("growth_pulse_structure", ""),
                })
    return pd.DataFrame(rows)


def _summarize_boot_metric(samples_df: pd.DataFrame, group_cols: List[str], value_col: str, prefix: str = "") -> pd.DataFrame:
    rows: List[dict] = []
    if samples_df is None or samples_df.empty:
        return pd.DataFrame()
    for keys, g in samples_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = g[value_col].to_numpy(dtype=float)
        s = _summarize_samples(vals)
        rec = dict(zip(group_cols, keys))
        rec.update({f"{prefix}{k}": v for k, v in s.items()})
        rec[f"{prefix}decision"] = _decision_from_samples(vals, "A_direction", "B_direction")
        rows.append(rec)
    return pd.DataFrame(rows)


def _estimate_timing_resolution(selection_df: pd.DataFrame, boot_peak_days_df: pd.DataFrame, cfg: MultiWinConfig, scope: WindowScope) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    pooled: List[float] = []
    if selection_df is None or selection_df.empty or boot_peak_days_df is None or boot_peak_days_df.empty:
        return pd.DataFrame(), pd.DataFrame([{"window_id": scope.window_id, "tau_sync_primary": np.nan, "tau_quality_flag": "insufficient_bootstrap"}])
    for _, r in selection_df.iterrows():
        obj = r["object"]
        obs = float(r["selected_peak_day"])
        vals = boot_peak_days_df[boot_peak_days_df["object"] == obj]["selected_peak_day"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        err = vals - obs if vals.size else np.array([], dtype=float)
        abs_err = np.abs(err)
        pooled.extend(abs_err.tolist())
        rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "observed_peak_day": obs,
            "bootstrap_peak_median": float(np.median(vals)) if vals.size else np.nan,
            "bootstrap_peak_q025": float(np.quantile(vals, 0.025)) if vals.size else np.nan,
            "bootstrap_peak_q975": float(np.quantile(vals, 0.975)) if vals.size else np.nan,
            "abs_error_median": float(np.median(abs_err)) if abs_err.size else np.nan,
            "abs_error_q75": float(np.quantile(abs_err, 0.75)) if abs_err.size else np.nan,
            "abs_error_q90": float(np.quantile(abs_err, 0.90)) if abs_err.size else np.nan,
            "abs_error_q975": float(np.quantile(abs_err, 0.975)) if abs_err.size else np.nan,
            "support_class": r.get("support_class", ""),
            "included_in_tau_estimation": bool(abs_err.size > 0),
        })
    arr = np.asarray(pooled, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        q50 = float(np.quantile(arr, cfg.tau_sync_quantile_low))
        q75 = float(np.quantile(arr, cfg.tau_sync_quantile_primary))
        q90 = float(np.quantile(arr, cfg.tau_sync_quantile_high))
        med_width = np.nanmedian(selection_df["selected_window_end"].to_numpy(float) - selection_df["selected_window_start"].to_numpy(float) + 1)
        flag = "broad_resolution_warning" if np.isfinite(med_width) and q75 > 0.5 * med_width else "normal"
        tau = pd.DataFrame([{
            "window_id": scope.window_id,
            "tau_sync_q50": q50,
            "tau_sync_q75": q75,
            "tau_sync_q90": q90,
            "tau_sync_primary": q75,
            "tau_source": "pooled_q75_abs_bootstrap_peak_error",
            "n_objects_used": int(selection_df.shape[0]),
            "n_bootstrap_values_used": int(arr.size),
            "tau_quality_flag": flag,
        }])
    else:
        tau = pd.DataFrame([{"window_id": scope.window_id, "tau_sync_primary": np.nan, "tau_quality_flag": "insufficient_bootstrap"}])
    return pd.DataFrame(rows), tau


def _pairwise_peak_order(selection_df: pd.DataFrame, boot_peak_days_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    by_sel = {r["object"]: r for _, r in selection_df.iterrows()} if selection_df is not None and not selection_df.empty else {}
    pivot = boot_peak_days_df.pivot(index="bootstrap_id", columns="object", values="selected_peak_day") if boot_peak_days_df is not None and not boot_peak_days_df.empty else pd.DataFrame()
    for a, b in OBJECT_ORDER_PAIRS:
        if a not in by_sel or b not in by_sel or a not in pivot.columns or b not in pivot.columns:
            continue
        obs_a = float(by_sel[a]["selected_peak_day"]); obs_b = float(by_sel[b]["selected_peak_day"])
        delta = pivot[b].to_numpy(dtype=float) - pivot[a].to_numpy(dtype=float)
        s = _summarize_samples(delta)
        p_same = float(np.mean(delta[np.isfinite(delta)] == 0)) if np.isfinite(delta).any() else np.nan
        if np.isfinite(s["q025"]) and s["q025"] > 0:
            dec = "A_peak_earlier_supported"
        elif np.isfinite(s["q975"]) and s["q975"] < 0:
            dec = "B_peak_earlier_supported"
        elif np.isfinite(s["median"]) and s["median"] > 0 and s["P_positive"] > s["P_negative"]:
            dec = "A_peak_earlier_tendency"
        elif np.isfinite(s["median"]) and s["median"] < 0 and s["P_negative"] > s["P_positive"]:
            dec = "B_peak_earlier_tendency"
        else:
            dec = "peak_order_unresolved"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a, "object_B": b,
            "A_peak_day": obs_a, "B_peak_day": obs_b,
            "delta_observed": obs_b - obs_a,
            "delta_median": s["median"], "delta_q025": s["q025"], "delta_q975": s["q975"],
            "P_A_earlier": s["P_positive"], "P_B_earlier": s["P_negative"], "P_same_day": p_same,
            "peak_order_decision": dec,
        })
    return pd.DataFrame(rows)


def _pairwise_synchrony(peak_df: pd.DataFrame, boot_peak_days_df: pd.DataFrame, tau_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    tau_row = tau_df.iloc[0] if tau_df is not None and not tau_df.empty else pd.Series(dtype=float)
    tau50 = float(tau_row.get("tau_sync_q50", np.nan)); tau75 = float(tau_row.get("tau_sync_q75", tau_row.get("tau_sync_primary", np.nan))); tau90 = float(tau_row.get("tau_sync_q90", np.nan))
    tau_primary = float(tau_row.get("tau_sync_primary", tau75))
    tau_flag = str(tau_row.get("tau_quality_flag", ""))
    pivot = boot_peak_days_df.pivot(index="bootstrap_id", columns="object", values="selected_peak_day") if boot_peak_days_df is not None and not boot_peak_days_df.empty else pd.DataFrame()
    for _, r in peak_df.iterrows():
        a, b = r["object_A"], r["object_B"]
        if a not in pivot.columns or b not in pivot.columns:
            continue
        delta = pivot[b].to_numpy(dtype=float) - pivot[a].to_numpy(dtype=float)
        valid = delta[np.isfinite(delta)]
        def p_within(tau: float) -> float:
            return float(np.mean(np.abs(valid) <= tau)) if valid.size and np.isfinite(tau) else np.nan
        q025 = float(r["delta_q025"]); q975 = float(r["delta_q975"])
        if np.isfinite(tau_primary) and np.isfinite(q025) and q025 >= -tau_primary and q975 <= tau_primary and tau_flag != "broad_resolution_warning":
            dec = "synchrony_supported"
        elif p_within(tau_primary) >= 0.80 if np.isfinite(p_within(tau_primary)) else False:
            dec = "synchrony_tendency"
        elif np.isfinite(tau_primary) and ((np.isfinite(q025) and q025 > tau_primary) or (np.isfinite(q975) and q975 < -tau_primary)):
            dec = "synchrony_not_supported"
        else:
            dec = "synchrony_indeterminate"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a, "object_B": b,
            "tau_sync_primary": tau_primary, "tau_sync_q50": tau50, "tau_sync_q75": tau75, "tau_sync_q90": tau90,
            "delta_observed": r["delta_observed"], "delta_median": r["delta_median"], "delta_q025": q025, "delta_q975": q975,
            "P_within_tau_q50": p_within(tau50), "P_within_tau_q75": p_within(tau75), "P_within_tau_q90": p_within(tau90),
            "synchrony_decision": dec, "tau_quality_flag": tau_flag,
        })
    return pd.DataFrame(rows)


def _pairwise_window_overlap(selection_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    by_sel = {r["object"]: r for _, r in selection_df.iterrows()} if selection_df is not None and not selection_df.empty else {}
    for a, b in OBJECT_ORDER_PAIRS:
        if a not in by_sel or b not in by_sel:
            continue
        ra, rb = by_sel[a], by_sel[b]
        a0, a1 = int(ra["selected_window_start"]), int(ra["selected_window_end"])
        b0, b1 = int(rb["selected_window_start"]), int(rb["selected_window_end"])
        lo, hi = max(a0, b0), min(a1, b1)
        overlap_days = max(0, hi - lo + 1)
        union_days = max(a1, b1) - min(a0, b0) + 1
        frac = overlap_days / max(1, union_days)
        if overlap_days <= 0:
            dec = "window_separated"
        elif frac >= 0.60:
            dec = "window_overlap_strong"
        elif frac >= 0.25:
            dec = "window_overlap_partial"
        else:
            dec = "window_overlap_weak"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a, "object_B": b,
            "A_window_start": a0, "A_window_end": a1,
            "B_window_start": b0, "B_window_end": b1,
            "overlap_start": lo if overlap_days > 0 else np.nan,
            "overlap_end": hi if overlap_days > 0 else np.nan,
            "overlap_days": overlap_days,
            "union_days": union_days,
            "overlap_fraction": frac,
            "A_peak_inside_B_window": bool(b0 <= int(ra["selected_peak_day"]) <= b1),
            "B_peak_inside_A_window": bool(a0 <= int(rb["selected_peak_day"]) <= a1),
            "both_peaks_inside_system_window": bool(scope.system_window_start <= int(ra["selected_peak_day"]) <= scope.system_window_end and scope.system_window_start <= int(rb["selected_peak_day"]) <= scope.system_window_end),
            "window_overlap_decision": dec,
        })
    return pd.DataFrame(rows)


def _add_bootstrap_summaries(observed: pd.DataFrame, samples: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    out = observed.copy()
    if samples is None or samples.empty:
        return out
    for v in value_cols:
        rows = []
        for keys, g in samples.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            arr = g[v].to_numpy(dtype=float)
            s = _summarize_samples(arr)
            rec = dict(zip(group_cols, keys))
            rec.update({f"{v}_median": s["median"], f"{v}_q025": s["q025"], f"{v}_q975": s["q975"], f"{v}_P_positive": s["P_positive"], f"{v}_P_negative": s["P_negative"], f"{v}_decision": _decision_from_samples(arr, "A_direction", "B_direction")})
            rows.append(rec)
        summ = pd.DataFrame(rows)
        if not summ.empty:
            out = out.merge(summ, on=group_cols, how="left")
    return out


def _build_w45_profile_order_tests(
    scope: WindowScope,
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    selection_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    boot_peak_days_df: pd.DataFrame,
    profile_state_df: pd.DataFrame,
    profile_growth_df: pd.DataFrame,
    cfg: MultiWinConfig,
) -> Dict[str, pd.DataFrame]:
    if scope.window_id != "W045" or not cfg.run_w45_profile_order_tests:
        return _empty_order_outputs()
    out = _empty_order_outputs()
    timing_df, tau_df = _estimate_timing_resolution(selection_df, boot_peak_days_df, cfg, scope)
    peak_df = _pairwise_peak_order(selection_df, boot_peak_days_df, scope)
    sync_df = _pairwise_synchrony(peak_df, boot_peak_days_df, tau_df, scope) if not peak_df.empty else pd.DataFrame()
    overlap_df = _pairwise_window_overlap(selection_df, scope)
    state_obs = _state_diff_metrics(profile_state_df, scope)
    catchup_df = _state_catchup_reversal(profile_state_df, scope)
    growth_sign_obs = _growth_sign_structure(profile_growth_df, scope)
    growth_pulse_obs = _growth_pulse_structure(profile_growth_df, scope, cfg)
    growth_pair_obs = _pairwise_growth_process_from_pulses(growth_pulse_obs, scope)

    # Bootstrap pre/post curve functionals.  We do not write full bootstrap curves by default.
    ny = next(iter(profiles.values()))[0].shape[0]
    boot_indices = _make_bootstrap_indices(ny, scope, cfg)
    state_samples, growth_sign_samples, growth_pulse_samples, growth_pair_samples = [], [], [], []
    for ib, idx in enumerate(boot_indices):
        st_b, gr_b = _state_for_bootstrap_indices(profiles, scope, cfg, idx)
        sd_b = _state_diff_metrics(st_b, scope)
        if not sd_b.empty:
            sd_b["bootstrap_id"] = ib
            state_samples.append(sd_b[["bootstrap_id", "object_A", "object_B", "baseline", "branch", "range_name", "mean_delta_observed", "area_balance_observed"]])
        gs_b = _growth_sign_structure(gr_b, scope)
        if not gs_b.empty:
            gs_b["bootstrap_id"] = ib
            growth_sign_samples.append(gs_b[["bootstrap_id", "object", "baseline", "branch", "negative_area_observed", "negative_ratio_observed"]])
        gp_b = _growth_pulse_structure(gr_b, scope, cfg)
        if not gp_b.empty:
            gp_b["bootstrap_id"] = ib
            growth_pulse_samples.append(gp_b[["bootstrap_id", "object", "baseline", "branch", "early_positive_area", "core_positive_area", "late_positive_area", "pre_anchor_positive_area", "post_anchor_positive_area", "growth_center_positive", "secondary_primary_area_ratio"]])
            gpair_b = _pairwise_growth_process_from_pulses(gp_b, scope)
            if not gpair_b.empty:
                gpair_b["bootstrap_id"] = ib
                growth_pair_samples.append(gpair_b[["bootstrap_id", "object_A", "object_B", "baseline", "branch", "metric", "delta_observed"]])
        if cfg.log_every_bootstrap > 0 and (ib + 1) % cfg.log_every_bootstrap == 0:
            _log(f"  bootstrap W45 profile order tests: {ib + 1}/{cfg.bootstrap_n}")

    state_samples_df = pd.concat(state_samples, ignore_index=True) if state_samples else pd.DataFrame()
    growth_sign_samples_df = pd.concat(growth_sign_samples, ignore_index=True) if growth_sign_samples else pd.DataFrame()
    growth_pulse_samples_df = pd.concat(growth_pulse_samples, ignore_index=True) if growth_pulse_samples else pd.DataFrame()
    growth_pair_samples_df = pd.concat(growth_pair_samples, ignore_index=True) if growth_pair_samples else pd.DataFrame()

    state_diff = _add_bootstrap_summaries(state_obs, state_samples_df, ["object_A", "object_B", "baseline", "branch", "range_name"], ["mean_delta_observed", "area_balance_observed"])
    # Rename bootstrap summary columns to explicit names.
    state_diff = state_diff.rename(columns={
        "mean_delta_observed_median": "mean_delta_median", "mean_delta_observed_q025": "mean_delta_q025", "mean_delta_observed_q975": "mean_delta_q975",
        "mean_delta_observed_P_positive": "P_A_ahead_mean", "mean_delta_observed_P_negative": "P_B_ahead_mean", "mean_delta_observed_decision": "mean_delta_decision",
        "area_balance_observed_median": "area_balance_median", "area_balance_observed_q025": "area_balance_q025", "area_balance_observed_q975": "area_balance_q975",
        "area_balance_observed_P_positive": "P_A_ahead_area", "area_balance_observed_P_negative": "P_B_ahead_area", "area_balance_observed_decision": "area_balance_decision",
    })
    if not state_diff.empty:
        state_diff["state_progress_decision"] = state_diff.get("mean_delta_decision", "unresolved")

    growth_sign = _add_bootstrap_summaries(growth_sign_obs, growth_sign_samples_df, ["object", "baseline", "branch"], ["negative_area_observed", "negative_ratio_observed"])
    growth_sign = growth_sign.rename(columns={
        "negative_area_observed_median": "negative_area_median", "negative_area_observed_q025": "negative_area_q025", "negative_area_observed_q975": "negative_area_q975", "negative_area_observed_P_positive": "P_negative_area_gt0", "negative_area_observed_decision": "negative_area_decision",
        "negative_ratio_observed_median": "negative_ratio_median", "negative_ratio_observed_q025": "negative_ratio_q025", "negative_ratio_observed_q975": "negative_ratio_q975",
    })
    if not growth_sign.empty:
        def _ng_dec(r):
            if np.isfinite(r.get("negative_area_q025", np.nan)) and r.get("negative_area_q025") > 0:
                return "negative_growth_supported"
            if np.isfinite(r.get("P_negative_area_gt0", np.nan)) and r.get("P_negative_area_gt0") >= 0.80:
                return "negative_growth_tendency"
            if r.get("negative_area_observed", np.nan) <= EPS:
                return "negative_growth_not_detected"
            return "negative_growth_unresolved"
        growth_sign["negative_growth_decision"] = growth_sign.apply(_ng_dec, axis=1)

    # Pulse structure gets bootstrap summaries for growth mass distribution but keeps pulse labels descriptive.
    pulse = growth_pulse_obs.copy()
    if not growth_pulse_samples_df.empty:
        for col in ["early_positive_area", "core_positive_area", "late_positive_area", "pre_anchor_positive_area", "post_anchor_positive_area", "growth_center_positive", "secondary_primary_area_ratio"]:
            summ = _summarize_boot_metric(growth_pulse_samples_df, ["object", "baseline", "branch"], col, prefix=f"{col}_")
            if not summ.empty:
                pulse = pulse.merge(summ, on=["object", "baseline", "branch"], how="left")

    growth_pair = _add_bootstrap_summaries(growth_pair_obs, growth_pair_samples_df, ["object_A", "object_B", "baseline", "branch", "metric"], ["delta_observed"])
    growth_pair = growth_pair.rename(columns={
        "delta_observed_median": "delta_median", "delta_observed_q025": "delta_q025", "delta_observed_q975": "delta_q975", "delta_observed_P_positive": "P_A_larger_or_earlier", "delta_observed_P_negative": "P_B_larger_or_earlier", "delta_observed_decision": "growth_process_decision",
    })

    prepost_interp = _build_prepost_interpretation(state_diff, catchup_df, growth_sign, pulse, growth_pair, scope)
    final_summary = _build_order_interpretation_summary(peak_df, sync_df, overlap_df, prepost_interp, scope)

    out.update({
        "timing_resolution_audit": timing_df,
        "tau_sync_estimate": tau_df,
        "pairwise_peak_order_test": peak_df,
        "pairwise_synchrony_equivalence_test": sync_df,
        "pairwise_window_overlap_test": overlap_df,
        "pairwise_state_progress_difference": state_diff,
        "pairwise_state_catchup_reversal": catchup_df,
        "object_growth_sign_structure": growth_sign,
        "object_growth_pulse_structure": pulse,
        "pairwise_growth_process_difference": growth_pair,
        "pairwise_prepost_curve_interpretation": prepost_interp,
        "pairwise_order_interpretation_summary": final_summary,
    })
    return out


def _pick_relation_from_state(state_diff: pd.DataFrame, a: str, b: str, branch: str) -> str:
    if state_diff is None or state_diff.empty:
        return "state_progress_unresolved"
    # Prefer C0/C1 system_window/full_analysis, with C2 as sensitivity.
    sub = state_diff[(state_diff["object_A"] == a) & (state_diff["object_B"] == b) & (state_diff["branch"] == branch) & (state_diff["range_name"].isin(["system_window", "full_analysis", "core"]))]
    if sub.empty:
        return "state_progress_unresolved"
    txt = "|".join(sub.get("state_progress_decision", pd.Series(dtype=str)).astype(str).tolist())
    if "A_direction_supported" in txt:
        return "A_state_ahead_supported"
    if "B_direction_supported" in txt:
        return "B_state_ahead_supported"
    if "A_direction_tendency" in txt:
        return "A_state_ahead_tendency"
    if "B_direction_tendency" in txt:
        return "B_state_ahead_tendency"
    return "state_progress_unresolved"


def _pick_relation_from_growth(growth_pair: pd.DataFrame, a: str, b: str, branch: str) -> str:
    if growth_pair is None or growth_pair.empty:
        return "growth_process_unresolved"
    sub = growth_pair[(growth_pair["object_A"] == a) & (growth_pair["object_B"] == b) & (growth_pair["branch"] == branch)]
    if sub.empty:
        return "growth_process_unresolved"
    txt = "|".join(sub.get("growth_process_decision", pd.Series(dtype=str)).astype(str).tolist())
    if "A_direction_supported" in txt:
        return "A_growth_metric_supported"
    if "B_direction_supported" in txt:
        return "B_growth_metric_supported"
    if "A_direction_tendency" in txt:
        return "A_growth_metric_tendency"
    if "B_direction_tendency" in txt:
        return "B_growth_metric_tendency"
    return "growth_process_unresolved"


def _build_prepost_interpretation(state_diff: pd.DataFrame, catchup_df: pd.DataFrame, growth_sign: pd.DataFrame, pulse: pd.DataFrame, growth_pair: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    for a, b in OBJECT_ORDER_PAIRS:
        sd_dist = _pick_relation_from_state(state_diff, a, b, "dist")
        sd_pat = _pick_relation_from_state(state_diff, a, b, "pattern")
        gr_dist = _pick_relation_from_growth(growth_pair, a, b, "dist")
        gr_pat = _pick_relation_from_growth(growth_pair, a, b, "pattern")
        notes = []
        # Object-level notes for negative growth / multi-pulse; kept descriptive.
        for obj in [a, b]:
            ng = growth_sign[growth_sign["object"] == obj] if growth_sign is not None and not growth_sign.empty else pd.DataFrame()
            if not ng.empty and any(ng.get("negative_growth_decision", pd.Series(dtype=str)).astype(str).str.contains("supported|tendency", regex=True)):
                notes.append(f"{obj}:negative_growth_present")
            pu = pulse[pulse["object"] == obj] if pulse is not None and not pulse.empty else pd.DataFrame()
            if not pu.empty and any(pu.get("growth_pulse_structure", pd.Series(dtype=str)).astype(str).str.contains("multi_pulse|early_plus_late", regex=True)):
                notes.append(f"{obj}:multi_pulse_candidate")
        if sd_dist.startswith("A_state") or gr_dist.startswith("A_growth"):
            interp = "A_distance_state_or_growth_ahead"
        elif sd_dist.startswith("B_state") or gr_dist.startswith("B_growth"):
            interp = "B_distance_state_or_growth_ahead"
        elif sd_pat.startswith("A_state") or gr_pat.startswith("A_growth"):
            interp = "A_pattern_state_or_growth_ahead"
        elif sd_pat.startswith("B_state") or gr_pat.startswith("B_growth"):
            interp = "B_pattern_state_or_growth_ahead"
        else:
            interp = "prepost_curve_unresolved"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a,
            "object_B": b,
            "state_dist_relation": sd_dist,
            "state_pattern_relation": sd_pat,
            "growth_dist_relation": gr_dist,
            "growth_pattern_relation": gr_pat,
            "negative_growth_notes": ";".join(notes),
            "multi_pulse_notes": ";".join([n for n in notes if "multi_pulse" in n]),
            "baseline_sensitivity": "not_summarized_in_hotfix06",
            "prepost_curve_interpretation": interp,
            "allowed_statement": "Use as profile pre-post curve evidence only; do not convert directly to hard peak lead.",
            "forbidden_statement": "Do not treat curve-ahead as causal lead or peak-order lead.",
        })
    return pd.DataFrame(rows)


def _build_order_interpretation_summary(peak_df: pd.DataFrame, sync_df: pd.DataFrame, overlap_df: pd.DataFrame, prepost_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    for a, b in OBJECT_ORDER_PAIRS:
        pk = peak_df[(peak_df["object_A"] == a) & (peak_df["object_B"] == b)] if peak_df is not None and not peak_df.empty else pd.DataFrame()
        sy = sync_df[(sync_df["object_A"] == a) & (sync_df["object_B"] == b)] if sync_df is not None and not sync_df.empty else pd.DataFrame()
        ov = overlap_df[(overlap_df["object_A"] == a) & (overlap_df["object_B"] == b)] if overlap_df is not None and not overlap_df.empty else pd.DataFrame()
        pp = prepost_df[(prepost_df["object_A"] == a) & (prepost_df["object_B"] == b)] if prepost_df is not None and not prepost_df.empty else pd.DataFrame()
        peak_dec = pk["peak_order_decision"].iloc[0] if not pk.empty else "peak_order_unavailable"
        sync_dec = sy["synchrony_decision"].iloc[0] if not sy.empty else "synchrony_unavailable"
        overlap_dec = ov["window_overlap_decision"].iloc[0] if not ov.empty else "overlap_unavailable"
        prepost = pp["prepost_curve_interpretation"].iloc[0] if not pp.empty else "prepost_curve_unavailable"
        sd_dist = pp["state_dist_relation"].iloc[0] if not pp.empty else ""
        sd_pat = pp["state_pattern_relation"].iloc[0] if not pp.empty else ""
        gr_dist = pp["growth_dist_relation"].iloc[0] if not pp.empty else ""
        gr_pat = pp["growth_pattern_relation"].iloc[0] if not pp.empty else ""
        if peak_dec == "A_peak_earlier_supported":
            final = "peak_A_earlier_supported"
        elif peak_dec == "B_peak_earlier_supported":
            final = "peak_B_earlier_supported"
        elif sync_dec == "synchrony_supported":
            final = "synchrony_supported"
        elif prepost.startswith("A_"):
            final = "peak_indeterminate_A_state_growth_ahead" if "unresolved" in peak_dec or "tendency" in peak_dec else "A_state_growth_ahead"
        elif prepost.startswith("B_"):
            final = "peak_indeterminate_B_state_growth_ahead" if "unresolved" in peak_dec or "tendency" in peak_dec else "B_state_growth_ahead"
        elif sync_dec == "synchrony_tendency":
            final = "synchrony_tendency"
        elif "overlap" in overlap_dec:
            final = "overlapping_windows_order_indeterminate"
        else:
            final = "timing_indeterminate"
        allowed = "Report peak, synchrony, state-progress and growth-process layers separately."
        forbidden = "Do not infer causality; do not call overlap synchrony unless synchrony_supported."
        rows.append({
            "window_id": scope.window_id,
            "object_A": a,
            "object_B": b,
            "peak_order_decision": peak_dec,
            "synchrony_decision": sync_dec,
            "window_overlap_decision": overlap_dec,
            "prepost_curve_interpretation": prepost,
            "state_dist_relation": sd_dist,
            "state_pattern_relation": sd_pat,
            "growth_dist_relation": gr_dist,
            "growth_pattern_relation": gr_pat,
            "final_combined_interpretation": final,
            "allowed_statement": allowed,
            "forbidden_statement": forbidden,
        })
    return pd.DataFrame(rows)


def _write_w45_profile_order_summary(path: Path, order_outputs: Dict[str, pd.DataFrame]) -> None:
    lines = [
        "# W45 profile order-test summary (hotfix06)", "",
        "This summary is profile-only. It does not use 2D mirror outputs and does not change the detector.", "",
        "## Key output tables", "",
    ]
    for name, df in order_outputs.items():
        lines.append(f"- {name}: {len(df)} rows")
    final = order_outputs.get("pairwise_order_interpretation_summary", pd.DataFrame())
    if final is not None and not final.empty:
        lines += ["", "## Pairwise combined interpretations", ""]
        for _, r in final.iterrows():
            lines.append(f"- {r['object_A']}-{r['object_B']}: {r['final_combined_interpretation']} | peak={r['peak_order_decision']} | sync={r['synchrony_decision']} | prepost={r['prepost_curve_interpretation']}")
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# -----------------------------------------------------------------------------
# Evidence gate and summaries
# -----------------------------------------------------------------------------

def _family_direction(decisions: Sequence[str]) -> str:
    txt = "|".join([str(x) for x in decisions if str(x) != "nan"])
    if "A_direction_supported" in txt or "A_earlier_supported" in txt:
        return "A_supported"
    if "B_direction_supported" in txt or "B_earlier_supported" in txt:
        return "B_supported"
    if "A_direction_tendency" in txt or "A_earlier_tendency" in txt:
        return "A_tendency"
    if "B_direction_tendency" in txt or "B_earlier_tendency" in txt:
        return "B_tendency"
    return "unresolved"


def _build_evidence_and_claims(
    scope: WindowScope,
    selection: pd.DataFrame,
    selected_delta: pd.DataFrame,
    profile_boot: pd.DataFrame,
    twod_boot: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    objs = sorted(selection["object"].dropna().unique())
    fam_rows, class_rows, gate_rows, final_rows, down_rows = [], [], [], [], []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            win_dec = selected_delta[(selected_delta["object_A"] == a) & (selected_delta["object_B"] == b)]
            win_family = _family_direction(win_dec["decision"].tolist()) if not win_dec.empty else "unresolved"
            sa = selection[selection["object"] == a].iloc[0]
            sb = selection[selection["object"] == b].iloc[0]
            ov, ovf = _interval_overlap(int(sa["selected_window_start"]), int(sa["selected_window_end"]), int(sb["selected_window_start"]), int(sb["selected_window_end"])) if np.isfinite(sa["selected_peak_day"]) and np.isfinite(sb["selected_peak_day"]) else (0, 0.0)
            cotrans_veto = bool(ovf >= 0.50 and win_family == "unresolved")
            prof_state_dist = _family_direction(profile_boot[(profile_boot["object_A"] == a) & (profile_boot["object_B"] == b) & (profile_boot["metric_family"] == "state_dist")]["decision"].tolist())
            prof_state_pattern = _family_direction(profile_boot[(profile_boot["object_A"] == a) & (profile_boot["object_B"] == b) & (profile_boot["metric_family"] == "state_pattern")]["decision"].tolist())
            prof_growth_dist = _family_direction(profile_boot[(profile_boot["object_A"] == a) & (profile_boot["object_B"] == b) & (profile_boot["metric_family"] == "growth_dist")]["decision"].tolist())
            prof_growth_pattern = _family_direction(profile_boot[(profile_boot["object_A"] == a) & (profile_boot["object_B"] == b) & (profile_boot["metric_family"] == "growth_pattern")]["decision"].tolist())
            twod_family = _family_direction(twod_boot[(twod_boot["object_A"] == a) & (twod_boot["object_B"] == b)]["decision"].tolist()) if not twod_boot.empty else "not_run"
            fam = {
                "window_id": scope.window_id,
                "object_A": a,
                "object_B": b,
                "raw_profile_window_family": win_family,
                "state_distance_family": prof_state_dist,
                "state_pattern_family": prof_state_pattern,
                "growth_distance_family": prof_growth_dist,
                "growth_pattern_family": prof_growth_pattern,
                "profile_2d_mirror_family": twod_family,
                "selected_window_overlap_fraction": ovf,
                "co_transition_veto": cotrans_veto,
            }
            fam_rows.append(fam)
            families = [win_family, prof_state_dist, prof_state_pattern, prof_growth_dist, prof_growth_pattern]
            A_hard = sum(x == "A_supported" for x in families)
            B_hard = sum(x == "B_supported" for x in families)
            A_any = sum(x.startswith("A_") for x in families)
            B_any = sum(x.startswith("B_") for x in families)
            if cotrans_veto:
                if A_any > B_any:
                    cls = "co_transition_with_A_curve_tendency"
                    gate = True
                    level = "Level2_curve_tendency_only"
                elif B_any > A_any:
                    cls = "co_transition_with_B_curve_tendency"
                    gate = True
                    level = "Level2_curve_tendency_only"
                else:
                    cls = "co_transition"
                    gate = True
                    level = "Level3_window_overlap"
            elif A_hard >= 2 or (win_family == "A_supported" and A_any >= 2):
                cls = "A_layer_specific_front_or_lead_candidate"
                gate = True
                level = "Level4_window_or_curve_supported"
            elif B_hard >= 2 or (win_family == "B_supported" and B_any >= 2):
                cls = "B_layer_specific_front_or_lead_candidate"
                gate = True
                level = "Level4_window_or_curve_supported"
            elif A_any >= 3 and B_any == 0:
                cls = "A_curve_tendency_only"
                gate = False
                level = "Level2_curve_tendency_only"
            elif B_any >= 3 and A_any == 0:
                cls = "B_curve_tendency_only"
                gate = False
                level = "Level2_curve_tendency_only"
            elif A_any and B_any:
                cls = "branch_or_layer_split"
                gate = False
                level = "downgraded_split"
            else:
                cls = "unresolved"
                gate = False
                level = "unresolved"
            class_rows.append({**fam, "final_structure_class": cls, "evidence_level": level})
            gate_rows.append({"window_id": scope.window_id, "object_A": a, "object_B": b, "claim_type": cls, "gate_pass": gate, "evidence_level": level, "required_evidence": "hardened family gate + co-transition veto", "available_evidence": json.dumps(fam, ensure_ascii=False)})
            claim = f"{a}-{b}: {cls}"
            row = {"window_id": scope.window_id, "object_A": a, "object_B": b, "claim": claim, "final_structure_class": cls, "evidence_level": level, "allowed_statement": "Use as profile-based timing structure with 2D mirror context; do not infer causality.", "forbidden_statement": "Do not write as causal/pathway result or full 2D field timing unless 2D mirror supports it."}
            if gate:
                final_rows.append(row)
            else:
                row["why_not_final"] = level
                down_rows.append(row)
    return pd.DataFrame(fam_rows), pd.DataFrame(class_rows), pd.DataFrame(gate_rows), pd.DataFrame(final_rows), pd.DataFrame(down_rows)


def _profile_vs_2d_comparison(profile_single: pd.DataFrame, twod_single: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    if twod_single.empty or profile_single.empty:
        return pd.DataFrame()
    # Normalize profile single rows to long metric format similar to 2D.
    prof_long = []
    for _, r in profile_single.iterrows():
        for seg in ["early", "core", "late"]:
            prof_long.append({"object": r["object"], "baseline_config": r["baseline_config"], "metric": f"{r['branch']}_mean_{seg}", "profile_value": r[f"mean_{seg}"]})
        prof_long.append({"object": r["object"], "baseline_config": r["baseline_config"], "metric": f"{r['branch']}_growth_center", "profile_value": r["growth_center"]})
    prof_df = pd.DataFrame(prof_long)
    twod = twod_single.rename(columns={"observed": "field2d_value"})[["object", "baseline_config", "metric", "field2d_value"]]
    m = prof_df.merge(twod, on=["object", "baseline_config", "metric"], how="outer")
    rows = []
    for _, r in m.iterrows():
        pv, tv = r.get("profile_value", np.nan), r.get("field2d_value", np.nan)
        if not np.isfinite(pv) or not np.isfinite(tv):
            cls = "unresolved"
        elif np.sign(pv - 0.5) == np.sign(tv - 0.5) or abs(pv - tv) <= 0.10:
            cls = "consistent"
        elif abs(pv - tv) <= 0.25:
            cls = "similar_tendency"
        elif (pv - 0.5) * (tv - 0.5) < 0:
            cls = "opposite"
        else:
            cls = "same_side_different_magnitude"
        rows.append({"window_id": scope.window_id, "object": r["object"], "baseline_config": r["baseline_config"], "metric": r["metric"], "profile_value": pv, "field2d_value": tv, "difference_2d_minus_profile": tv - pv if np.isfinite(pv) and np.isfinite(tv) else np.nan, "agreement_class": cls})
    return pd.DataFrame(rows)


def _write_window_summary(path: Path, scope: WindowScope, final_df: pd.DataFrame, downgraded_df: pd.DataFrame) -> None:
    lines = [
        f"# Window {scope.window_id} summary",
        "",
        f"- system_window: day{scope.system_window_start}–{scope.system_window_end}; anchor_day={scope.anchor_day}",
        f"- detector_search_range: day{scope.detector_search_start}–{scope.detector_search_end}",
        f"- analysis_range: day{scope.analysis_start}–{scope.analysis_end}",
        f"- early/core/late: {scope.early_start}-{scope.early_end} / {scope.core_start}-{scope.core_end} / {scope.late_start}-{scope.late_end}",
        "",
        "## Final claims",
    ]
    if final_df.empty:
        lines.append("- None passed the hardened gate.")
    else:
        for _, r in final_df.iterrows():
            lines.append(f"- {r['claim']} ({r['evidence_level']})")
    lines.append("")
    lines.append("## Downgraded / unresolved signals")
    if downgraded_df.empty:
        lines.append("- None.")
    else:
        for _, r in downgraded_df.iterrows():
            lines.append(f"- {r['claim']} ({r['evidence_level']})")
    path.write_text("\n".join(lines), encoding="utf-8")


def _object_role_matrix(scopes: List[WindowScope], final_all: pd.DataFrame, downgraded_all: pd.DataFrame, selection_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in scopes:
        sel_w = selection_all[selection_all["window_id"] == scope.window_id]
        for obj in sorted(sel_w["object"].dropna().unique()):
            s = sel_w[sel_w["object"] == obj].iloc[0]
            if not final_all.empty and {"window_id", "object_A", "object_B"}.issubset(final_all.columns):
                related_final = final_all[(final_all["window_id"] == scope.window_id) & ((final_all["object_A"] == obj) | (final_all["object_B"] == obj))]
            else:
                related_final = pd.DataFrame()
            if not downgraded_all.empty and {"window_id", "object_A", "object_B"}.issubset(downgraded_all.columns):
                related_down = downgraded_all[(downgraded_all["window_id"] == scope.window_id) & ((downgraded_all["object_A"] == obj) | (downgraded_all["object_B"] == obj))]
            else:
                related_down = pd.DataFrame()
            rows.append({
                "window_id": scope.window_id,
                "object": obj,
                "raw_profile_role": s.get("selected_role", "unresolved"),
                "selected_peak_day": s.get("selected_peak_day", np.nan),
                "support_class": s.get("support_class", ""),
                "n_final_pair_claims": int(len(related_final)),
                "n_downgraded_pair_signals": int(len(related_down)),
                "final_object_role": "front" if str(s.get("selected_role", "")).startswith("front") else str(s.get("selected_role", "unresolved")),
                "notes": ";".join(related_final["final_structure_class"].astype(str).tolist()[:5]) if not related_final.empty else "",
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main per-window processing
# -----------------------------------------------------------------------------

def _process_one_window(
    scope: WindowScope,
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    regions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_win: Path,
    cfg: MultiWinConfig,
) -> Dict[str, pd.DataFrame]:
    _ensure_dir(out_win)
    _safe_to_csv(pd.DataFrame([asdict(scope)]), out_win / f"window_scope_{scope.window_id}.csv")
    _log(f"  [1/6] profile detector: {scope.window_id}")
    score_df, cand_df, selection_df, selected_delta_df, boot_peak_days_df = _run_detector_and_bootstrap(profiles, scope, cfg)
    _safe_to_csv(score_df, out_win / f"raw_profile_detector_scores_{scope.window_id}.csv")
    _safe_to_csv(cand_df, out_win / f"object_profile_window_registry_{scope.window_id}.csv")
    _safe_to_csv(selection_df, out_win / f"main_window_selection_{scope.window_id}.csv")
    _safe_to_csv(selected_delta_df, out_win / f"selected_peak_delta_{scope.window_id}.csv")
    _safe_to_csv(boot_peak_days_df, out_win / f"bootstrap_selected_peak_days_{scope.window_id}.csv")

    _log(f"  [2/6] profile pre-post: {scope.window_id}")
    state, growth, branch, profile_single, profile_pair = _profile_curves(profiles, scope, cfg)
    if cfg.save_daily_curves:
        _safe_to_csv(state, out_win / f"profile_state_progress_curves_{scope.window_id}.csv")
        _safe_to_csv(growth, out_win / f"profile_growth_speed_curves_{scope.window_id}.csv")
    _safe_to_csv(branch, out_win / f"profile_pattern_branch_validity_{scope.window_id}.csv")
    _safe_to_csv(profile_single, out_win / f"single_object_metric_summary_profile_{scope.window_id}.csv")
    _safe_to_csv(profile_pair, out_win / f"pairwise_metric_summary_profile_observed_{scope.window_id}.csv")

    _log(f"  [3/6] 2D mirror: {scope.window_id}")
    curves2d, growth2d, single2d, pair2d, _ = _compute_2d_for_window(regions, scope, cfg)
    if cfg.run_2d and cfg.save_daily_curves:
        _safe_to_csv(curves2d, out_win / f"2d_state_progress_curves_{scope.window_id}.csv")
        _safe_to_csv(growth2d, out_win / f"2d_growth_speed_curves_{scope.window_id}.csv")
    _safe_to_csv(single2d, out_win / f"single_object_metric_summary_2d_{scope.window_id}.csv")
    _safe_to_csv(pair2d, out_win / f"pairwise_metric_summary_2d_observed_{scope.window_id}.csv")
    comp = _profile_vs_2d_comparison(profile_single, single2d, scope)
    _safe_to_csv(comp, out_win / f"profile_vs_2d_metric_comparison_{scope.window_id}.csv")

    _log(f"  [4/6] paired bootstrap: {scope.window_id}")
    profile_boot, twod_boot, samples = _bootstrap_pairwise_metrics(profiles, regions if cfg.run_2d else {}, profile_pair, pair2d, scope, cfg)
    _safe_to_csv(profile_boot, out_win / f"pairwise_metric_summary_profile_bootstrap_{scope.window_id}.csv")
    _safe_to_csv(twod_boot, out_win / f"pairwise_metric_summary_2d_bootstrap_{scope.window_id}.csv")
    if samples is not None:
        _safe_to_csv(samples, out_win / f"pairwise_bootstrap_metric_samples_{scope.window_id}.csv")

    order_outputs = _empty_order_outputs()
    if scope.window_id == "W045" and cfg.run_w45_profile_order_tests:
        _log(f"  [5/7] W45 profile order tests: {scope.window_id}")
        order_outputs = _build_w45_profile_order_tests(scope, profiles, selection_df, cand_df, boot_peak_days_df, state, growth, cfg)
        for name, df in order_outputs.items():
            _safe_to_csv(df, out_win / f"{name}_{scope.window_id}.csv")
        _write_w45_profile_order_summary(out_win / f"W45_profile_order_test_summary_hotfix06.md", order_outputs)

    _log(f"  [6/7] evidence gate: {scope.window_id}")
    fam, clas, gate, final, downgraded = _build_evidence_and_claims(scope, selection_df, selected_delta_df, profile_boot, twod_boot)
    _safe_to_csv(fam, out_win / f"evidence_family_summary_{scope.window_id}.csv")
    _safe_to_csv(clas, out_win / f"timing_structure_classification_{scope.window_id}.csv")
    _safe_to_csv(gate, out_win / f"evidence_gate_table_{scope.window_id}.csv")
    _safe_to_csv(final, out_win / f"final_claim_registry_{scope.window_id}.csv")
    _safe_to_csv(downgraded, out_win / f"downgraded_signal_registry_{scope.window_id}.csv")

    _log(f"  [7/7] write summary: {scope.window_id}")
    _write_window_summary(out_win / f"window_summary_{scope.window_id}.md", scope, final, downgraded)
    return {
        "score": score_df, "candidate": cand_df, "selection": selection_df, "selected_delta": selected_delta_df, "bootstrap_peak_days": boot_peak_days_df,
        "profile_state": state, "profile_growth": growth, "profile_single": profile_single, "profile_pair_obs": profile_pair,
        "2d_state": curves2d, "2d_growth": growth2d, "2d_single": single2d, "2d_pair_obs": pair2d,
        "profile_vs_2d": comp, "profile_pair_boot": profile_boot, "2d_pair_boot": twod_boot, "order_outputs": order_outputs,
        "family": fam, "classification": clas, "gate": gate, "final": final, "downgraded": downgraded,
    }


def _write_w45_regression_audit(out_cross: Path, selection_df: pd.DataFrame, cand_df: pd.DataFrame) -> None:
    """Write a lightweight W045 regression audit against the original V7-z W45 reference.

    This is an implementation-regression check, not a scientific proof.  It helps
    detect accidental detector/input substitutions by comparing selected W045
    profile peaks to the previously audited W45 reference structure.
    """
    ref = {
        "P": 45,
        "V": 45,
        "H": 35,
        "Je": 46,
        "Jw": 41,
    }
    rows = []
    if selection_df is None or selection_df.empty:
        selection_df = pd.DataFrame()
    for obj, ref_day in ref.items():
        sel = selection_df[selection_df.get("object", pd.Series(dtype=object)) == obj] if "object" in selection_df.columns else pd.DataFrame()
        selected = np.nan
        selected_role = "missing"
        support = "missing"
        if not sel.empty:
            selected = sel["selected_peak_day"].iloc[0]
            selected_role = sel.get("selected_role", pd.Series([""])).iloc[0]
            support = sel.get("support_class", pd.Series([""])).iloc[0]
        cand_days = []
        if cand_df is not None and not cand_df.empty and "object" in cand_df.columns:
            cc = cand_df[cand_df["object"] == obj]
            cand_days = [int(x) for x in cc.get("peak_day", pd.Series(dtype=float)).dropna().tolist()]
        diff = np.nan if not np.isfinite(selected) else float(selected) - float(ref_day)
        rows.append({
            "window_id": "W045",
            "object": obj,
            "reference_v7z_peak_day": ref_day,
            "selected_peak_day": selected,
            "selected_minus_reference_days": diff,
            "selected_role": selected_role,
            "support_class": support,
            "all_candidate_peak_days": ";".join(map(str, cand_days)),
            "regression_pass_loose": bool(np.isfinite(diff) and abs(diff) <= 5),
            "note": "Loose day tolerance only; inspect candidate/support class before scientific use.",
        })
    _safe_to_csv(pd.DataFrame(rows), out_cross / "W045_regression_against_v7z_reference_v7_z_multiwin_a_hotfix04.csv")


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run_accepted_windows_multi_object_prepost_v7_z_multiwin_a(v7_root: Path | str) -> None:
    v7_root = Path(v7_root)
    cfg = MultiWinConfig.from_env()
    out_root = _ensure_dir(v7_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v7_root / "logs" / OUTPUT_TAG)

    t0 = time.time()
    _log("[1/8] Load configured significant windows")
    wins = _load_accepted_windows(v7_root, out_cross, cfg)
    _log(f"  configured windows available: {len(wins)}")

    _log("[2/8] Build window_scope_registry")
    scopes, validity = _build_window_scopes(wins, cfg)
    scope_df = pd.DataFrame([asdict(s) for s in scopes])
    _safe_to_csv(scope_df, out_cross / "window_scope_registry_v7_z_multiwin_a.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v7_z_multiwin_a.csv")
    run_scopes, run_scope_audit = _filter_scopes_for_run(scopes, cfg)
    _safe_to_csv(run_scope_audit, out_cross / "run_window_selection_audit_v7_z_multiwin_a.csv")
    _safe_to_csv(pd.DataFrame([asdict(s) for s in run_scopes]), out_cross / "run_window_scope_registry_v7_z_multiwin_a.csv")
    _log(f"  windows selected for this run: {len(run_scopes)} / {len(scopes)}")

    _log("[3/8] Load smoothed fields")
    smoothed = Path(cfg.smoothed_fields_path) if cfg.smoothed_fields_path else v7_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"
    if not smoothed.exists():
        # v7_root = D:/easm_project01/stage_partition/V7 => project root is parents[1]
        smoothed = v7_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"
    fields, audit = clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    _safe_to_csv(audit, out_cross / "input_key_audit_v7_z_multiwin_a.csv")
    _write_json({"version": VERSION, "config": asdict(cfg), "smoothed_fields": str(smoothed), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, out_root / "run_meta.json")

    _log("[4/8] Build object profiles and 2D fields")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    arr_fields: Dict[str, np.ndarray] = {}
    object_rows = []
    for spec in clean.OBJECT_SPECS:
        arr = clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        arr_fields[spec.field_role] = arr
        prof, target_lat, weights = clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
        object_rows.append({**asdict(spec), "profile_shape": str(prof.shape), "target_lat_min": float(np.nanmin(target_lat)), "target_lat_max": float(np.nanmax(target_lat))})
    _safe_to_csv(pd.DataFrame(object_rows), out_cross / "object_registry_v7_z_multiwin_a.csv")
    # Build 2D only if requested; convert fields dict to year/day/lat/lon arrays.
    regions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if cfg.run_2d:
        # Need complete fields with converted arrays.
        converted = dict(fields)
        for role in ["precip", "v850", "z500", "u200"]:
            converted[role] = clean._as_year_day_lat_lon(fields[role], lat, lon, years)
        regions = _build_2d_regions(converted, lat, lon)

    _log("[5/8] Process windows")
    all_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in [
        "candidate", "selection", "selected_delta", "profile_single", "profile_pair_boot", "2d_single", "2d_pair_boot", "profile_vs_2d", "family", "classification", "gate", "final", "downgraded"
    ]}
    for iw, scope in enumerate(run_scopes, start=1):
        _log(f"[{iw}/{len(run_scopes)}] Processing {scope.window_id}")
        out_win = _ensure_dir(out_per / scope.window_id)
        res = _process_one_window(scope, profiles, regions, out_win, cfg)
        for k in all_parts:
            if k in res and isinstance(res[k], pd.DataFrame) and not res[k].empty:
                all_parts[k].append(res[k])

    _log("[6/8] Build cross-window summaries")
    concat = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in all_parts.items()}
    _safe_to_csv(concat["candidate"], out_cross / "object_window_summary_all_windows.csv")
    _safe_to_csv(concat["selection"], out_cross / "main_window_selection_all_windows.csv")
    _safe_to_csv(concat["profile_single"], out_cross / "profile_state_growth_summary_all_windows.csv")
    _safe_to_csv(concat["2d_single"], out_cross / "2d_state_growth_summary_all_windows.csv")
    _safe_to_csv(concat["profile_vs_2d"], out_cross / "profile_vs_2d_comparison_all_windows.csv")
    _safe_to_csv(concat["classification"], out_cross / "pairwise_timing_structure_all_windows.csv")
    _safe_to_csv(concat["family"], out_cross / "evidence_family_summary_all_windows.csv")
    _safe_to_csv(concat["final"], out_cross / "final_claim_registry_all_windows.csv")
    _safe_to_csv(concat["downgraded"], out_cross / "downgraded_signal_registry_all_windows.csv")
    role = _object_role_matrix(run_scopes, concat["final"], concat["downgraded"], concat["selection"])
    _safe_to_csv(role, out_cross / "object_role_evolution_matrix.csv")
    if any(s.window_id == "W045" for s in run_scopes):
        _write_w45_regression_audit(out_cross, concat["selection"], concat["candidate"])

    _log("[7/8] Write multiwindow summary")
    lines = ["# V7-z-multiwin-a summary", "", f"Accepted windows processed: {len(run_scopes)}", f"Accepted windows available: {len(scopes)}", f"run_mode: {cfg.window_mode}; targets: {cfg.target_windows}; run_2d: {cfg.run_2d}", "", "## Window scopes processed"]
    for s in run_scopes:
        lines.append(f"- {s.window_id}: system day{s.system_window_start}-{s.system_window_end}, detector day{s.detector_search_start}-{s.detector_search_end}, analysis day{s.analysis_start}-{s.analysis_end}")
    lines.append("")
    lines.append("## Final claims by window")
    if concat["final"].empty:
        lines.append("- No final claims passed hardened gate.")
    else:
        for wid, g in concat["final"].groupby("window_id"):
            lines.append(f"### {wid}")
            for _, r in g.iterrows():
                lines.append(f"- {r['claim']} ({r['evidence_level']})")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("- system_window is the accepted core band, not detector search range.")
    lines.append("- raw/profile object-window detector uses detector_search_range and original V7-z z-scored climatological profile input.")
    lines.append("- profile and 2D pre-post metrics use analysis_range and C0/C1/C2 baselines.")
    lines.append("- 2D mirror does not rediscover windows and does not rewrite final claims by itself.")
    (out_cross / "multiwindow_summary.md").write_text("\n".join(lines), encoding="utf-8")

    _log("[8/8] Done")
    _write_json({
        "version": VERSION,
        "elapsed_seconds": time.time() - t0,
        "n_windows_available": len(scopes),
        "n_windows_processed": len(run_scopes),
        "n_final_claims": int(len(concat["final"])),
        "n_downgraded_signals": int(len(concat["downgraded"])),
        "output_root": str(out_root),
    }, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run_accepted_windows_multi_object_prepost_v7_z_multiwin_a(Path(__file__).resolve().parents[3])
