from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import StagePartitionV7Settings

FIELDS: tuple[str, ...] = ("P", "V", "H", "Je", "Jw")
OUTPUT_TAG = "progress_order_significance_audit_v7_e1"
SOURCE_V7E_TAG = "field_transition_progress_timing_v7_e"


@dataclass
class ProgressOrderSignificanceSettings:
    """Configuration for V7-e1 significance audit.

    This is intentionally small: V7-e1 does not recompute progress timing,
    and it does not introduce any minimum effective day threshold.
    """

    source_v7e_output_tag: str = SOURCE_V7E_TAG
    output_tag: str = OUTPUT_TAG
    # Exact sign-direction test under a sign-flip null; no Monte-Carlo permutations needed.
    random_seed: int = 20260430
    fdr_alpha: float = 0.05


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv_required(path: Path, required_cols: Iterable[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {name}: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns {missing}; path={path}")
    return df


def _as_int_or_none(x) -> int | None:
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def _sign_of_number(x: float | int | None) -> int:
    if x is None or pd.isna(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _direction_from_delta_sign(sign: int, field_a: str, field_b: str) -> tuple[str, str, str]:
    """Return bootstrap direction and early/late field for delta=B-A.

    delta > 0 means field_a is earlier than field_b.
    delta < 0 means field_b is earlier than field_a.
    """
    if sign > 0:
        return "field_a_before_field_b", field_a, field_b
    if sign < 0:
        return "field_b_before_field_a", field_b, field_a
    return "tie_or_zero_median", "", ""


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * m / (np.arange(m, dtype=float) + 1.0)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    out = np.empty_like(q_ranked)
    out[order] = q_ranked
    q[valid] = out
    return q


def _log_binom_pmf(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)


def _logsumexp(log_values: list[float]) -> float:
    if not log_values:
        return float("-inf")
    m = max(log_values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(x - m) for x in log_values))


def _signflip_p_value(delta: np.ndarray) -> float:
    """Exact two-sided sign-direction test under a sign-flip null.

    Null: signs of non-zero Delta values are exchangeable.
    Statistic: extremeness of the majority sign count.

    This intentionally does not introduce any minimum effective day threshold.
    Magnitudes are still retained in the bootstrap CI columns; this test is only
    for directional asymmetry.
    """
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    signs = np.sign(d).astype(int)
    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = max(n_pos, n_neg)
    # Two-sided binomial tail under p=0.5.
    log_tail = _logsumexp([_log_binom_pmf(n, i) for i in range(k, n + 1)])
    p = min(1.0, 2.0 * math.exp(log_tail))
    return float(p)


def _pivot_midpoint_samples(
    df: pd.DataFrame,
    *,
    sample_col: str,
) -> pd.DataFrame:
    required = ["window_id", sample_col, "field", "midpoint_day"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot pivot midpoint samples; missing columns: {missing}")
    use = df[required].copy()
    use["midpoint_day"] = pd.to_numeric(use["midpoint_day"], errors="coerce")
    # If duplicated rows ever appear, median is a safe no-op for the expected single row.
    pivot = use.pivot_table(
        index=["window_id", sample_col],
        columns="field",
        values="midpoint_day",
        aggfunc="median",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def _anchor_lookup(accepted_df: pd.DataFrame | None, bootstrap_df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    if accepted_df is not None and {"window_id", "anchor_day"}.issubset(accepted_df.columns):
        for _, r in accepted_df[["window_id", "anchor_day"]].dropna().iterrows():
            out[str(r["window_id"])] = int(r["anchor_day"])
    if "anchor_day" in bootstrap_df.columns:
        for _, r in bootstrap_df[["window_id", "anchor_day"]].dropna().drop_duplicates().iterrows():
            out.setdefault(str(r["window_id"]), int(r["anchor_day"]))
    return out


def _delta_stats(delta: np.ndarray) -> dict[str, float | int]:
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {
            "n_valid": 0,
            "median_delta": np.nan,
            "mean_delta": np.nan,
            "q025_delta": np.nan,
            "q975_delta": np.nan,
            "prob_delta_gt_0": np.nan,
            "prob_delta_lt_0": np.nan,
            "prob_delta_eq_0": np.nan,
        }
    return {
        "n_valid": int(d.size),
        "median_delta": float(np.nanmedian(d)),
        "mean_delta": float(np.nanmean(d)),
        "q025_delta": float(np.nanpercentile(d, 2.5)),
        "q975_delta": float(np.nanpercentile(d, 97.5)),
        "prob_delta_gt_0": float(np.mean(d > 0)),
        "prob_delta_lt_0": float(np.mean(d < 0)),
        "prob_delta_eq_0": float(np.mean(d == 0)),
    }


def _make_bootstrap_pairwise_summary(
    bootstrap_df: pd.DataFrame,
    accepted_df: pd.DataFrame | None,
    *,
    random_seed: int,
) -> pd.DataFrame:
    pivot = _pivot_midpoint_samples(bootstrap_df, sample_col="bootstrap_id")
    windows = sorted(pivot["window_id"].dropna().unique().tolist())
    anchors = _anchor_lookup(accepted_df, bootstrap_df)
    rows: list[dict] = []
    # random_seed is retained in run metadata for reproducibility symmetry with other V7 outputs;
    # the exact sign-direction test below is deterministic.

    for window_id in windows:
        sub = pivot[pivot["window_id"] == window_id]
        for field_a, field_b in combinations(FIELDS, 2):
            if field_a not in sub.columns or field_b not in sub.columns:
                continue
            a = pd.to_numeric(sub[field_a], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(sub[field_b], errors="coerce").to_numpy(dtype=float)
            delta = b - a
            finite = np.isfinite(delta)
            delta = delta[finite]
            stats = _delta_stats(delta)
            p = _signflip_p_value(delta)
            median_delta = stats["median_delta"]
            bsign = _sign_of_number(median_delta)
            bootstrap_direction, early, late = _direction_from_delta_sign(bsign, field_a, field_b)
            ci_excludes_zero = bool(
                np.isfinite(stats["q025_delta"])
                and np.isfinite(stats["q975_delta"])
                and ((stats["q025_delta"] > 0) or (stats["q975_delta"] < 0))
            )
            rows.append(
                {
                    "window_id": window_id,
                    "anchor_day": anchors.get(str(window_id), np.nan),
                    "field_a": field_a,
                    "field_b": field_b,
                    "median_delta_b_minus_a": stats["median_delta"],
                    "mean_delta_b_minus_a": stats["mean_delta"],
                    "q025_delta_b_minus_a": stats["q025_delta"],
                    "q975_delta_b_minus_a": stats["q975_delta"],
                    "prob_delta_gt_0": stats["prob_delta_gt_0"],
                    "prob_delta_lt_0": stats["prob_delta_lt_0"],
                    "prob_delta_eq_0": stats["prob_delta_eq_0"],
                    "n_bootstrap_valid": stats["n_valid"],
                    "bootstrap_direction": bootstrap_direction,
                    "bootstrap_ci_excludes_zero": ci_excludes_zero,
                    "field_early_candidate": early,
                    "field_late_candidate": late,
                    "signflip_p_value": p,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["signflip_q_value"] = _bh_fdr(out["signflip_p_value"].to_numpy(dtype=float))
    else:
        out["signflip_q_value"] = []
    return out


def _make_loyo_pairwise_summary(loyo_df: pd.DataFrame) -> pd.DataFrame:
    sample_col = "left_out_year_index" if "left_out_year_index" in loyo_df.columns else "left_out_year"
    pivot = _pivot_midpoint_samples(loyo_df, sample_col=sample_col)
    windows = sorted(pivot["window_id"].dropna().unique().tolist())
    rows: list[dict] = []
    for window_id in windows:
        sub = pivot[pivot["window_id"] == window_id]
        for field_a, field_b in combinations(FIELDS, 2):
            if field_a not in sub.columns or field_b not in sub.columns:
                continue
            a = pd.to_numeric(sub[field_a], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(sub[field_b], errors="coerce").to_numpy(dtype=float)
            delta = b - a
            finite = np.isfinite(delta)
            delta = delta[finite]
            stats = _delta_stats(delta)
            median_delta = stats["median_delta"]
            sign = _sign_of_number(median_delta)
            loyo_direction, _, _ = _direction_from_delta_sign(sign, field_a, field_b)
            # Direction counts are filled later relative to the bootstrap direction.
            rows.append(
                {
                    "window_id": window_id,
                    "field_a": field_a,
                    "field_b": field_b,
                    "loyo_median_delta_b_minus_a": stats["median_delta"],
                    "loyo_prob_delta_gt_0": stats["prob_delta_gt_0"],
                    "loyo_prob_delta_lt_0": stats["prob_delta_lt_0"],
                    "loyo_prob_delta_eq_0": stats["prob_delta_eq_0"],
                    "n_loyo_valid": stats["n_valid"],
                    "loyo_direction": loyo_direction,
                }
            )
    return pd.DataFrame(rows)


def _add_loyo_direction_counts(merged: pd.DataFrame, loyo_df: pd.DataFrame) -> pd.DataFrame:
    sample_col = "left_out_year_index" if "left_out_year_index" in loyo_df.columns else "left_out_year"
    pivot = _pivot_midpoint_samples(loyo_df, sample_col=sample_col)
    count_rows: list[dict] = []
    for _, row in merged[["window_id", "field_a", "field_b", "bootstrap_direction"]].iterrows():
        window_id = row["window_id"]
        field_a = row["field_a"]
        field_b = row["field_b"]
        boot_sign = 1 if row["bootstrap_direction"] == "field_a_before_field_b" else -1 if row["bootstrap_direction"] == "field_b_before_field_a" else 0
        sub = pivot[pivot["window_id"] == window_id]
        if field_a not in sub.columns or field_b not in sub.columns or boot_sign == 0:
            count_rows.append(
                {
                    "window_id": window_id,
                    "field_a": field_a,
                    "field_b": field_b,
                    "loyo_n_same_direction": 0,
                    "loyo_n_opposite_direction": 0,
                    "loyo_n_zero_or_tie": int(len(sub)) if boot_sign == 0 else 0,
                    "loyo_same_direction_fraction": np.nan,
                }
            )
            continue
        delta = pd.to_numeric(sub[field_b], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(sub[field_a], errors="coerce").to_numpy(dtype=float)
        delta = delta[np.isfinite(delta)]
        signs = np.sign(delta).astype(int)
        same = int(np.sum(signs == boot_sign))
        opp = int(np.sum(signs == -boot_sign))
        tie = int(np.sum(signs == 0))
        denom = same + opp + tie
        count_rows.append(
            {
                "window_id": window_id,
                "field_a": field_a,
                "field_b": field_b,
                "loyo_n_same_direction": same,
                "loyo_n_opposite_direction": opp,
                "loyo_n_zero_or_tie": tie,
                "loyo_same_direction_fraction": float(same / denom) if denom else np.nan,
            }
        )
    counts = pd.DataFrame(count_rows)
    return merged.merge(counts, on=["window_id", "field_a", "field_b"], how="left")


def _assign_labels(df: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    out = df.copy()
    significance_labels = []
    loyo_labels = []
    final_labels = []
    cautions = []
    for _, r in out.iterrows():
        q = r.get("signflip_q_value", np.nan)
        sig = bool(np.isfinite(q) and q < alpha)
        significance_labels.append("direction_significant" if sig else "direction_not_significant")

        bdir = r.get("bootstrap_direction", "tie_or_zero_median")
        ldir = r.get("loyo_direction", "tie_or_zero_median")
        conflict = False
        if bdir == "field_a_before_field_b" and ldir == "field_b_before_field_a":
            conflict = True
        if bdir == "field_b_before_field_a" and ldir == "field_a_before_field_b":
            conflict = True
        loyo_labels.append("loyo_conflicting" if conflict else "loyo_consistent_or_tie")

        ci_excl = bool(r.get("bootstrap_ci_excludes_zero", False))
        no_bootstrap_direction = bdir == "tie_or_zero_median"
        if conflict:
            final = "conflicting_evidence"
        elif no_bootstrap_direction:
            final = "not_distinguishable"
        elif sig and ci_excl:
            final = "confirmed_directional_order"
        elif sig and not ci_excl:
            final = "supported_directional_tendency"
        else:
            final = "not_distinguishable"
        final_labels.append(final)

        caution = ["progress_order_not_causality", "no_minimum_effective_day_threshold"]
        if no_bootstrap_direction:
            caution.append("bootstrap_median_delta_is_zero_no_direction_assigned")
        if final == "supported_directional_tendency":
            caution.append("direction_significant_but_bootstrap_ci_crosses_zero")
        if final == "not_distinguishable":
            caution.append("do_not_interpret_as_synchrony_without_equivalence_margin")
        if conflict:
            caution.append("bootstrap_loyo_direction_conflict")
        cautions.append("; ".join(caution))

    out["significance_label"] = significance_labels
    out["loyo_conflict_flag"] = [x == "loyo_conflicting" for x in loyo_labels]
    out["loyo_label"] = loyo_labels
    out["final_evidence_label"] = final_labels
    out["caution"] = cautions
    # Ensure early/late is blank when no interpretable direction.
    mask_no_dir = out["bootstrap_direction"].eq("tie_or_zero_median")
    out.loc[mask_no_dir, ["field_early_candidate", "field_late_candidate"]] = ""
    return out


def _edge_string(row: pd.Series) -> str:
    early = str(row.get("field_early_candidate", ""))
    late = str(row.get("field_late_candidate", ""))
    if early and late:
        return f"{early}<{late}"
    return f"{row.get('field_a','?')}/{row.get('field_b','?')}"


def _window_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for window_id, sub in pair_df.groupby("window_id", sort=True):
        confirmed = sub[sub["final_evidence_label"] == "confirmed_directional_order"]
        supported = sub[sub["final_evidence_label"] == "supported_directional_tendency"]
        notdist = sub[sub["final_evidence_label"] == "not_distinguishable"]
        conflict = sub[sub["final_evidence_label"] == "conflicting_evidence"]
        if len(confirmed) > 0:
            result = "has_confirmed_directional_orders"
        elif len(supported) > 0:
            result = "has_supported_directional_tendencies_only"
        elif len(conflict) > 0:
            result = "conflicting_without_confirmed_or_supported"
        else:
            result = "no_directional_order_supported"
        rows.append(
            {
                "window_id": window_id,
                "anchor_day": sub["anchor_day"].dropna().iloc[0] if "anchor_day" in sub and sub["anchor_day"].notna().any() else np.nan,
                "n_total_pairs": int(len(sub)),
                "n_confirmed_directional_order": int(len(confirmed)),
                "n_supported_directional_tendency": int(len(supported)),
                "n_not_distinguishable": int(len(notdist)),
                "n_conflicting_evidence": int(len(conflict)),
                "confirmed_edges": "; ".join(_edge_string(r) for _, r in confirmed.iterrows()) or "none",
                "supported_edges": "; ".join(_edge_string(r) for _, r in supported.iterrows()) or "none",
                "not_distinguishable_pairs": "; ".join(f"{r.field_a}/{r.field_b}" for _, r in notdist.iterrows()) or "none",
                "conflicting_pairs": "; ".join(f"{r.field_a}/{r.field_b}" for _, r in conflict.iterrows()) or "none",
                "window_test_result": result,
                "caution": "progress_timing_order_only_not_causality; not_distinguishable_is_not_equivalence_or_synchrony",
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_audit_log(path: Path, *, source_dir: Path, output_dir: Path, settings: ProgressOrderSignificanceSettings, pair_df: pd.DataFrame) -> None:
    counts = pair_df["final_evidence_label"].value_counts().to_dict() if not pair_df.empty else {}
    text = f"""# progress_order_significance_audit_v7_e1 audit log

## Scope

This audit reads V7-e progress-timing outputs and tests pairwise progress-midpoint differences.
It does not recompute progress curves, field states, bootstrap samples, or LOYO samples.

## Source

- Source V7-e output directory: `{source_dir}`
- Output directory: `{output_dir}`

## Test object

For each window and field pair A/B:

```text
Delta = midpoint_B - midpoint_A
```

- Delta > 0 means A's progress midpoint is earlier than B's.
- Delta < 0 means B's progress midpoint is earlier than A's.

No minimum effective day threshold is used.

## Statistical components

1. Bootstrap Delta distribution and 95% percentile interval.
2. Exact sign-direction test under a sign-flip null: non-zero Delta signs are treated as exchangeable to test directional asymmetry.
3. BH-FDR over pairwise sign-flip p-values.
4. LOYO direction audit as stability evidence, not as a primary significance test.

## Parameters

```json
{json.dumps(asdict(settings), indent=2)}
```

## Evidence label counts

```json
{json.dumps(counts, indent=2)}
```

## Interpretation limits

- This is timing/progress order, not causality.
- `not_distinguishable` does not prove synchrony; equivalence/synchrony would require a justified equivalence margin, which is intentionally not introduced here.
- `supported_directional_tendency` means direction is significant by sign-flip, but the bootstrap 95% CI still crosses zero.
"""
    path.write_text(text, encoding="utf-8")


def run_progress_order_significance_audit_v7_e1(
    settings: StagePartitionV7Settings | None = None,
    audit_settings: ProgressOrderSignificanceSettings | None = None,
    *,
    source_output_dir: Path | None = None,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> dict:
    settings = settings or StagePartitionV7Settings()
    audit_settings = audit_settings or ProgressOrderSignificanceSettings()

    layer_root = settings.layer_root()
    source_dir = source_output_dir or (layer_root / "outputs" / audit_settings.source_v7e_output_tag)
    out_dir = _ensure_dir(output_dir or (layer_root / "outputs" / audit_settings.output_tag))
    logs = _ensure_dir(log_dir or (layer_root / "logs" / audit_settings.output_tag))

    bootstrap_path = source_dir / "field_transition_progress_bootstrap_samples_v7_e.csv"
    loyo_path = source_dir / "field_transition_progress_loyo_samples_v7_e.csv"
    pair_path = source_dir / "pairwise_progress_order_summary_v7_e.csv"
    accepted_path = source_dir / "accepted_windows_used_v7_e.csv"
    source_meta_path = source_dir / "run_meta.json"

    bootstrap_df = _read_csv_required(
        bootstrap_path,
        ["window_id", "bootstrap_id", "field", "midpoint_day"],
        "V7-e bootstrap samples",
    )
    loyo_df = _read_csv_required(
        loyo_path,
        ["window_id", "field", "midpoint_day"],
        "V7-e LOYO samples",
    )
    # Read for provenance and cross-check; not required for the delta calculations.
    pair_source_df = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    accepted_df = pd.read_csv(accepted_path) if accepted_path.exists() else None

    boot_summary = _make_bootstrap_pairwise_summary(
        bootstrap_df,
        accepted_df,
        random_seed=int(audit_settings.random_seed),
    )
    loyo_summary = _make_loyo_pairwise_summary(loyo_df)
    merged = boot_summary.merge(loyo_summary, on=["window_id", "field_a", "field_b"], how="left")
    merged = _add_loyo_direction_counts(merged, loyo_df)
    merged = _assign_labels(merged, alpha=float(audit_settings.fdr_alpha))

    # Stable column order.
    cols = [
        "window_id", "anchor_day", "field_a", "field_b",
        "median_delta_b_minus_a", "mean_delta_b_minus_a", "q025_delta_b_minus_a", "q975_delta_b_minus_a",
        "prob_delta_gt_0", "prob_delta_lt_0", "prob_delta_eq_0", "n_bootstrap_valid",
        "bootstrap_direction", "bootstrap_ci_excludes_zero",
        "signflip_p_value", "signflip_q_value", "significance_label",
        "loyo_median_delta_b_minus_a", "loyo_prob_delta_gt_0", "loyo_prob_delta_lt_0", "loyo_prob_delta_eq_0",
        "n_loyo_valid", "loyo_direction", "loyo_n_same_direction", "loyo_n_opposite_direction", "loyo_n_zero_or_tie",
        "loyo_same_direction_fraction", "loyo_conflict_flag", "loyo_label",
        "field_early_candidate", "field_late_candidate", "final_evidence_label", "caution",
    ]
    merged = merged[[c for c in cols if c in merged.columns]]
    summary = _window_summary(merged)

    pair_out = out_dir / "pairwise_progress_delta_test_v7_e1.csv"
    summary_out = out_dir / "window_progress_delta_test_summary_v7_e1.csv"
    merged.to_csv(pair_out, index=False)
    summary.to_csv(summary_out, index=False)

    # Preserve a small provenance copy of source pairwise labels when available.
    if not pair_source_df.empty:
        pair_source_df.to_csv(out_dir / "source_pairwise_progress_order_summary_v7_e_copy.csv", index=False)

    source_meta = {}
    if source_meta_path.exists():
        try:
            source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        except Exception:
            source_meta = {"warning": f"Could not parse {source_meta_path}"}

    run_meta = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_tag": audit_settings.output_tag,
        "source_v7e_output_dir": str(source_dir),
        "output_dir": str(out_dir),
        "log_dir": str(logs),
        "input_files": {
            "bootstrap_samples": str(bootstrap_path),
            "loyo_samples": str(loyo_path),
            "pairwise_progress_order_summary": str(pair_path),
            "accepted_windows": str(accepted_path),
            "source_run_meta": str(source_meta_path),
        },
        "settings": asdict(audit_settings),
        "n_pair_tests": int(len(merged)),
        "n_windows": int(merged["window_id"].nunique()) if not merged.empty else 0,
        "evidence_label_counts": merged["final_evidence_label"].value_counts().to_dict() if not merged.empty else {},
        "notes": [
            "V7-e1 does not recompute progress timing; it audits pairwise midpoint deltas from V7-e outputs.",
            "No minimum effective day threshold is used.",
            "No pseudo-window null is used.",
            "LOYO is treated as stability audit, not primary significance test.",
            "not_distinguishable is not a proof of synchrony/equivalence.",
        ],
        "source_v7e_run_meta_status": source_meta.get("status") if isinstance(source_meta, dict) else None,
    }
    _write_json(out_dir / "run_meta.json", run_meta)
    _write_json(logs / "run_meta.json", run_meta)
    _write_json(logs / "config_used.json", asdict(audit_settings))
    _write_audit_log(
        logs / "progress_order_significance_audit_v7_e1.md",
        source_dir=source_dir,
        output_dir=out_dir,
        settings=audit_settings,
        pair_df=merged,
    )

    print(f"[V7-e1] Wrote {pair_out}")
    print(f"[V7-e1] Wrote {summary_out}")
    print(f"[V7-e1] Output directory: {out_dir}")
    return run_meta


__all__ = [
    "ProgressOrderSignificanceSettings",
    "run_progress_order_significance_audit_v7_e1",
]
