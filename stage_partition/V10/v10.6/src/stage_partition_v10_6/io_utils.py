from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


OBJECT_ALIASES = {
    "joint": "joint_all",
    "jointall": "joint_all",
    "joint_all": "joint_all",
    "all": "joint_all",
    "p": "P",
    "precip": "P",
    "precipitation": "P",
    "v": "V",
    "v850": "V",
    "h": "H",
    "z500": "H",
    "je": "Je",
    "jw": "Jw",
}


def normalize_object_name(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    key = re.sub(r"[^A-Za-z0-9_]+", "", s).lower()
    return OBJECT_ALIASES.get(key, s)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return clean_columns(pd.read_csv(path))


def require_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} not found: {path}")
    return clean_columns(pd.read_csv(path))


def find_first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def find_by_glob(root: Path, patterns: Iterable[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def first_matching_column(df: pd.DataFrame, candidates: Iterable[str], contains: Iterable[str] | None = None) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if contains:
        for c in df.columns:
            cl = c.lower()
            if all(tok.lower() in cl for tok in contains):
                return c
    return None


def infer_day_col(df: pd.DataFrame) -> str:
    col = first_matching_column(df, ["day", "day_index", "t", "time", "time_index", "candidate_day", "peak_day", "center_day"])
    if col:
        return col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        cl = c.lower()
        if "day" in cl or cl in {"t", "time"}:
            return c
    raise ValueError(f"Cannot infer day column from columns: {list(df.columns)}")


def infer_object_col(df: pd.DataFrame) -> str | None:
    return first_matching_column(df, ["object", "scope", "target_object", "field", "variable", "name", "object_name"])


def infer_k_col(df: pd.DataFrame) -> str | None:
    return first_matching_column(df, ["k", "profile_k", "profile_energy_k", "window_k"])


def infer_value_col(df: pd.DataFrame, kind: str) -> str:
    if kind == "main_curve":
        candidates = ["detector_score", "main_score", "score", "value", "strength", "curve_value"]
    elif kind == "profile_curve":
        candidates = ["profile_energy", "profile_energy_score", "energy_score", "score", "value", "strength", "curve_value"]
    elif kind == "marker":
        candidates = ["candidate_score", "peak_score", "detector_score", "score", "value", "strength"]
    else:
        candidates = ["score", "value", "strength"]
    col = first_matching_column(df, candidates)
    if col:
        return col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    day_like = {"day", "day_index", "candidate_day", "peak_day", "center_day", "k", "profile_k"}
    numeric_cols = [c for c in numeric_cols if c.lower() not in day_like]
    if not numeric_cols:
        raise ValueError(f"Cannot infer value column for {kind} from columns: {list(df.columns)}")
    return numeric_cols[-1]


def standardize_curve_long(df: pd.DataFrame, kind: str, objects: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return standardized long curve with object/day/value and optional k.

    Supports both long form:
        object, day, score[, k]
    and wide form:
        day[, k], P, V, H, Je, Jw, joint_all
    """
    df = clean_columns(df)
    objects = list(objects)
    day_col = infer_day_col(df)
    object_col = infer_object_col(df)
    k_col = infer_k_col(df)
    meta: dict[str, Any] = {"input_columns": list(df.columns), "day_col": day_col, "object_col": object_col, "k_col": k_col}

    if object_col is not None:
        value_col = infer_value_col(df, kind)
        keep = [object_col, day_col, value_col]
        if k_col:
            keep.append(k_col)
        out = df[keep].copy()
        rename = {object_col: "object", day_col: "day", value_col: "value"}
        if k_col:
            rename[k_col] = "k"
        out = out.rename(columns=rename)
        out["object"] = out["object"].map(normalize_object_name)
        out["day"] = pd.to_numeric(out["day"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        if "k" in out.columns:
            out["k"] = pd.to_numeric(out["k"], errors="coerce")
        meta["value_col"] = value_col
        meta["format"] = "long"
        return out.dropna(subset=["day", "value"]), meta

    # Wide form: object columns are P/V/H/Je/Jw/joint_all or aliases.
    normalized_map = {c: normalize_object_name(c) for c in df.columns}
    value_cols = [c for c, obj in normalized_map.items() if obj in objects and c != day_col and c != k_col]
    if not value_cols:
        raise ValueError(f"Cannot standardize {kind}: no object column and no object-named wide columns in {list(df.columns)}")
    id_vars = [day_col] + ([k_col] if k_col else [])
    out = df.melt(id_vars=id_vars, value_vars=value_cols, var_name="object", value_name="value")
    out["object"] = out["object"].map(normalize_object_name)
    out = out.rename(columns={day_col: "day"})
    if k_col:
        out = out.rename(columns={k_col: "k"})
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if "k" in out.columns:
        out["k"] = pd.to_numeric(out["k"], errors="coerce")
    meta["value_cols"] = value_cols
    meta["format"] = "wide"
    return out.dropna(subset=["day", "value"]), meta


def standardize_markers(df: pd.DataFrame, objects: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = clean_columns(df)
    objects = list(objects)
    day_col = first_matching_column(df, ["candidate_day", "peak_day", "day", "center_day", "day_index"])
    if day_col is None:
        day_col = infer_day_col(df)
    object_col = infer_object_col(df)
    score_col = infer_value_col(df, "marker")
    meta: dict[str, Any] = {"input_columns": list(df.columns), "day_col": day_col, "object_col": object_col, "score_col": score_col}
    if object_col is None:
        # Try wide marker format; uncommon but supported.
        normalized_map = {c: normalize_object_name(c) for c in df.columns}
        value_cols = [c for c, obj in normalized_map.items() if obj in objects and c != day_col]
        if not value_cols:
            raise ValueError(f"Cannot standardize markers: no object column in {list(df.columns)}")
        out = df.melt(id_vars=[day_col], value_vars=value_cols, var_name="object", value_name="candidate_score")
        out = out.rename(columns={day_col: "candidate_day"})
        out["object"] = out["object"].map(normalize_object_name)
        meta["format"] = "wide"
    else:
        out = df[[object_col, day_col, score_col]].copy()
        out = out.rename(columns={object_col: "object", day_col: "candidate_day", score_col: "candidate_score"})
        out["object"] = out["object"].map(normalize_object_name)
        meta["format"] = "long"
    out["candidate_day"] = pd.to_numeric(out["candidate_day"], errors="coerce")
    out["candidate_score"] = pd.to_numeric(out["candidate_score"], errors="coerce")
    out = out.dropna(subset=["candidate_day"])
    return out, meta


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
