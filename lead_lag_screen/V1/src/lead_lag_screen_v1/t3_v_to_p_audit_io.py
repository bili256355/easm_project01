from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_v1_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize V1 result tables that may use source/target or source_variable/target_variable."""
    out = df.copy()
    rename = {}
    if "source" in out.columns and "source_variable" not in out.columns:
        rename["source"] = "source_variable"
    if "target" in out.columns and "target_variable" not in out.columns:
        rename["target"] = "target_variable"
    out = out.rename(columns=rename)
    return out


def family_from_variable(name: object) -> str:
    text = str(name)
    if text.startswith("Jw_"):
        return "Jw"
    if text.startswith("Je_"):
        return "Je"
    if text.startswith("P_"):
        return "P"
    if text.startswith("V_"):
        return "V"
    if text.startswith("H_"):
        return "H"
    return "unknown"


def add_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_v1_keys(df)
    if "source_family" not in out.columns and "source_variable" in out.columns:
        out["source_family"] = out["source_variable"].map(family_from_variable)
    if "target_family" not in out.columns and "target_variable" in out.columns:
        out["target_family"] = out["target_variable"].map(family_from_variable)
    if "source_family" in out.columns and "target_family" in out.columns:
        out["family_direction"] = out["source_family"].astype(str) + "→" + out["target_family"].astype(str)
    return out
