from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


FAMILY_BY_PREFIX = {
    "P_": "P",
    "V_": "V",
    "H_": "H",
    "Je_": "Je",
    "Jw_": "Jw",
}


def family_from_variable(name: str) -> str:
    s = str(name)
    for prefix, fam in FAMILY_BY_PREFIX.items():
        if s.startswith(prefix):
            return fam
    return "UNKNOWN"


def read_first_existing(directory: Path, names: Iterable[str], required: bool = True) -> pd.DataFrame | None:
    tried: list[str] = []
    for name in names:
        path = directory / name
        tried.append(str(path))
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig")
    if required:
        raise FileNotFoundError("None of the required input tables was found:\n" + "\n".join(tried))
    return None


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    lower = {str(c).lower(): c for c in out.columns}
    if "source_variable" not in out.columns:
        if "source" in out.columns:
            rename["source"] = "source_variable"
        elif "source_var" in lower:
            rename[lower["source_var"]] = "source_variable"
    if "target_variable" not in out.columns:
        if "target" in out.columns:
            rename["target"] = "target_variable"
        elif "target_var" in lower:
            rename[lower["target_var"]] = "target_variable"
    out = out.rename(columns=rename)
    required = {"window", "source_variable", "target_variable"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required key columns: {sorted(missing)}")
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
