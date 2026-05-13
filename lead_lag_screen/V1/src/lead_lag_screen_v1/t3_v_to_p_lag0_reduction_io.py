from __future__ import annotations

from pathlib import Path
import numpy as np
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
    out = df.copy()
    rename = {}
    if "source" in out.columns and "source_variable" not in out.columns:
        rename["source"] = "source_variable"
    if "target" in out.columns and "target_variable" not in out.columns:
        rename["target"] = "target_variable"
    return out.rename(columns=rename)


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
    if "source_variable" in out.columns and "source_family" not in out.columns:
        out["source_family"] = out["source_variable"].map(family_from_variable)
    if "target_variable" in out.columns and "target_family" not in out.columns:
        out["target_family"] = out["target_variable"].map(family_from_variable)
    if {"source_family", "target_family"}.issubset(out.columns):
        out["family_direction"] = out["source_family"].astype(str) + "→" + out["target_family"].astype(str)
    return out


def load_smoothed_fields(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing smoothed_fields.npz: {path}\n"
            "This audit needs local foundation smooth5 preprocess fields for composite checks."
        )
    with np.load(path, allow_pickle=False) as data:
        arrays = {str(k): np.asarray(data[k]) for k in data.files}

    # Support the foundation/V1 key contract and a few common aliases.
    key_map = {
        "precip": ["precip_smoothed", "precip", "P", "precipitation"],
        "u200": ["u200_smoothed", "u200"],
        "z500": ["z500_smoothed", "z500"],
        "v850": ["v850_smoothed", "v850"],
    }

    def pick(logical: str) -> np.ndarray:
        for key in key_map[logical]:
            if key in arrays:
                return np.asarray(arrays[key], dtype=float)
        raise KeyError(f"smoothed_fields.npz missing any key for {logical}: {key_map[logical]}")

    for req in ["lat", "lon", "years"]:
        if req not in arrays:
            raise KeyError(f"smoothed_fields.npz missing key: {req}")

    return {
        "precip": pick("precip"),
        "u200": pick("u200"),
        "z500": pick("z500"),
        "v850": pick("v850"),
        "lat": np.asarray(arrays["lat"], dtype=float),
        "lon": np.asarray(arrays["lon"], dtype=float),
        "years": np.asarray(arrays["years"]).astype(int),
    }


def read_index_values(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing index_values_smoothed.csv: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "year" not in df.columns or "day" not in df.columns:
        raise ValueError(f"Index table must include year/day columns: {path}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    return df
