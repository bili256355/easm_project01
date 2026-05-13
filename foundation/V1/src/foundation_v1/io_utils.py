from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .settings import INPUT_FILES


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_file_manifest(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256": _sha256_of_file(path),
    }


def load_input_arrays(data_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Dict[str, Any]] = {}
    missing = []
    for name, filename in INPUT_FILES.items():
        path = data_root / filename
        if not path.exists():
            missing.append(str(path))
            continue
        arrays[name] = np.load(path, allow_pickle=False)
        manifest[name] = build_file_manifest(path)
    if missing:
        raise FileNotFoundError("以下输入文件不存在：\n" + "\n".join(missing))
    return arrays, manifest


def load_npz_bundle(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 npz 文件：{path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {str(k): np.asarray(data[k]) for k in data.files}
    manifest = build_file_manifest(path)
    manifest["npz_keys"] = sorted(arrays.keys())
    return arrays, manifest


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_npz(path: Path, arrays: Dict[str, np.ndarray], compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
