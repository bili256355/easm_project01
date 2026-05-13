from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

PROJECT_NAME = "easm_project01"
LAYER_NAME = "foundation"
VERSION_NAME = "V1"
PACKAGE_NAME = "foundation_v1"
PACKAGE_VERSION = "1.0.0"
RUNTIME_TAG = "baseline_a"

PROJECT_ROOT = Path(r"D:\easm_project01")
DATA_ROOT = Path(r"D:\工作目录\data\my_data")

INPUT_FILES: Dict[str, str] = {
    "precip": "precip.npy",
    "u200": "u200.npy",
    "z500": "z500.npy",
    "v850": "v850.npy",
    "lat": "lat.npy",
    "lon": "lon.npy",
    "years": "years.npy",
}

CORE_FIELDS: Tuple[str, ...] = ("precip", "u200", "z500", "v850")
COORD_FIELDS: Tuple[str, ...] = ("lat", "lon", "years")
REQUIRED_INPUTS: Tuple[str, ...] = CORE_FIELDS + COORD_FIELDS

SMOOTH_WINDOW = 9


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path = PROJECT_ROOT
    data_root: Path = DATA_ROOT
    runtime_tag: str = RUNTIME_TAG
    layer_name: str = LAYER_NAME
    version_name: str = VERSION_NAME

    @property
    def layer_root(self) -> Path:
        return self.project_root / self.layer_name / self.version_name

    @property
    def outputs_root(self) -> Path:
        return self.layer_root / "outputs" / self.runtime_tag

    @property
    def logs_root(self) -> Path:
        return self.layer_root / "logs" / self.runtime_tag / "mainline"

    @property
    def preprocess_root(self) -> Path:
        return self.outputs_root / "preprocess"

    @property
    def indices_root(self) -> Path:
        return self.outputs_root / "indices"


@dataclass(frozen=True)
class PreprocessConfig:
    smooth_window: int = SMOOTH_WINDOW
    require_odd_window: bool = True
    full_window_only: bool = True
    anomaly_definition: str = "smoothed_minus_daily_climatology"


@dataclass(frozen=True)
class IndicesConfig:
    compute_anomaly_after_index_build: bool = True
    source_bundle_name: str = "smoothed_fields.npz"


@dataclass(frozen=True)
class FullConfig:
    paths: RuntimePaths = RuntimePaths()
    preprocess: PreprocessConfig = PreprocessConfig()
    indices: IndicesConfig = IndicesConfig()

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "project_name": PROJECT_NAME,
            "layer_name": LAYER_NAME,
            "version_name": VERSION_NAME,
            "package_name": PACKAGE_NAME,
            "package_version": PACKAGE_VERSION,
            "runtime_tag": self.paths.runtime_tag,
            "paths": {
                "project_root": str(self.paths.project_root),
                "data_root": str(self.paths.data_root),
                "layer_root": str(self.paths.layer_root),
                "outputs_root": str(self.paths.outputs_root),
                "logs_root": str(self.paths.logs_root),
                "preprocess_root": str(self.paths.preprocess_root),
                "indices_root": str(self.paths.indices_root),
            },
            "preprocess": asdict(self.preprocess),
            "indices": asdict(self.indices),
            "input_files": INPUT_FILES,
        }
