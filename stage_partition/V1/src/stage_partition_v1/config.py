
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class FoundationInputSettings:
    foundation_root: Path = Path(os.getenv("EASM_PROJECT01_FOUNDATION_ROOT", r"D:\easm_project01\foundation\V1"))
    strict_input_check: bool = True

    @property
    def preprocess_root(self) -> Path:
        return self.foundation_root / "outputs" / "baseline_a" / "preprocess"

    @property
    def indices_root(self) -> Path:
        return self.foundation_root / "outputs" / "baseline_a" / "indices"


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path = Path(os.getenv("EASM_PROJECT01_STAGE_PARTITION_ROOT", r"D:\easm_project01\stage_partition\V1"))
    output_tag: str = "baseline_a"

    @property
    def outputs_root(self) -> Path:
        return self.project_root / "outputs" / self.output_tag

    @property
    def logs_root(self) -> Path:
        return self.project_root / "logs" / self.output_tag / "mainline"


@dataclass(frozen=True)
class ProfileSettings:
    lat_step_deg: float = 2.0
    interp_method: str = "linear"
    nan_fraction_warn_threshold: float = 0.20
    profile_specs: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "P": {"lon_range": (105.0, 125.0), "lat_grid": (15.0, 39.0)},
        "V": {"lon_range": (105.0, 125.0), "lat_grid": (10.0, 30.0)},
        "H": {"lon_range": (110.0, 140.0), "lat_grid": (15.0, 35.0)},
        "Je": {"lon_range": (120.0, 150.0), "lat_grid": (25.0, 45.0)},
        "Jw": {"lon_range": (80.0, 110.0), "lat_grid": (25.0, 45.0)},
    })


@dataclass(frozen=True)
class StateVectorSettings:
    standardize: bool = True
    block_equal_weight: bool = True
    standardize_eps: float = 1e-6


@dataclass(frozen=True)
class ScoreSettings:
    half_window: int = 5
    smooth_window: int = 3
    local_scale_eps: float = 1e-6


@dataclass(frozen=True)
class CandidateSettings:
    base_quantile: float = 0.75
    edge_quantile: float = 0.68
    min_width_days: int = 3
    merge_gap_days: int = 2


@dataclass(frozen=True)
class GeneralTestSettings:
    struct_test_reps: int = 500
    continuity_min_score: float = 0.55
    occurrence_min: float = 0.40
    dominance_drop_ratio_threshold: float = 0.35
    yearwise_peak_quantile: float = 0.75


@dataclass(frozen=True)
class OutputSettings:
    save_figures: bool = True
    save_debug_tables: bool = True


@dataclass(frozen=True)
class StagePartitionV1Settings:
    foundation: FoundationInputSettings = FoundationInputSettings()
    paths: RuntimePaths = RuntimePaths()
    profiles: ProfileSettings = ProfileSettings()
    state_vector: StateVectorSettings = StateVectorSettings()
    score: ScoreSettings = ScoreSettings()
    candidate: CandidateSettings = CandidateSettings()
    tests: GeneralTestSettings = GeneralTestSettings()
    output: OutputSettings = OutputSettings()

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "layer_name": "stage_partition",
            "version_name": "V1",
            "foundation_root": str(self.foundation.foundation_root),
            "paths": {
                "project_root": str(self.paths.project_root),
                "outputs_root": str(self.paths.outputs_root),
                "logs_root": str(self.paths.logs_root),
            },
            "profiles": asdict(self.profiles),
            "state_vector": asdict(self.state_vector),
            "score": asdict(self.score),
            "candidate": asdict(self.candidate),
            "tests": asdict(self.tests),
            "output": asdict(self.output),
        }
