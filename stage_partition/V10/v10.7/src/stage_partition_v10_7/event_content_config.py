from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    target_day: int
    pre_min: int
    pre_max: int
    post_min: int
    post_max: int
    role_seed: str


@dataclass
class EventContentConfig:
    version: str = "v10.7_c"
    output_tag: str = "h_w045_event_content_audit_v10_7_c"
    event_windows: tuple[EventWindow, ...] = (
        EventWindow("H18", 18, 14, 17, 19, 22, "early H adjustment candidate"),
        EventWindow("H35", 35, 31, 34, 36, 39, "second pre-window H local structure"),
        EventWindow("H45", 45, 40, 43, 45, 48, "W045 main-cluster H control"),
        EventWindow("H57", 57, 52, 55, 57, 60, "post-W045 H reference"),
    )
    # Full-domain context used as a background/reference map.
    lon_range: tuple[float, float] = (20.0, 150.0)
    lat_range: tuple[float, float] = (10.0, 60.0)
    # H-object domain used for primary H-content metrics/classification.
    # This matches the H profile/object definition inherited from V10.7_a.
    object_lon_range: tuple[float, float] = (110.0, 140.0)
    object_lat_range: tuple[float, float] = (15.0, 35.0)
    # H object definition is imported from V10.7_a config when reconstructing profile.
    possible_h_field_keys: tuple[str, ...] = (
        "z500_smoothed", "z500", "H", "h", "hgt500", "geopotential_500", "Z500", "zg500"
    )
    possible_lat_keys: tuple[str, ...] = ("lat", "latitude", "lats")
    possible_lon_keys: tuple[str, ...] = ("lon", "longitude", "lons")
    possible_year_keys: tuple[str, ...] = ("years", "year", "time_years")
    representative_sigmas: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 9.0, 14.0)
    feature_contribution_sigmas: dict[str, tuple[float, ...]] = field(default_factory=lambda: {
        "H18": (3.0, 5.0, 9.0, 14.0),
        "H35": (2.0, 3.0, 4.0),
        "H45": (3.0, 5.0, 9.0),
        "H57": (3.0, 5.0, 9.0),
    })

    def output_root(self, project_root: Path) -> Path:
        return project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / self.output_tag

    def to_dict(self) -> dict[str, Any]:
        def convert(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [convert(v) for v in x]
            if isinstance(x, list):
                return [convert(v) for v in x]
            if isinstance(x, dict):
                return {str(k): convert(v) for k, v in x.items()}
            if hasattr(x, "__dataclass_fields__"):
                return {str(k): convert(v) for k, v in asdict(x).items()}
            return x
        return convert(asdict(self))
