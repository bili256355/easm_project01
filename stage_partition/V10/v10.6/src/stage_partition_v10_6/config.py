from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ClusterDef:
    cluster_id: str
    center_day: int
    day_min: int
    day_max: int
    role_seed: str
    included_in_order_test: bool = True


@dataclass(frozen=True)
class W045PreclusterConfig:
    """Configuration for V10.6_a W045 precluster audit.

    This version intentionally stays at method/derived-structure level:
    it audits E1/E2/M/H_post around W045 using V10.5_e curves and
    V10.5_b/d family diagnostics. It does NOT infer causality, does NOT
    re-detect windows, and does NOT run spatial/yearwise validation.
    """

    project_root: Path
    version: str = "v10.6_a"
    task_name: str = "W045 precluster audit"
    objects: tuple[str, ...] = ("joint_all", "P", "V", "H", "Je", "Jw")
    profile_k_values: tuple[int, ...] = (7, 9, 11)

    v10_5e_root: Path | None = None
    v10_5a_root: Path | None = None
    output_root: Path | None = None

    clusters: tuple[ClusterDef, ...] = field(default_factory=lambda: (
        ClusterDef(
            cluster_id="E1_early_precluster",
            center_day=18,
            day_min=12,
            day_max=23,
            role_seed="early precluster around day16-19; joint/P/V/H expected, Jw absent",
            included_in_order_test=True,
        ),
        ClusterDef(
            cluster_id="E2_second_precluster",
            center_day=33,
            day_min=27,
            day_max=38,
            role_seed="second object-layer precluster around day30-35; P/V/H/Je expected, Jw absent",
            included_in_order_test=True,
        ),
        ClusterDef(
            cluster_id="M_w045_main_cluster",
            center_day=45,
            day_min=40,
            day_max=48,
            role_seed="W045 main cluster around day41-46; joint/P/V/Je/Jw expected, H absence under audit",
            included_in_order_test=True,
        ),
        ClusterDef(
            cluster_id="H_post_reference",
            center_day=57,
            day_min=52,
            day_max=62,
            role_seed="post-W045 H reference candidate around day57; reference only, not W045 main order",
            included_in_order_test=False,
        ),
    ))

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "W045PreclusterConfig":
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            v10_5e_root=root / "stage_partition" / "V10" / "v10.5" / "outputs" / "strength_curve_export_v10_5_e",
            v10_5a_root=root / "stage_partition" / "V10" / "v10.5" / "outputs" / "field_index_validation_v10_5_a",
            output_root=root / "stage_partition" / "V10" / "v10.6" / "outputs" / "w045_precluster_audit_v10_6_a",
        )

    @property
    def tables_dir(self) -> Path:
        assert self.output_root is not None
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        assert self.output_root is not None
        return self.output_root / "figures"

    @property
    def run_meta_dir(self) -> Path:
        assert self.output_root is not None
        return self.output_root / "run_meta"

    def ensure_output_dirs(self) -> None:
        for p in (self.tables_dir, self.figures_dir, self.run_meta_dir):
            p.mkdir(parents=True, exist_ok=True)
