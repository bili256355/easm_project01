from __future__ import annotations

from pathlib import Path
import argparse
import sys


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "foundation").exists() and (p / "stage_partition").exists():
            return p
    return here.parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(description="V10.7_l H_E2 structure -> M_P spatial verification")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutation count for high-low composite tests")
    parser.add_argument("--n-boot", type=int, default=500, help="Bootstrap count for high-low CI")
    parser.add_argument("--group-frac", type=float, default=0.30, help="Top/bottom fraction for H metric grouping")
    parser.add_argument("--progress", action="store_true", help="Print stage progress")
    parser.add_argument("--smoothed-fields", type=str, default=None, help="Optional path to smoothed_fields.npz")
    args = parser.parse_args()

    project_root = _find_project_root()
    src_dir = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from stage_partition_v10_7.h_to_p_spatial_pipeline import run_h_e2_to_m_p_spatial_verification_v10_7_l

    meta = run_h_e2_to_m_p_spatial_verification_v10_7_l(
        project_root=project_root,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        group_frac=args.group_frac,
        progress=args.progress,
        smoothed_fields_path=Path(args.smoothed_fields) if args.smoothed_fields else None,
    )
    print("[V10.7_l] completed")
    print(f"[V10.7_l] output_root = {meta.get('output_root')}")
    print(f"[V10.7_l] summary = {meta.get('summary_path')}")


if __name__ == "__main__":
    main()
