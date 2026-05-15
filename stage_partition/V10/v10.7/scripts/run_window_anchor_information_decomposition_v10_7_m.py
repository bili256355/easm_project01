from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    # .../stage_partition/V10/v10.7/scripts/<file>.py -> project root
    return here.parents[4]


def main() -> None:
    project_root = _find_project_root()
    src_root = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from stage_partition_v10_7.window_anchor_information_decomposition_pipeline import (  # noqa: E501
        Settings,
        run_window_anchor_information_decomposition_v10_7_m,
    )

    parser = argparse.ArgumentParser(
        description=(
            "V10.7_m window-anchor information decomposition: tests Q2/Q1-Q4/Q3/Q5-min "
            "for the H_E2 -> M_P rainband narrow channel without changing V10.7_l."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--smoothed-fields", type=Path, default=None)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--group-frac", type=float, default=0.30)
    parser.add_argument("--n-random-windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["anomaly", "local_background_removed", "raw"],
        choices=["anomaly", "local_background_removed", "raw"],
    )
    args = parser.parse_args()

    settings = Settings(
        project_root=args.project_root,
        smoothed_fields_path_override=args.smoothed_fields,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        group_frac=args.group_frac,
        n_random_windows=args.n_random_windows,
        random_seed=args.seed,
        progress=args.progress,
        modes=tuple(args.modes),
    )
    run_window_anchor_information_decomposition_v10_7_m(settings)


if __name__ == "__main__":
    main()
