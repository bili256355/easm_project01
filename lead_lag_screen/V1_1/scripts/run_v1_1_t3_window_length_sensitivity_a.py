from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    v1_1_root = here.parents[1]
    src = v1_1_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V1_1 T3 window-length sensitivity audit: equal-length controls and T3 expansions."
    )
    p.add_argument("--project-root", type=Path, default=Path(r"D:\easm_project01"))
    p.add_argument("--n-surrogates", type=int, default=None)
    p.add_argument("--n-audit-surrogates", type=int, default=None)
    p.add_argument("--n-direction-bootstrap", type=int, default=None)
    p.add_argument("--debug-fast", action="store_true", help="Use smaller resampling counts for quick connection tests.")
    p.add_argument("--no-audit-surrogate-null", action="store_true", help="Disable audit surrogate null for faster sensitivity runs.")
    return p.parse_args()


def main() -> None:
    _add_src_to_path()
    from lead_lag_screen_v1_1.settings import LeadLagScreenSettings
    from lead_lag_screen_v1_1.window_length_sensitivity import run_v1_1_t3_window_length_sensitivity_a

    args = parse_args()
    settings = LeadLagScreenSettings(project_root=args.project_root)

    if args.debug_fast:
        settings = replace(
            settings,
            n_surrogates=100,
            n_audit_surrogates=100,
            n_direction_bootstrap=100,
            surrogate_chunk_size=50,
            bootstrap_chunk_size=50,
        )

    updates = {}
    if args.n_surrogates is not None:
        updates["n_surrogates"] = int(args.n_surrogates)
    if args.n_audit_surrogates is not None:
        updates["n_audit_surrogates"] = int(args.n_audit_surrogates)
    if args.n_direction_bootstrap is not None:
        updates["n_direction_bootstrap"] = int(args.n_direction_bootstrap)
    if args.no_audit_surrogate_null:
        updates["run_audit_surrogate_null"] = False
    if updates:
        settings = replace(settings, **updates)

    summary = run_v1_1_t3_window_length_sensitivity_a(settings)
    print("V1_1 T3 window-length sensitivity finished:", summary["root_output_dir"])


if __name__ == "__main__":
    main()
