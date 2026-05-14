from __future__ import annotations

from pathlib import Path
import argparse
import sys


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "foundation").exists() and (p / "stage_partition").exists():
            return p
    # Expected script location: <root>/stage_partition/V10/v10.7/scripts/...
    return here.parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V10.7_k W45 structure-transition mapping audit."
    )
    parser.add_argument("--n-perm", type=int, default=None, help="Number of shuffled-year permutations.")
    parser.add_argument("--n-boot", type=int, default=None, help="Number of bootstrap resamples.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of worker processes for pairwise permutation/bootstrap tests.")
    parser.add_argument("--progress-every", type=int, default=None, help="Print progress every N completed pairwise tasks.")
    parser.add_argument("--pairwise-scope", type=str, default=None, choices=["all", "h-source", "h-related"], help="Limit pairwise table to all pairs, H as E2 source only, or any H-related pair.")
    parser.add_argument("--pairwise-bootstrap-policy", type=str, default=None, choices=["all", "candidate", "none"], help="Pairwise bootstrap policy. 'candidate' computes CI only for weak/clear candidate pairs.")
    parser.add_argument("--multivariate-policy", type=str, default=None, choices=["full", "fast", "skip"], help="HOTFIX04: run full ridge mapping, a fast reduced version, or skip stage 5 entirely.")
    parser.add_argument("--multivariate-n-perm", type=int, default=None, help="HOTFIX04: permutation count for multivariate ridge / remove-one-source. Defaults to n-perm for full, min(200,n-perm) for fast.")
    parser.add_argument("--object-contribution-policy", type=str, default=None, choices=["full", "fast", "skip"], help="HOTFIX04: run full, fast, or skip remove-one-source contribution stage.")
    args = parser.parse_args()

    project_root = _find_project_root()
    src_dir = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stage_partition_v10_7.w45_structure_transition_pipeline import run_w45_structure_transition_mapping_v10_7_k

    meta = run_w45_structure_transition_mapping_v10_7_k(
        project_root,
        n_permutation=args.n_perm,
        n_bootstrap=args.n_boot,
        n_jobs=args.n_jobs,
        progress_every=args.progress_every,
        pairwise_scope=args.pairwise_scope,
        pairwise_bootstrap_policy=args.pairwise_bootstrap_policy,
        multivariate_policy=args.multivariate_policy,
        multivariate_n_permutation=args.multivariate_n_perm,
        object_contribution_policy=args.object_contribution_policy,
    )
    settings = meta.get("settings", {})
    print("[V10.7_k] completed")
    print(f"[V10.7_k] n_permutation = {settings.get('n_permutation')}")
    print(f"[V10.7_k] n_bootstrap = {settings.get('n_bootstrap')}")
    print(f"[V10.7_k] n_jobs = {settings.get('n_jobs')}")
    print(f"[V10.7_k] progress_every = {settings.get('progress_every')}")
    print(f"[V10.7_k] pairwise_scope = {settings.get('pairwise_scope')}")
    print(f"[V10.7_k] pairwise_bootstrap_policy = {settings.get('pairwise_bootstrap_policy')}")
    print(f"[V10.7_k] multivariate_policy = {settings.get('multivariate_policy')}")
    print(f"[V10.7_k] multivariate_n_permutation = {settings.get('multivariate_n_permutation')}")
    print(f"[V10.7_k] object_contribution_policy = {settings.get('object_contribution_policy')}")
    print(f"[V10.7_k] output_root = {meta.get('output_root')}")
    print(f"[V10.7_k] summary = {meta.get('summary_path')}")


if __name__ == "__main__":
    main()
