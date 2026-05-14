from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import W045PreclusterConfig


def _normalize_by_object(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    out = df.copy()
    mx = out.groupby("object")[value_col].transform("max")
    out["norm_value"] = np.where(mx > 0, out[value_col] / mx, np.nan)
    return out


def _add_cluster_spans(ax, cfg: W045PreclusterConfig) -> None:
    for c in cfg.clusters:
        ax.axvspan(c.day_min, c.day_max, alpha=0.08)
        ax.text(c.center_day, 1.03, c.cluster_id.split("_")[0], ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())


def plot_main_zoom(cfg: W045PreclusterConfig, main_curve: pd.DataFrame, markers: pd.DataFrame) -> None:
    df = _normalize_by_object(main_curve[(main_curve["day"] >= 10) & (main_curve["day"] <= 65)])
    fig, ax = plt.subplots(figsize=(12, 6))
    for obj in cfg.objects:
        part = df[df["object"] == obj].sort_values("day")
        if not part.empty:
            ax.plot(part["day"], part["norm_value"], label=obj)
    _add_cluster_spans(ax, cfg)
    m = markers[(markers["candidate_day"] >= 10) & (markers["candidate_day"] <= 65)]
    for obj in cfg.objects:
        mm = m[m["object"] == obj]
        if not mm.empty:
            y = 1.02 - 0.035 * list(cfg.objects).index(obj)
            ax.scatter(mm["candidate_day"], [y] * len(mm), s=20, marker="x")
    ax.set_title("V10.6_a W045 zoom: normalized main-method curves and candidate markers")
    ax.set_xlabel("day index from Apr 1")
    ax.set_ylabel("normalized detector score by object")
    ax.set_xlim(10, 65)
    ax.set_ylim(-0.02, 1.12)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "w045_zoom_main_method_curves_v10_6_a.png", dpi=180)
    plt.close(fig)


def plot_profile_k9_zoom(cfg: W045PreclusterConfig, profile_curve: pd.DataFrame) -> None:
    if profile_curve.empty or "k" not in profile_curve.columns:
        return
    df = profile_curve[(profile_curve["day"] >= 10) & (profile_curve["day"] <= 65) & (profile_curve["k"] == 9)].copy()
    df = _normalize_by_object(df)
    fig, ax = plt.subplots(figsize=(12, 6))
    for obj in cfg.objects:
        part = df[df["object"] == obj].sort_values("day")
        if not part.empty:
            ax.plot(part["day"], part["norm_value"], label=obj)
    _add_cluster_spans(ax, cfg)
    ax.set_title("V10.6_a W045 zoom: normalized profile-energy curves (k=9)")
    ax.set_xlabel("day index from Apr 1")
    ax.set_ylabel("normalized profile-energy by object")
    ax.set_xlim(10, 65)
    ax.set_ylim(-0.02, 1.12)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "w045_zoom_profile_energy_k9_curves_v10_6_a.png", dpi=180)
    plt.close(fig)


def _tier_code(cls: str) -> str:
    if cls == "candidate_inside_cluster":
        return "M"
    if cls == "curve_peak_without_marker":
        return "C"
    if cls == "weak_curve_signal":
        return "w"
    if cls == "no_signal":
        return "."
    if cls == "missing_input":
        return "?"
    return ""


def plot_participation_heatmap(cfg: W045PreclusterConfig, metrics: pd.DataFrame) -> None:
    clusters = [c.cluster_id for c in cfg.clusters]
    data = []
    labels = []
    for obj in cfg.objects:
        row = []
        label_row = []
        for cid in clusters:
            part = metrics[(metrics["cluster_id"] == cid) & (metrics["object"] == obj)]
            row.append(float(part["relative_main_strength_to_object_fullseason_max"].iloc[0]) if not part.empty else np.nan)
            cls = part["participation_class"].iloc[0] if not part.empty and "participation_class" in part.columns else "missing_input"
            label_row.append(_tier_code(str(cls)))
        data.append(row)
        labels.append(label_row)
    arr = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    im = ax.imshow(arr, aspect="auto")
    ax.set_yticks(range(len(cfg.objects)))
    ax.set_yticklabels(cfg.objects)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([c.split("_")[0] for c in clusters], rotation=0)
    for i in range(len(cfg.objects)):
        for j in range(len(clusters)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=10)
    ax.set_title("V10.6_a W045 cluster participation heatmap (M=marker, C=curve-only, w=weak, .=none)")
    fig.colorbar(im, ax=ax, label="relative main strength")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "w045_cluster_participation_heatmap_v10_6_a.png", dpi=180)
    plt.close(fig)


def plot_similarity_heatmap(cfg: W045PreclusterConfig, similarity: pd.DataFrame) -> None:
    clusters = [c.cluster_id for c in cfg.clusters if c.included_in_order_test]
    mat = np.eye(len(clusters))
    for _, r in similarity.iterrows():
        i = clusters.index(r["cluster_a"])
        j = clusters.index(r["cluster_b"])
        mat[i, j] = mat[j, i] = r["cosine_similarity"]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([c.split("_")[0] for c in clusters])
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([c.split("_")[0] for c in clusters])
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("V10.6_a E1/E2/M participation-vector similarity")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "w045_e1_e2_m_similarity_heatmap_v10_6_a.png", dpi=180)
    plt.close(fig)


def plot_h_role_zoom(cfg: W045PreclusterConfig, main_curve: pd.DataFrame, profile_curve: pd.DataFrame, markers: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    h = main_curve[(main_curve["object"] == "H") & (main_curve["day"] >= 10) & (main_curve["day"] <= 65)].copy()
    if not h.empty:
        mx = h["value"].max()
        ax.plot(h["day"], h["value"] / mx if mx else h["value"], label="H main detector")
    if not profile_curve.empty and "k" in profile_curve.columns:
        for k in cfg.profile_k_values:
            p = profile_curve[(profile_curve["object"] == "H") & (profile_curve["k"] == k) & (profile_curve["day"] >= 10) & (profile_curve["day"] <= 65)].copy()
            if not p.empty:
                mx = p["value"].max()
                ax.plot(p["day"], p["value"] / mx if mx else p["value"], label=f"H profile energy k={k}")
    _add_cluster_spans(ax, cfg)
    hm = markers[(markers["object"] == "H") & (markers["candidate_day"] >= 10) & (markers["candidate_day"] <= 65)]
    if not hm.empty:
        ax.scatter(hm["candidate_day"], [1.04] * len(hm), marker="x", s=35, label="H candidate markers")
        for _, r in hm.iterrows():
            ax.text(r["candidate_day"], 1.07, str(int(r["candidate_day"])), ha="center", va="bottom", fontsize=8)
    for d, txt in [(19, "energy-dominant family?"), (35, "lineage-assigned family"), (45, "no H marker expected"), (57, "post-H reference")]:
        ax.axvline(d, linestyle="--", linewidth=0.8)
        ax.text(d, -0.03, txt, rotation=90, va="bottom", ha="right", fontsize=7)
    ax.set_title("V10.6_a H role zoom around W045")
    ax.set_xlabel("day index from Apr 1")
    ax.set_ylabel("normalized H score")
    ax.set_xlim(10, 65)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "w045_H_role_curve_zoom_v10_6_a.png", dpi=180)
    plt.close(fig)


def make_all_figures(cfg: W045PreclusterConfig, main_curve: pd.DataFrame, profile_curve: pd.DataFrame, markers: pd.DataFrame, metrics: pd.DataFrame, similarity: pd.DataFrame) -> None:
    plot_main_zoom(cfg, main_curve, markers)
    plot_profile_k9_zoom(cfg, profile_curve)
    plot_participation_heatmap(cfg, metrics)
    plot_similarity_heatmap(cfg, similarity)
    plot_h_role_zoom(cfg, main_curve, profile_curve, markers)
