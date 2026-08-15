#!/usr/bin/env python3
"""Render the technical paper's figures as vector PDFs.

The paper carries twenty tables and no figures, which is the wrong balance for its central
argument: three of its claims are about the SHAPE of a curve, and a shape is what a table
communicates worst.

  Figure 1  skill over persistence against history depth -- the paper's main result, and an
            emphasis form: the sparse model is the point, dense ridge and the two unfitted
            controls are context, so one hue carries the subject and grey carries the rest.
  Figure 2  core-position R2 against rollout horizon for the two fitted models and the frozen
            control. Two panels: the full range, and a zoom, because the values span -75 to -2
            and a single linear axis hides the comparison that matters.
  Figure 3  observed against predicted one-step increments for one held-out clip. State-space
            plots are useless here -- persistence alone scores 0.97 on the state, so every
            curve overlays -- and the increment is what the model actually predicts.

Colours are the validated categorical slots 1-3 (blue, orange, aqua): worst adjacent CVD
deltaE 9.2, normal-vision 27.6, both above their gates. The aqua slot carries a contrast
warning against a light surface, which is relieved here by direct labels on every series and
by the tables that carry the same numbers. Every series also has a distinct line style, so
identity survives greyscale printing and colourblind readers alike.

Usage:
    python src/analysis/make_paper_figures.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.evaluate_baseball_sindy import CORE_POSITIONS

# Validated categorical slots 1-3 plus the ink and de-emphasis tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd"

COL_W, FULL_W = 3.4, 7.0   # IEEE single-column and full-width, inches


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def recessive(ax) -> None:
    """Grid and axes stay behind the data, not compete with it."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig1_skill_by_depth(out: Path) -> None:
    rows = list(csv.DictReader((PROJECT_ROOT / "results/analysis/delay_sindy/skill_by_depth.csv").open()))
    depths = sorted({int(r["depth"]) for r in rows if int(r["depth"]) > 0})
    sparse, dense = [], []
    for d in depths:
        at = [r for r in rows if int(r["depth"]) == d]
        st = [float(r["skill_uniform"]) for r in at if r["estimator"].startswith("STLSQ thr=0.05")]
        rg = [float(r["skill_uniform"]) for r in at if r["estimator"].startswith("RidgeCV")]
        sparse.append(st[0] if st else np.nan)
        dense.append(rg[0] if rg else np.nan)

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    recessive(ax)

    # Reference levels, labelled at the left so they never compete with the legend.
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.axhline(0.1311, color=MUTED, lw=1.0, ls=":")
    ax.text(depths[0] - 0.05, 0.012, "persistence", color=INK_2, fontsize=6.5, va="bottom")
    ax.text(depths[0] - 0.05, 0.143, "momentum control", color=INK_2, fontsize=6.5, va="bottom")

    # The two estimators coincide, which is the finding rather than a plotting accident.
    # Dense ridge is drawn as a wide pale band so the overlap is legible as overlap.
    ax.plot(depths, dense, color=MUTED, lw=5.0, alpha=0.35, solid_capstyle="round",
            label="dense ridge, 240 coefficients")
    ax.plot(depths, sparse, color=BLUE, lw=1.8, marker="o", ms=4.5,
            label="sparse STLSQ, 32 terms")

    ax.annotate("sparse fit sits inside\nthe dense-ridge band",
                xy=(4, sparse[3]), xytext=(3.0, 0.245), color=INK_2, fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.annotate("one frame of history:\n+0.01 to +0.37", xy=(2, sparse[1]),
                xytext=(1.25, 0.30), color=BLUE, fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))

    ax.set_xlabel("history depth (frames of past increments)")
    ax.set_ylabel("skill over persistence")
    ax.set_xticks(depths)
    ax.set_xlim(depths[0] - 0.35, depths[-1] + 0.2)
    ax.set_ylim(-0.03, 0.47)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    fig.savefig(out / "fig1_skill_by_depth.pdf")
    plt.close(fig)


def fig2_horizon(out: Path) -> None:
    h = [5, 10, 20, 40]
    plain = [-75.47, -47.81, -37.99, -29.56]
    weak = [-2.68, -5.40, -4.90, -7.89]
    frozen = [-2.16, -2.83, -4.23, -5.57]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for ax in (a1, a2):
        recessive(ax)
        ax.plot(h, plain, color=BLUE, lw=2.0, marker="o", ms=4.5, label="first-order, plain")
        ax.plot(h, weak, color=ORANGE, lw=2.0, ls="--", marker="s", ms=4, label="weak-form")
        ax.plot(h, frozen, color=AQUA, lw=2.0, ls="-.", marker="^", ms=4.5,
                label="frozen control (no dynamics)")
        ax.set_xlabel("rollout horizon (frames)")
        ax.set_xticks(h)
    a1.set_ylabel("core-position reconstruction $R^2$")
    a1.set_title("full range", color=INK_2, loc="left")
    a2.set_title("zoom: the comparison that matters", color=INK_2, loc="left")
    a2.set_ylim(-9.5, -1.0)
    a2.annotate("frozen control is best\nat every horizon", xy=(20, -4.23),
                xytext=(9, -8.9), color=INK_2, fontsize=7,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    # Direct labels in the zoom, so identity does not rest on the other panel's legend.
    a2.text(40.4, -5.57, "frozen", color=AQUA, fontsize=7, va="center")
    a2.text(40.4, -7.89, "weak-form", color=ORANGE, fontsize=7, va="center")
    a2.set_xlim(4, 52)
    a1.legend(loc="lower right")
    fig.savefig(out / "fig2_horizon.pdf")
    plt.close(fig)


def fig3_increments(out: Path) -> None:
    """Observed vs predicted increments for one held-out clip.

    Plotted on the increment rather than the state: persistence alone scores 0.97 on the
    state, so state-space curves overlay and show nothing. The increment is the quantity the
    model predicts and the quantity the skill score is computed on.
    """
    from sklearn.preprocessing import StandardScaler
    import pysindy as ps

    def load(split):
        root = PROJECT_ROOT / "data/baseball/processed" / split / "features"
        out_ = []
        with (root / "manifest.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["task_type"] != "pitching":
                    continue
                o = np.load(PROJECT_ROOT / r["filepath"])
                out_.append({"X": np.asarray(o["X"], float), "names": o["feature_names"].tolist()})
        return out_

    tr, te = load("train"), load("test")
    names = tr[0]["names"]
    drop = names.index("hip_shoulder_separation_angle")
    kept = [n for n in names if n != "hip_shoulder_separation_angle"]
    tgt = [names.index(n) for n in CORE_POSITIONS if n in names]
    show = kept.index("shoulder_rotation_angle")
    show_t = [names[i] for i in tgt].index("shoulder_rotation_angle")

    def design(cl, depth=3):
        D, Y = [], []
        for c in cl:
            X = c["X"]
            for t in range(depth - 1, X.shape[0] - 1):
                f = [X[t]] + [X[t - k] - X[t - k - 1] for k in range(depth - 1)]
                D.append(np.concatenate([np.delete(x, drop) for x in f]))
                Y.append(X[t + 1, tgt] - X[t, tgt])
        return np.asarray(D), np.asarray(Y)

    Dtr, Ytr = design(tr)
    sc = StandardScaler().fit(Dtr)
    Ztr = sc.transform(Dtr)
    ymu, ysd = Ytr.mean(axis=0), Ytr.std(axis=0)
    ysd[ysd < 1e-12] = 1.0
    opt = ps.STLSQ(threshold=0.05, alpha=0.001, max_iter=20)
    opt.fit(Ztr, (Ytr[:, show_t] - ymu[show_t]) / ysd[show_t])
    coef = np.asarray(opt.coef_).ravel()

    clip = max(te, key=lambda c: c["X"].shape[0])
    Dte, Yte = design([clip])
    pred = (sc.transform(Dte) @ coef) * ysd[show_t] + ymu[show_t]
    true = Yte[:, show_t]
    n = min(90, len(true))
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    recessive(ax)
    ax.axhline(0.0, color=AQUA, lw=1.6, ls="-.")
    ax.plot(t, true[:n], color=MUTED, lw=1.8, label="observed")
    ax.plot(t, pred[:n], color=BLUE, lw=1.6, ls="--", label="sparse model")
    # All three labels sit in the clear band below the traces; placing any of them near the
    # zero line puts them underneath the curves, which cross it repeatedly.
    lab_y = min(true[:n]) * 1.28
    ax.annotate("persistence", xy=(12, 0.0), xytext=(1, lab_y), color=AQUA, fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=0.7))
    ax.annotate("observed", xy=(39, true[39]), xytext=(31, lab_y), color=INK_2, fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.annotate("sparse model", xy=(49, pred[49]), xytext=(58, lab_y), color=BLUE, fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))
    ax.set_xlabel("frame of held-out clip")
    ax.set_ylabel("shoulder rotation increment")
    ax.set_ylim(min(true[:n]) * 1.45, max(true[:n]) * 1.12)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    fig.savefig(out / "fig3_increments.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results/figures")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig1_skill_by_depth(out)
    fig2_horizon(out)
    fig3_increments(out)
    for f in sorted(out.glob("*.pdf")):
        print(f"  wrote {f.relative_to(PROJECT_ROOT)}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
