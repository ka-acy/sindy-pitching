#!/usr/bin/env python3
"""Render the technical paper's figures as vector PDFs.

Every figure shares one style: the same palette, type sizes, line weights and a legend set
above the axes. Nothing is written inside a plot area -- no arrows, callouts or direct labels.
What the reader should look at is said in the caption instead.

  Figure 0  the pipeline, from broadcast video through pose extraction to sparse
            identification and evaluation against controls.
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
deltaE 9.2, normal-vision 27.6, both above their gates. Every series also has a distinct line
style and marker, so identity survives greyscale printing and colourblind readers alike.

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


def legend_above(ax, ncol: int, handles=None, labels=None) -> None:
    """Every figure keys its series the same way: one row above the axes, outside the data."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=ncol,
              borderaxespad=0.0, columnspacing=1.2, handlelength=2.4)


def fig0_pipeline(out: Path) -> None:
    from matplotlib.patches import FancyBboxPatch

    stages = [
        ("Broadcast video", "MLB-YouTube,\n140 pitching clips"),
        ("Pose extraction", "MediaPipe BlazePose,\nper-frame landmarks"),
        ("Standardization", "15-joint schema,\nhip-centred"),
        ("Preprocessing", "gap fill, Savitzky-\nGolay smoothing, trim"),
        ("Features", "10 kinematic variables,\nhandedness-canonical"),
        ("State", "first-order $x$, or\n$x$ + past increments"),
        ("Sparse model", "SINDy, STLSQ on a\npolynomial library"),
        ("Evaluation", "held-out clips vs\nfrozen, persistence, AR"),
    ]
    fig, ax = plt.subplots(figsize=(FULL_W, 1.55))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w, h, gx = 0.86, 0.78, 0.14
    for i, (title, body) in enumerate(stages):
        row, col = divmod(i, 4)
        x = col * (w + gx) + 0.02
        y = 1.12 - row * 1.06
        modeling = i >= 5
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#eef4fc" if modeling else "#f4f3f0",
                                    ec=BLUE if modeling else MUTED, lw=0.8))
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=7.5,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.36, body, ha="center", va="top", fontsize=6.5, color=INK_2,
                linespacing=1.25)
        if col < 3:
            ax.annotate("", xy=(x + w + gx - 0.01, y + h / 2), xytext=(x + w + 0.01, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.8))
    # Row wrap: down from the last box of row one, left through the gap between the rows, and
    # down into the first box of row two, so the connector never crosses a box.
    xe, xs = 3 * (w + gx) + 0.02 + w / 2, 0.02 + w / 2
    top2, mid = 0.06 + h, (0.06 + h + 1.12) / 2
    ax.plot([xe, xe, xs], [1.12, mid, mid], color=INK_2, lw=0.8, solid_joinstyle="miter")
    ax.annotate("", xy=(xs, top2 + 0.01), xytext=(xs, mid + 0.004),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.8, shrinkA=0, shrinkB=0))
    fig.savefig(out / "fig0_pipeline.pdf")
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    recessive(ax)
    ax.axhline(0.0, color=MUTED, lw=1.0, label="persistence")
    ax.axhline(0.1265, color=INK_2, lw=1.0, ls=":", label="momentum, unfitted")
    ax.axhline(0.3356, color=AQUA, lw=1.2, ls="-.", label="damped momentum, 8 coef.")
    # Dense ridge is a wide pale band so that its overlap with the sparse line reads as overlap.
    ax.plot(depths, dense, color=MUTED, lw=5.0, alpha=0.35, solid_capstyle="round",
            label="dense ridge (240 coef. at depth 3)")
    ax.plot(depths, sparse, color=BLUE, lw=1.8, marker="o", ms=4.5, label="sparse STLSQ (30 terms at depth 3)")
    ax.set_xlabel("history depth (frames of past increments)")
    ax.set_ylabel("skill over persistence")
    ax.set_xticks(depths)
    ax.set_xlim(depths[0] - 0.2, depths[-1] + 0.2)
    ax.set_ylim(-0.03, 0.47)
    h_, l_ = ax.get_legend_handles_labels()
    order = [4, 3, 2, 1, 0]
    legend_above(ax, 2, [h_[i] for i in order], [l_[i] for i in order])
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
        ax.plot(h, plain, color=BLUE, lw=1.8, marker="o", ms=4.5, label="first-order, plain")
        ax.plot(h, weak, color=ORANGE, lw=1.8, ls="--", marker="s", ms=4, label="weak-form")
        ax.plot(h, frozen, color=AQUA, lw=1.8, ls="-.", marker="^", ms=4.5,
                label="frozen control (no dynamics)")
        ax.set_xlabel("rollout horizon (frames)")
        ax.set_xticks(h)
    a1.set_ylabel("core-position reconstruction $R^2$")
    a1.set_title("(a) full range", color=INK_2, loc="left", pad=16)
    a2.set_title(r"(b) zoom, $R^2 \geq -9.5$", color=INK_2, loc="left", pad=16)
    a2.set_ylim(-9.5, -1.0)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, columnspacing=1.6, handlelength=2.4)
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

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    recessive(ax)
    ax.axhline(0.0, color=AQUA, lw=1.4, ls="-.", label="persistence")
    ax.plot(t, true[:n], color=MUTED, lw=1.8, label="observed")
    ax.plot(t, pred[:n], color=BLUE, lw=1.6, ls="--", label="sparse model")
    ax.set_xlabel("frame of held-out clip")
    ax.set_ylabel("shoulder rotation increment")
    ax.set_xlim(0, n - 1)
    h_, l_ = ax.get_legend_handles_labels()
    legend_above(ax, 3, [h_[1], h_[2], h_[0]], [l_[1], l_[2], l_[0]])
    fig.savefig(out / "fig3_increments.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results/figures")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig0_pipeline(out)
    fig1_skill_by_depth(out)
    fig2_horizon(out)
    fig3_increments(out)
    for f in sorted(out.glob("*.pdf")):
        print(f"  wrote {f.relative_to(PROJECT_ROOT)}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
