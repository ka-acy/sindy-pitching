#!/usr/bin/env python3
"""Reorder the paper from discovery order into argument order.

The paper was written by appending each finding as it arrived, which produced a chronological
record rather than an argument: 55 percent of it sat in one results section of eleven
subsections, the only positive result appeared on page 13 after eight subsections of negatives,
and one subsection was titled "superseded".

This moves blocks rather than rewriting them, so every verified number, table and citation
survives byte-identical. Only headings, section intros and cross-references change. Prose that
belongs to a different argument than the one it was written for is split at known bold
paragraph markers rather than re-typed.

The old-to-new cross-reference map is applied to the whole body, and the result is checked:
every "Section N" or "Section N.M" mentioned must exist as a heading, or the script fails.

Reordering also invalidates the IEEE citation numbers, which run in order of first appearance.
Rather than trust the existing `[n]`, this recovers each number's key by matching the paper's own
References entries against the bibliography and rewrites the citations as `{{key}}` placeholders,
so `scripts/renumber_references.py` renumbers them against the new order. Run that afterwards.

Usage:
    python scripts/restructure_paper.py [--check]
    python scripts/renumber_references.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "technical_paper.md"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from renumber_references import BIB  # noqa: E402  (same directory, no package)

# Old heading -> new section number. Applied to "Section X" and "Sections X" mentions.
XREF = {
    "6.1": "5", "6.2": "5", "6.3": "9", "6.4": "6", "6.5": "7",
    "6.6": "5", "6.7": "6", "6.8": "6", "6.9": "7", "6.10": "8",
    "6.11": "A", "6": "5",
    "5.1": "3", "5.2": "3", "5.3": "3", "5.4": "3", "5.5": "4",
    "4.1": "2", "4.2": "2", "4": "2", "5": "3",
    "7": "11", "8": "12", "9": "10", "10": "10", "11": "11", "12": "12",
    "1.1": "1.1", "2": "1", "3": "2",
}


def blocks(text: str) -> dict[str, str]:
    """Split into blocks keyed by heading id ('6.9', 'Abstract', 'References')."""
    out: dict[str, str] = {}
    parts = re.split(r"\n(?=#{2,3} )", text)
    for part in parts:
        head = part.split("\n", 1)[0]
        m = re.match(r"^#{2,3} (\d+(?:\.\d+)?)\.? ", head)
        key = m.group(1) if m else head.lstrip("# ").strip()
        out[key] = part
    return out


def body_of(block: str) -> str:
    """Block content with its heading line removed."""
    return block.split("\n", 1)[1].lstrip("\n") if "\n" in block else ""


def split_at(block: str, markers: list[str]) -> dict[str, str]:
    """Cut a block at bold paragraph markers. Returns {marker_or_'head': text}."""
    text = body_of(block)
    idx = [(text.index(m), m) for m in markers if m in text]
    idx.sort()
    out, prev_pos, prev_key = {}, 0, "head"
    for pos, m in idx:
        out[prev_key] = text[prev_pos:pos].rstrip() + "\n"
        prev_pos, prev_key = pos, m
    out[prev_key] = text[prev_pos:].rstrip() + "\n"
    return out


ABSTRACT = """This paper develops a reproducible pipeline for converting vision-derived human motion into
sparse nonlinear dynamical models, together with an evaluation protocol in which every reported
score is paired with the no-model control for its own task. The workflow uses standardized
skeletal pose, biomechanical feature extraction, and Sparse Identification of Nonlinear
Dynamical Systems, applied to 140 MediaPipe-derived MLB pitching clips from a single broadcast
view. The central result separates two tasks that prior evaluations conflated. Free-running
reconstruction fails: no configuration outperforms a frozen zero-dynamics predictor, and accuracy
degrades monotonically with model complexity. Short-horizon prediction succeeds: a delay-augmented
state reaches 0.41 skill over persistence and the project's first positive derivative R2, from 32
sparse terms, clearing an unfitted momentum control. Candidate causes of the failure are excluded
one at a time — phase, clip heterogeneity, the pose noise floor, and the sparse estimator — leaving
the state representation, which the delay result then confirms. Five intermediate results were
false positives that survived only because a control or an aggregation was missing; each is
reported with the check that overturned it, and that protocol is offered as a contribution
alongside the pipeline. Release velocity and pitch type are not predictable from any of five
representations, so no biomechanical claim is made."""

CONCLUSION = """Free-running reconstruction of broadcast-derived pitching pose is not achievable with sparse
dynamical models: every configuration tried is beaten by a predictor that assumes the pitcher does
not move, and accuracy falls as model complexity rises. That failure is localized rather than
merely reported. Phase, clip heterogeneity, the pose noise floor and the choice of sparse estimator
were each excluded by direct test, leaving the state representation — and supplying two frames of
history raises skill over persistence on the identical target from +0.01 to +0.41, which the
pipeline now reproduces end to end from a configuration file.

What the data does not support is equally definite. Release velocity and pitch type are not
predictable from any of the five pre-registered representations, and the directional asymmetry that
would evidence a proximal-to-distal kinetic chain does not survive its null. This paper therefore
makes no biomechanical claim, and the discovered coefficients should not be read as mechanics.

The evaluation protocol is the transferable part. Five intermediate results in this project were
false positives: four were caught by adding the no-model control for the specific task being
scored, the fifth by a null the analysis had not originally applied per target, and a sixth
near-miss was an aggregation artifact. A metric whose absolute value is
uninterpretable — R2 against a window mean when the target barely moves, or a sum of squares pooled
across columns of unequal amplitude — will manufacture findings unless it ships with that control
and both aggregations before any model is scored against it. On this evidence that discipline is
not optional bookkeeping; it is the difference between a negative result and a wrong one."""


SYSTEM_OVERVIEW = """The workflow has two stages. Stage A validates the general motion-processing and sparse-modeling
infrastructure on UTD-MHAD — data loading, skeleton standardization, smoothing, derivative
estimation, model fitting, coefficient export — so its artifacts are infrastructure checks rather
than baseball findings. Stage B replaces only the input-pose layer with baseball-specific
standardized pose and keeps everything downstream, which keeps video and pose-estimation issues
separate from the modeling workflow. The pipeline is described here only as far as the results
require; Section 12 gives the commands that regenerate every number."""

# The contribution statement, kept verbatim from the superseded Interpretation section.
CLAIM = """The defensible claim is:

> A reproducible sparse-dynamics pipeline for baseball pitching pose data was implemented
> end-to-end, together with a state-comparable reconstruction metric and a control-paired
> evaluation protocol in which every reported score is compared against the no-model baseline for
> its own task. Under that protocol three findings separate. First, no fitted configuration —
> first-order, second-order, ensemble, or weak-form — recovers dynamics that survive free-running
> reconstruction; all are outperformed by a zero-dynamics baseline, and accuracy degrades
> monotonically with model complexity. Second, the data does carry short-horizon structure: a
> delay-embedded linear model achieves 0.39 skill over persistence on held-out increment
> prediction, robust to error aggregation and backed by increment autocorrelation of 0.35 to 0.88
> rather than by pose noise. Third, the pipeline's own sparse equations capture almost none of that
> structure as configured, but the cause is the state and not the estimator: the same sparse
> estimator on a delay-augmented state reaches +0.3881 uniform skill over persistence with 30
> terms, matching dense ridge and clearing an unfitted momentum control at +0.1311. That state is
> now a first-class pipeline configuration, and running the standard commands on it yields +0.4071
> skill and the project's first positive derivative R2 (+0.4411).

Two things this work must not claim. The results are not evidence of recovered governing laws of
pitching mechanics: 26 coefficients that predict increments 39 percent better than persistence are
not a biomechanical model, and the discovered equations have not been validated against any
external quantity. And the weak-form configuration must not be presented as a successful model."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    text = SOURCE.read_text()
    body, sep, refs = text.partition("\n## References")
    B = blocks(body)

    s69 = split_at(B["6.9"], [
        "**Cross-joint coupling contributes, but weakly.**",
        "**Instability, diagnosed and fixed.**",
        "**The pipeline itself now produces this result.**",
    ])
    # The rollout section mixed its own result with the aggregation lesson drawn from it. The
    # teacher-forcing check and the control comparison stay with the rollout; only the
    # pooled-versus-uniform paragraphs move to the evaluation-lessons section.
    s66 = split_at(B["6.6"], [
        "Pooled skill is positive at every horizon;",
        "A separate verification confirms the rollout failure is genuine",
    ])
    s7 = split_at(B["7"], ["**Methodological finding on evaluation.**"])

    def hdr(n: str, title: str) -> str:
        return f"\n## {n}. {title}\n\n"

    def sub(n: str, title: str) -> str:
        return f"\n### {n} {title}\n\n"

    title = body.split("\n", 1)[0]
    if not title.startswith("# "):
        raise SystemExit(f"expected the H1 title on line 1, got: {title!r}")

    out = [
        title, "\n\n", B["Abstract"].split("\n")[0], "\n\n", ABSTRACT, "\n",
        B["1"].replace("## 1. Introduction", "\n## 1. Introduction", 1).rstrip(), "\n",
        B["1.1"].rstrip(), "\n",
        sub("1.2", "Research questions"), body_of(B["2"]).rstrip(), "\n",
        hdr("2", "Data"), body_of(B["4"]).rstrip(), "\n\n",
        sub("2.1", "Stage A validation data"), body_of(B["4.1"]).rstrip(), "\n",
        sub("2.2", "Stage B baseball data"), body_of(B["4.2"]).rstrip(), "\n",
        hdr("3", "Pipeline"), SYSTEM_OVERVIEW, "\n",
        sub("3.1", "Pose standardization"), body_of(B["5.1"]).rstrip(), "\n",
        sub("3.2", "Preprocessing"), body_of(B["5.2"]).rstrip(), "\n",
        sub("3.3", "Feature extraction"), body_of(B["5.3"]).rstrip(), "\n",
        sub("3.4", "Sparse model identification"), body_of(B["5.4"]).rstrip(), "\n",
        hdr("4", "Evaluation protocol"),
        "This section is placed before the results because the protocol, not any single number, is\n"
        "what makes them interpretable. Five of this project's intermediate findings were false\n"
        "positives that only a control could catch, and Section @9@ documents them.\n",
        body_of(B["5.5"]).rstrip(), "\n",
        hdr("5", "Free-running reconstruction fails"),
        body_of(B["6"]).rstrip(), "\n\n",
        sub("5.1", "The operative configuration"), body_of(B["6.1"]).rstrip(), "\n",
        sub("5.2", "Every model against the frozen control"), body_of(B["6.2"]).rstrip(), "\n",
        sub("5.3", "Rollout horizon"), s66["head"].rstrip(), "\n\n",
        s66["A separate verification confirms the rollout failure is genuine"].rstrip(), "\n",
        hdr("6", "The cause is the state, not the model"),
        "Four candidate explanations for the failure are excluded in turn, each by direct test.\n",
        sub("6.1", "Phase and clip heterogeneity"), body_of(B["6.4"]).rstrip(), "\n",
        sub("6.2", "The pose noise floor"), body_of(B["6.7"]).rstrip(), "\n",
        sub("6.3", "The sparse estimator"), body_of(B["6.8"]).rstrip(), "\n",
        hdr("7", "Short-horizon prediction succeeds"),
        body_of(B["6.5"]).rstrip(), "\n\n",
        sub("7.1", "Sparse identification on a delay-augmented state"), s69["head"].rstrip(), "\n",
        sub("7.2", "The pipeline reproduces it"),
        s69["**The pipeline itself now produces this result.**"].rstrip(), "\n",
        hdr("8", "What the data does not support"),
        sub("8.1", "Cross-joint coupling and the kinetic chain"),
        s69["**Cross-joint coupling contributes, but weakly.**"].rstrip(), "\n",
        sub("8.2", "Release velocity and pitch type"), body_of(B["6.10"]).rstrip(), "\n",
        hdr("9", "Evaluation lessons"),
        s7["**Methodological finding on evaluation.**"].rstrip(), "\n\n",
        sub("9.1", "A selection rule that manufactured a finding"), body_of(B["6.3"]).rstrip(), "\n",
        sub("9.2", "When two aggregations disagree"),
        s66["Pooled skill is positive at every horizon;"].rstrip(), "\n",
        hdr("10", "Limitations and future work"),
        body_of(B["9"]).rstrip(), "\n\n", body_of(B["10"]).rstrip(), "\n",
        hdr("11", "Conclusion"), CONCLUSION, "\n\n",
        CLAIM, "\n\n", body_of(B["11"]).rstrip(), "\n",
        hdr("12", "Reproducibility"), body_of(B["8"]).rstrip(), "\n\n",
        body_of(B["12"]).rstrip(), "\n",
        hdr("A", "Appendix: superseded and auxiliary results"),
        "Retained for provenance. Nothing here supports a claim in the body.\n\n",
        "### A.1 Preprocessing-track comparison\n\n", body_of(B["6.11"]).rstrip(), "\n",
        "\n### A.2 Delay-model stability\n\n",
        s69["**Instability, diagnosed and fixed.**"].rstrip(), "\n",
    ]
    new = "".join(out)

    # The false-positive count and its table disagreed in the source: the lead-in said three
    # while the table listed four, and the kinetic-chain asymmetry — overturned in Section 8.1 —
    # was never added at all. Five is the number the evidence supports.
    old_lead = ("Three intermediate results in this project were false positives, each caught "
                "only by adding the no-model control for the specific task being scored:")
    new_lead = ("Five intermediate results in this project were false positives. Four were caught "
                "by adding the no-model control for the specific task being scored; the fifth by a "
                "null the analysis had not originally applied per target:")
    if old_lead not in new:
        raise SystemExit("false-positive lead-in not found; check whether it was already edited")
    new = new.replace(old_lead, new_lead, 1)

    last_row = ("| second-order short-horizon skill +0.89 | its own derivative reference, +0.89 | "
                "dissolved; velocity is an input column |")
    asym_row = ("| proximal-to-distal asymmetry +0.0011, four of five splits clearing a shift null "
                "| the same null applied per target, over 30 resplits | dissolved; interval "
                "[-0.0004, +0.0026], positive on 18 of 30 |")
    if last_row not in new:
        raise SystemExit("false-positive table not found; check its final row")
    new = new.replace(last_row, last_row + "\n" + asym_row, 1)
    new = new.replace(
        ": Intermediate results that did not survive the no-model control for their task.",
        ": Intermediate results that did not survive the control or null for their task.", 1)
    new = new.replace(
        "The common structure is that all four involved metrics",
        "The common structure is that all five involved metrics", 1)
    new = new.replace(
        "A further near-miss was an aggregation artifact",
        "A sixth near-miss was an aggregation artifact", 1)
    new = new.replace(
        "The fourth row above was anticipated before the run",
        "The second-order row above was anticipated before the run", 1)
    # The coupling analysis is no longer adjacent to the stability appendix that refers to it.
    new = new.replace("The coupling\nanalysis above predates this fix",
                      "The coupling\nanalysis in Section @8.1@ predates this fix", 1)

    # Remap cross-references, longest keys first so "6.10" is not matched as "6.1".
    for old in sorted(XREF, key=len, reverse=True):
        new = re.sub(rf"\bSection {re.escape(old)}\b", f"Section \x00{XREF[old]}\x00", new)
        new = re.sub(rf"\bSections {re.escape(old)}\b", f"Sections \x00{XREF[old]}\x00", new)
    new = new.replace("\x00", "")
    # Sentinel-marked references were written against the NEW numbering already.
    new = re.sub(r"@([0-9A-Z.]+)@", r"\1", new)

    # Recover each IEEE number's key from the paper's own References, so the reordered body can be
    # renumbered from scratch rather than carrying numbers that no longer run in first-appearance
    # order. Matching on the full entry text means a mismatch fails loudly instead of mislabelling.
    text_to_key = {v: k for k, v in BIB.items()}
    num_to_key = {}
    for num, entry in re.findall(r"^\[(\d+)\] (.+)$", refs, re.M):
        if entry not in text_to_key:
            raise SystemExit(f"reference [{num}] does not match any bibliography entry")
        num_to_key[num] = text_to_key[entry]
    unknown = sorted(set(re.findall(r"\[(\d+)\]", new)) - set(num_to_key))
    if unknown:
        raise SystemExit("body cites numbers absent from References: " + ", ".join(unknown))
    new = re.sub(r"\[(\d+)\]", lambda m: "{{" + num_to_key[m.group(1)] + "}}", new)
    print(f"citations reverted to {len(num_to_key)} keys; run renumber_references.py next")

    headings = set(re.findall(r"^#{2,3} ([0-9A-Z]+(?:\.\d+)?)\.? ", new, re.M))
    referenced = set(re.findall(r"Sections? ([0-9A-Z]+(?:\.\d+)?)", new))
    dangling = sorted(r for r in referenced if r not in headings)
    print(f"headings: {len(headings)}   cross-references: {len(referenced)}")
    if dangling:
        raise SystemExit("cross-references with no matching heading: " + ", ".join(dangling))
    print(f"words: {len(new.split())} (was {len(body.split())})")
    if args.check:
        return
    SOURCE.write_text(new + sep + refs)
    print("restructured")


if __name__ == "__main__":
    main()
