#!/usr/bin/env python3
"""Narrow the paper onto its two findings and move the secondary analyses into appendices.

Reviewer feedback on the previous draft: the strongest part is the separation of the failed
free-running result from the positive short-horizon delay result, and the next draft should be
narrowed around those two findings with more of the secondary analysis moved to an appendix.
Nothing is cut -- the reviewer was explicit that the draft has not gone too far -- so this moves
blocks and rewrites only headings, signposts and cross-references.

Body, after:      1 Intro, 2 Data, 3 Pipeline, 4 Evaluation protocol,
                  5 Finding one (free-running fails), 6 The cause is the state,
                  7 Finding two (short-horizon succeeds), 8 Evaluation lessons,
                  9 Limitations, 10 Conclusion, 11 Reproducibility.
Appendices, after: A coupling and the kinetic chain (was 8.1)
                   B release velocity and pitch type (was 8.2)
                   C the selection rule that manufactured a finding (was 9.1)
                   D when two aggregations disagree (was 9.2)
                   E preprocessing-track comparison (was A.1)
                   F delay-model stability (was A.2)

Appendices are lettered top-level sections rather than subsections of one appendix, so the
in-text form is "Appendix C", which needs no remapping in the IEEE build: IEEEtran letters
appendices in the same order.

Cross-references are remapped in two passes. Section 10 and 12 shift by one and are done with a
regex; every reference into the moved material is context-dependent (the same "Section 7" means
Section 7 in one place and Appendix A in another) so those are exact-string edits that fail loudly
if the source has drifted. Every "Section N" and "Appendix X" in the result must resolve to a
heading, including the trailing member of a pair or range -- the check the previous restructure
missed, which is how "Sections 5 to 6.8" survived into the last draft.

Moving blocks invalidates the IEEE citation numbers, which run in order of first appearance, so
this reverts them to `{{key}}` placeholders. Run scripts/renumber_references.py afterwards.

Usage:
    python scripts/narrow_paper.py [--check]
    python scripts/renumber_references.py
    python scripts/build_paper_pdf.py --style ieee -o YU_TECHNICAL_PAPER.pdf
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


def blocks(text: str) -> dict[str, str]:
    """Split into blocks keyed by heading id ('6.3', 'A.1', 'Abstract', 'References')."""
    out: dict[str, str] = {}
    for part in re.split(r"\n(?=#{2,3} )", text):
        head = part.split("\n", 1)[0]
        m = re.match(r"^#{2,3} ([0-9A-Z]+(?:\.\d+)?)\.? ", head)
        key = m.group(1) if m else head.lstrip("# ").strip()
        if key in out:
            raise SystemExit(f"duplicate heading id {key!r}")
        out[key] = part
    return out


def body_of(block: str) -> str:
    """Block content with its heading line removed."""
    return block.split("\n", 1)[1].lstrip("\n") if "\n" in block else ""


def hdr(num: str, title: str) -> str:
    return f"\n## {num}. {title}\n\n"


def sub(num: str, title: str) -> str:
    return f"\n### {num} {title}\n\n"


# --- Replaced prose ------------------------------------------------------------------------

# Reordered so the two findings arrive first and in the order the body argues them; the
# protocol and the null secondary analyses are demoted to one sentence each. Every figure
# here is the one the body reports: +0.39 is the delay ridge and the 30-term sparse fit
# (Section 7, 7.1), +0.41 and +0.44 are the pipeline's own run (Section 7.2).
ABSTRACT = """This paper reports two findings about sparse dynamical modeling of vision-derived baseball
pitching motion, separated by the task each is scored on. The pipeline converts standardized
skeletal pose into biomechanical features and fits sparse nonlinear dynamical models by Sparse
Identification of Nonlinear Dynamical Systems, applied to 140 MediaPipe-derived MLB pitching
clips from a single broadcast view. Free-running reconstruction fails: across first-order,
second-order, ensemble and weak-form configurations, no fitted model outperforms a frozen
zero-dynamics predictor on held-out clips, and accuracy degrades monotonically as model
complexity rises. Short-horizon prediction succeeds on the same clips: a delay-augmented state
reaches 0.39 skill over persistence, the same sparse estimator retains that skill with 30 terms
while clearing an unfitted momentum control, and running the pipeline end to end on that
configuration yields 0.41 skill and the project's first positive held-out derivative R2 of 0.44.
Between the two, candidate causes of the failure are excluded one at a time — phase, clip
heterogeneity, the pose noise floor and the sparse estimator — leaving the state representation,
which the second finding confirms. Both are readable only because every score is paired with the
no-model control for its own task and reported under two aggregations; five intermediate results
in this project were false positives that survived only because a control or an aggregation was
missing, and that protocol is offered as a contribution alongside the pipeline. The secondary
analyses are null or negligible and are reported in appendices: cross-joint coupling accounts for
under two percent of the delay model's skill, the proximal-to-distal kinetic chain does not
survive its null, and release velocity and pitch type are not predictable from any of five
pre-registered representations. No biomechanical claim is made."""

# New: the reviewer asked for a tighter central story, which the draft argued but never stated
# up front. This is the only place the reader is told what the two findings are and where the
# secondary material went.
ROADMAP = """This paper is organized around two findings that differ only in the task being scored, and the
reader who wants the argument and nothing else can read Sections 4 through 7. Section 4 states
the evaluation protocol, because no score below is interpretable without the no-model control
for its own task. Section 5 reports the negative finding: free-running reconstruction of held-out
pitching clips is not achievable with any configuration tried, and every fitted model is beaten
by a predictor that assumes the pitcher does not move. Section 6 localizes that failure by
excluding phase, clip heterogeneity, the pose noise floor and the sparse estimator in turn,
leaving the state representation. Section 7 reports the positive finding: on the same clips a
delay-augmented state supports short-horizon prediction, the pipeline's own sparse estimator
recovers it, and that is the diagnosis of Section 6 confirmed. Section 8 records the five
intermediate results the protocol overturned. The secondary analyses bear on neither finding and
are reported in Appendices A through F, where the two that bound what may be claimed —
cross-joint coupling with the kinetic chain, and the label-validated representation study — are
Appendices A and B."""

# Narrowed from six flat questions to the two the paper answers, the one diagnostic that
# follows, and a sentence disposing of the three engineering prerequisites.
QUESTIONS = """Two questions are primary, and they differ only in the task being scored:

1. Do the fitted sparse systems outperform a no-dynamics control baseline on held-out
   free-running reconstruction?
2. Do they outperform the corresponding no-model control on short-horizon prediction, and does
   the answer differ from question 1?

One diagnostic question follows from the answers:

3. Where the two answers differ, is the limiting factor the model class, the state
   representation, the clip heterogeneity, or the pose noise floor?

Three further questions are engineering prerequisites rather than findings, and are answered in
passing: whether raw or vision-derived skeletal trajectories can be converted into standardized
pitching feature matrices without manual per-clip intervention (Section 3), whether derivative
and sparse coefficient matrices regenerate reproducibly from those feature matrices (Section 11),
and whether the fitted systems are numerically stable on held-out clips (Section 5.1 — they are,
and Section 4 explains why that is close to uninformative)."""

# New: closes the two findings by naming what was tested and found absent, so the body still
# states the negative space in three sentences after the analyses themselves moved out.
NULL_POINTER = """Two further analyses were run on this model and neither supports a claim. Cross-joint coupling
is statistically detectable but accounts for about 1.8 percent of its skill, and the directional
asymmetry that would evidence a proximal-to-distal kinetic chain does not survive across
held-out splits. Release velocity and pitch type are not predictable from any of five
pre-registered representations, none of which clears both its trivial baseline and a permutation
null. Appendices A and B report both in full, and together they are why this paper makes no
biomechanical claim."""

# New: replaces the two subsections that moved to Appendices C and D.
LESSONS_POINTER = """The two cases worth reading in detail are in the appendices. Appendix C is the selection rule
that manufactured the weak-form result, retained because the repair is a change to the pipeline
rather than to the write-up. Appendix D is the aggregation disagreement behind the sixth
near-miss."""

# New: Appendix A opens mid-argument otherwise, because it was written as a subsection following
# Section 7.
COUPLING_LEAD = """This appendix ablates the depth-3 delay model of Section 7 to ask whether its skill reflects
coordination between joints, and in particular whether it evidences the proximal-to-distal
kinetic chain. It does not, which is why Section 7 claims short-horizon predictive structure and
nothing biomechanical.

"""

# The Section 7 limitation bullet was written before the sparse fit on a delay state existed and
# still says the positive result is not a SINDy result, which Section 7.1 contradicts.
OLD_ATTRIBUTION = (
    "- The Section 7 skill figure comes from a delay-embedded ridge model, not from a sparse "
    "SINDy fit. Section 6 tests the fitted equations directly and finds they capture 0 percent "
    "(first-order) to 12 percent (second-order, acceleration) of the structure available in "
    "their own training targets. The short-horizon result therefore characterizes the data, not "
    "the pipeline's sparse artifacts, and must not be attributed to SINDy."
)
NEW_ATTRIBUTION = (
    "- The Section 7 skill figure was first obtained with a delay-embedded ridge model. Section "
    "7.1 shows the project's own sparse estimator matches it on a delay-augmented state, so the "
    "result does belong to the method — but it belongs to that state, not to the operative "
    "configuration: Section 6.3 tests the operative fitted equations directly and finds they "
    "capture 0 percent (first-order) to 12 percent (second-order, acceleration) of the structure "
    "available in their own training targets. Short-horizon skill must be attributed to the "
    "delay state and not to the configuration Section 5 evaluates."
)

# Context-dependent cross-references into the moved material. Exact strings, each applied once,
# each required to be present.
XREF_EXACT: list[tuple[str, str]] = [
    ("kinetic chain this paper tests against in Section 7 [2]",
     "kinetic chain this paper tests against in Appendix A [2]"),
    ("Sections 5 to 6.8 show both differences bite.",
     "Sections 5 and 6 show both differences bite."),
    ("which is the prior against which Section 8 should be",
     "which is the prior against which Appendix B should be"),
    ("The delay-coordinate results in Sections 6 and 6.9 rest on",
     "The delay-coordinate results in Sections 6 and 7 rest on"),
    ("counterpart used in Section 8.", "counterpart used in Appendix B."),
    ("Section A records why.", "Appendix E records why."),
    ("Section 9 documents a degeneracy in this selection rule.",
     "Appendix C documents a degeneracy in this selection rule."),
    ("The noise floor does not explain the failures in Sections 5 and 6.6",
     "The noise floor does not explain the failures in Sections 5 and 6"),
    ("Section 7 established short-horizon structure using a delay-embedded ridge model",
     "Section 7 establishes short-horizon structure using a delay-embedded ridge model"),
    ("- The Section 8 null cannot distinguish absence",
     "- The Appendix B null cannot distinguish absence"),
    ("are indistinguishable (Section 9); selected sparsity levels",
     "are indistinguishable (Appendix C); selected sparsity levels"),
    ("Sections 5, 6.8 and 6.9 now carry intervals — paired bootstrap over clips for the "
     "control comparison, resplits elsewhere —",
     "The control comparison in Section 5.2, the estimator comparison in Section 6.3 and the "
     "coupling test in Appendix A now carry intervals — paired bootstrap over clips for the "
     "first, resplits for the others —"),
    ("(Section A). It has never been evaluated",
     "(Appendix E). It has never been evaluated"),
    ("bar (Section 8), including the delay-embedded spectrum",
     "bar (Appendix B), including the delay-embedded spectrum"),
    ("previously used (Sections 9, 6.4, 6.5, 6.6)",
     "previously used (Sections 5.2, 6.3, 7 and 8)"),
    ("Reproduce Section 9 (weak-form model).",
     "Reproduce the weak-form model of Section 5.2 and Appendix C."),
    ("Reproduce Sections 7, 6.6 and 6.7 (one-step skill, horizon sweep, noise floor):",
     "Reproduce Section 7, Section 6.2 and Appendix D (one-step skill, noise floor, horizon "
     "sweep):"),
    ("Reproduce Section 7 (sparse fit on a delay-augmented state, with the momentum controls",
     "Reproduce Section 7 and Appendix A (sparse fit on a delay-augmented state, with the "
     "momentum controls"),
    ("Reproduce Section 8 (label-validated representation study).",
     "Reproduce Appendix B (label-validated representation study)."),
    ("Reproduce the Section 7 coupling test (five held-out splits, 60 null draws each):",
     "Reproduce the Appendix A coupling test (five held-out splits, 60 null draws each):"),
    ("Compare the old and repaired selection rules (Section 9):",
     "Compare the old and repaired selection rules (Appendix C):"),
    ("Reproduce the Section 7 stability comparison:",
     "Reproduce the Appendix F stability comparison:"),
    ("Reproduce the Section 7 directional asymmetry test:",
     "Reproduce the Appendix A directional asymmetry test:"),
    ("analysis in Section 8.1 predates this fix", "analysis in Appendix A predates this fix"),
    ("Every result above judges the pipeline by self-prediction of the motion.",
     "Every result in the body judges the pipeline by self-prediction of the motion."),
    ("and Section 9 documents them.", "and Section 8 documents them."),
]

# Body sections that shift by one now that Section 8 and the two Section 9 subsections have left.
XREF_SHIFT = {"12": "11", "10": "9"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Report only; do not write.")
    args = ap.parse_args()

    text = SOURCE.read_text()
    B = blocks(text)
    # The H1 title is not a '##' block, so it rides along at the head of the first split.
    head, _, _ = text.partition("\n## Abstract")
    refs = B["References"]

    out = [
        head.rstrip(), "\n\n## Abstract\n\n", ABSTRACT, "\n",
        hdr("1", "Introduction"), body_of(B["1"]).rstrip(), "\n\n", ROADMAP, "\n",
        sub("1.1", "Related work"), body_of(B["1.1"]).rstrip(), "\n",
        sub("1.2", "Research questions"), QUESTIONS, "\n",
        hdr("2", "Data"), body_of(B["2"]).rstrip(), "\n",
        sub("2.1", "Stage A validation data"), body_of(B["2.1"]).rstrip(), "\n",
        sub("2.2", "Stage B baseball data"), body_of(B["2.2"]).rstrip(), "\n",
        hdr("3", "Pipeline"), body_of(B["3"]).rstrip(), "\n",
        sub("3.1", "Pose standardization"), body_of(B["3.1"]).rstrip(), "\n",
        sub("3.2", "Preprocessing"), body_of(B["3.2"]).rstrip(), "\n",
        sub("3.3", "Feature extraction"), body_of(B["3.3"]).rstrip(), "\n",
        sub("3.4", "Sparse model identification"), body_of(B["3.4"]).rstrip(), "\n",
        hdr("4", "Evaluation protocol"), body_of(B["4"]).rstrip(), "\n",
        hdr("5", "Free-running reconstruction fails"), body_of(B["5"]).rstrip(), "\n",
        sub("5.1", "The operative configuration"), body_of(B["5.1"]).rstrip(), "\n",
        sub("5.2", "Every model against the frozen control"), body_of(B["5.2"]).rstrip(), "\n",
        sub("5.3", "Rollout horizon"), body_of(B["5.3"]).rstrip(), "\n",
        hdr("6", "The cause is the state, not the model"), body_of(B["6"]).rstrip(), "\n",
        sub("6.1", "Phase and clip heterogeneity"), body_of(B["6.1"]).rstrip(), "\n",
        sub("6.2", "The pose noise floor"), body_of(B["6.2"]).rstrip(), "\n",
        sub("6.3", "The sparse estimator"), body_of(B["6.3"]).rstrip(), "\n",
        hdr("7", "Short-horizon prediction succeeds"), body_of(B["7"]).rstrip(), "\n",
        sub("7.1", "Sparse identification on a delay-augmented state"),
        body_of(B["7.1"]).rstrip(), "\n",
        sub("7.2", "The pipeline reproduces it"), body_of(B["7.2"]).rstrip(), "\n\n",
        NULL_POINTER, "\n",
        hdr("8", "Evaluation lessons"), body_of(B["9"]).rstrip(), "\n\n", LESSONS_POINTER, "\n",
        hdr("9", "Limitations and future work"), body_of(B["10"]).rstrip(), "\n",
        hdr("10", "Conclusion"), body_of(B["11"]).rstrip(), "\n",
        hdr("11", "Reproducibility"), body_of(B["12"]).rstrip(), "\n",
        hdr("A", "Cross-joint coupling and the kinetic chain"),
        COUPLING_LEAD, body_of(B["8.1"]).rstrip(), "\n",
        hdr("B", "Release velocity and pitch type"), body_of(B["8.2"]).rstrip(), "\n",
        hdr("C", "A selection rule that manufactured a finding"),
        body_of(B["9.1"]).rstrip(), "\n",
        hdr("D", "When two aggregations disagree"), body_of(B["9.2"]).rstrip(), "\n",
        hdr("E", "Preprocessing-track comparison"), body_of(B["A.1"]).rstrip(), "\n",
        hdr("F", "Delay-model stability"), body_of(B["A.2"]).rstrip(), "\n",
        "\n", refs.rstrip(), "\n",
    ]
    new = "".join(out)

    if body_of(B["8"]).strip():
        raise SystemExit("Section 8 has body text of its own; it would be dropped by this move")

    # Exact-string edits first: the shift regex below would otherwise rewrite numbers inside them.
    for old, repl in XREF_EXACT + [(OLD_ATTRIBUTION, NEW_ATTRIBUTION)]:
        if new.count(old) != 1:
            raise SystemExit(f"expected exactly one occurrence of {old[:60]!r}, "
                             f"found {new.count(old)}")
        new = new.replace(old, repl, 1)

    for old in sorted(XREF_SHIFT, key=len, reverse=True):
        new = re.sub(rf"\bSection {re.escape(old)}\b", f"Section \x00{XREF_SHIFT[old]}\x00", new)
        new = re.sub(rf"\bSections {re.escape(old)}\b", f"Sections \x00{XREF_SHIFT[old]}\x00", new)
    new = new.replace("\x00", "")

    # Every "Section N", "Section N.M" and "Appendix X" must resolve, including the trailing
    # member of "Sections 5 and 6.3" or "Sections 5 to 6" -- the case the last restructure's
    # check did not look at, which is how stale references survived into the previous draft.
    headings = set(re.findall(r"^#{2,3} ([0-9A-Z]+(?:\.\d+)?)\.? ", new, re.M))
    referenced: set[str] = set()
    # Plural "Sections" may carry a list; singular must not, or "(Section 5, 13 clips)" would
    # read 13 as a section number.
    for m in re.finditer(r"\bSections ([0-9]+(?:\.[0-9]+)?)"
                         r"((?:(?:,| and| to) [0-9]+(?:\.[0-9]+)?)*)", new):
        referenced.add(m.group(1))
        referenced.update(re.findall(r"[0-9]+(?:\.[0-9]+)?", m.group(2) or ""))
    referenced.update(re.findall(r"\bSection ([0-9]+(?:\.[0-9]+)?)", new))
    appendices = set(re.findall(r"\bAppendix ([A-Z])\b", new))
    for m in re.finditer(r"\bAppendices ([A-Z]) (?:through|and) ([A-Z])\b", new):
        appendices.update(m.groups())
    dangling = sorted((referenced | appendices) - headings)
    if dangling:
        raise SystemExit("cross-references with no matching heading: " + ", ".join(dangling))
    stale = sorted(set(re.findall(r"\bSections? ([0-9]+\.[0-9]+)", new)) - headings)
    if stale:
        raise SystemExit("stale subsection references: " + ", ".join(stale))

    # Recover each IEEE number's key from the paper's own References so the reordered body can be
    # renumbered from scratch: moving Section 8 behind Section 11 changes which section cites the
    # permutation-test references first.
    text_to_key = {v: k for k, v in BIB.items()}
    num_to_key = {}
    for num, entry in re.findall(r"^\[(\d+)\] (.+)$", refs, re.M):
        if entry not in text_to_key:
            raise SystemExit(f"reference [{num}] does not match any bibliography entry")
        num_to_key[num] = text_to_key[entry]
    body_only = new[: new.index("\n## References")]
    unknown = sorted(set(re.findall(r"\[(\d+)\]", body_only)) - set(num_to_key))
    if unknown:
        raise SystemExit("body cites numbers absent from References: " + ", ".join(unknown))
    new = re.sub(r"\[(\d+)\]", lambda m: "{{" + num_to_key[m.group(1)] + "}}", body_only)
    new = new + text[text.index("\n## References"):].rstrip() + "\n"
    print(f"citations reverted to {len(num_to_key)} keys; run renumber_references.py next")

    body_words = len(re.split(r"\n## A\. ", new)[0].split())
    old_body_words = len(re.split(r"\n## A\. ", text)[0].split())
    print(f"headings: {len(sorted(headings))}   cross-references: {len(referenced | appendices)}")
    print(f"body words: {body_words} (was {old_body_words})   total: {len(new.split())} "
          f"(was {len(text.split())})")
    if args.check:
        return
    SOURCE.write_text(new)
    print("narrowed")


if __name__ == "__main__":
    main()
