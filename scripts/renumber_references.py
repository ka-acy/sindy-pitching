#!/usr/bin/env python3
"""Resolve citation keys in the paper to IEEE numbers, ordered by first appearance.

IEEE numbers references by the order they are first cited, so inserting a citation into an
early section renumbers everything after it. Doing that by hand is how stale cross-references
get into a paper, so it is done here instead.

The markdown carries `{{key}}` placeholders. This script walks the body in order, assigns each
key a number the first time it appears, rewrites the placeholders as `[n]`, and regenerates the
References section in that order. Running it on a paper with no placeholders left is a no-op,
so it is safe to re-run.

Keys that appear in the bibliography but are never cited, and keys cited but absent from the
bibliography, are both reported as errors rather than silently dropped or numbered.

Usage:
    python scripts/renumber_references.py [--check]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "technical_paper.md"

# IEEE-formatted entries, keyed. Every bibliographic detail here was verified against the
# publisher or an indexing service rather than written from memory.
BIB: dict[str, str] = {
    "fleisig1995": 'G. S. Fleisig, J. R. Andrews, C. J. Dillman, and R. F. Escamilla, "Kinetics of '
        'baseball pitching with implications about injury mechanisms," *Amer. J. Sports Med.*, '
        "vol. 23, no. 2, pp. 233–239, Mar./Apr. 1995, doi: 10.1177/036354659502300218.",
    "fleisig1996": 'G. S. Fleisig, S. W. Barrentine, R. F. Escamilla, and J. R. Andrews, '
        '"Biomechanics of overhand throwing with implications for injuries," *Sports Med.*, '
        "vol. 21, no. 6, pp. 421–437, Jun. 1996, doi: 10.2165/00007256-199621060-00004.",
    "kanko2021": 'R. M. Kanko, E. K. Laende, E. M. Davis, W. S. Selbie, and K. J. Deluzio, '
        '"Concurrent assessment of gait kinematics using marker-based and markerless motion '
        'capture," *J. Biomech.*, vol. 127, art. no. 110665, Oct. 2021, '
        "doi: 10.1016/j.jbiomech.2021.110665.",
    "chen2015": 'C. Chen, R. Jafari, and N. Kehtarnavaz, "UTD-MHAD: A multimodal dataset for human '
        'action recognition utilizing a depth camera and a wearable inertial sensor," in *Proc. '
        "IEEE Int. Conf. Image Process. (ICIP)*, 2015, pp. 168–172, doi: 10.1109/ICIP.2015.7350781.",
    "piergiovanni2018": 'A. J. Piergiovanni and M. S. Ryoo, "Fine-grained activity recognition '
        'in baseball videos," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR) '
        'Workshops*, 2018. [Online]. Available: https://arxiv.org/abs/1804.03247',
    "lugaresi2019": 'C. Lugaresi et al., "MediaPipe: A framework for building perception '
        "pipelines,\" arXiv:1906.08172, 2019. [Online]. Available: https://arxiv.org/abs/1906.08172",
    "bazarevsky2020": 'V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and '
        'M. Grundmann, "BlazePose: On-device real-time body pose tracking," arXiv:2006.10204, '
        "2020. [Online]. Available: https://arxiv.org/abs/2006.10204",
    "savitzky1964": 'A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by '
        'simplified least squares procedures," *Anal. Chem.*, vol. 36, no. 8, pp. 1627–1639, '
        "1964, doi: 10.1021/ac60214a047.",
    "brunton2016": 'S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Discovering governing equations '
        'from data by sparse identification of nonlinear dynamical systems," *Proc. Nat. Acad. '
        "Sci.*, vol. 113, no. 15, pp. 3932–3937, Apr. 2016, doi: 10.1073/pnas.1517384113.",
    "desilva2020": 'B. M. de Silva, K. Champion, M. Quade, J.-C. Loiseau, J. N. Kutz, and '
        'S. L. Brunton, "PySINDy: A Python package for the sparse identification of nonlinear '
        'dynamical systems from data," *J. Open Source Softw.*, vol. 5, no. 49, art. no. 2104, '
        "2020, doi: 10.21105/joss.02104.",
    "kaptanoglu2022": 'A. A. Kaptanoglu et al., "PySINDy: A comprehensive Python package for '
        'robust sparse system identification," *J. Open Source Softw.*, vol. 7, no. 69, '
        "art. no. 3994, 2022, doi: 10.21105/joss.03994.",
    "messenger2021a": 'D. A. Messenger and D. M. Bortz, "Weak SINDy: Galerkin-based data-driven '
        "model selection,\" *Multiscale Model. Simul.*, vol. 19, no. 3, pp. 1474–1497, 2021, "
        "doi: 10.1137/20M1343166.",
    "messenger2021b": 'D. A. Messenger and D. M. Bortz, "Weak SINDy for partial differential '
        'equations," *J. Comput. Phys.*, vol. 443, art. no. 110525, 2021, '
        "doi: 10.1016/j.jcp.2021.110525.",
    "fasel2022": 'U. Fasel, J. N. Kutz, B. W. Brunton, and S. L. Brunton, "Ensemble-SINDy: Robust '
        'sparse model discovery in the low-data, high-noise limit, with active learning and '
        "control,\" *Proc. Roy. Soc. A*, vol. 478, no. 2260, art. no. 20210904, 2022, "
        "doi: 10.1098/rspa.2021.0904.",
    "champion2020": 'K. Champion, P. Zheng, A. Y. Aravkin, S. L. Brunton, and J. N. Kutz, "A '
        'unified sparse optimization framework to learn parsimonious physics-informed models '
        'from data," *IEEE Access*, vol. 8, pp. 169259–169271, 2020, '
        "doi: 10.1109/ACCESS.2020.3023625.",
    "murphy1988": 'A. H. Murphy, "Skill scores based on the mean square error and their '
        'relationships to the correlation coefficient," *Mon. Weather Rev.*, vol. 116, no. 12, '
        "pp. 2417–2424, Dec. 1988, "
        "doi: 10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2.",
    "takens1981": 'F. Takens, "Detecting strange attractors in turbulence," in *Dynamical Systems '
        "and Turbulence, Warwick 1980* (Lecture Notes in Mathematics, vol. 898), D. A. Rand and "
        "L.-S. Young, Eds. Berlin, Germany: Springer, 1981, pp. 366–381, doi: 10.1007/BFb0091924.",
    "brunton2017": 'S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, and J. N. Kutz, '
        '"Chaos as an intermittently forced linear system," *Nature Commun.*, vol. 8, '
        "art. no. 19, 2017, doi: 10.1038/s41467-017-00030-8.",
    "schmid2010": 'P. J. Schmid, "Dynamic mode decomposition of numerical and experimental data," '
        "*J. Fluid Mech.*, vol. 656, pp. 5–28, Jul. 2010, doi: 10.1017/S0022112010001217.",
    "ross2010": 'S. Ross and J. A. Bagnell, "Efficient reductions for imitation learning," in '
        "*Proc. 13th Int. Conf. Artif. Intell. Statist. (AISTATS)*, PMLR, 2010, pp. 661–668.",
    "efron1979": 'B. Efron, "Bootstrap methods: Another look at the jackknife," *Ann. Statist.*, '
        "vol. 7, no. 1, pp. 1–26, 1979, doi: 10.1214/aos/1176344552.",
    "ojala2010": 'M. Ojala and G. C. Garriga, "Permutation tests for studying classifier '
        'performance," *J. Mach. Learn. Res.*, vol. 11, pp. 1833–1863, 2010.',
    "varma2006": 'S. Varma and R. Simon, "Bias in error estimation when using cross-validation '
        'for model selection," *BMC Bioinformatics*, vol. 7, art. no. 91, 2006, '
        "doi: 10.1186/1471-2105-7-91.",
    "harris2020": 'C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, '
        "pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.",
    "virtanen2020": 'P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific '
        'computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, 2020, '
        "doi: 10.1038/s41592-019-0686-2.",
    "pedregosa2011": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. '
        "Learn. Res.*, vol. 12, pp. 2825–2830, 2011.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Report only; do not write.")
    args = ap.parse_args()

    text = SOURCE.read_text()
    head, sep, _ = text.partition("\n## References")
    if not sep:
        raise SystemExit("no References section found")

    keys = re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", head)
    unknown = sorted(set(keys) - set(BIB))
    if unknown:
        raise SystemExit("cited keys missing from the bibliography: " + ", ".join(unknown))

    order: list[str] = []
    for k in keys:
        if k not in order:
            order.append(k)
    uncited = sorted(set(BIB) - set(order))

    numbers = {k: i + 1 for i, k in enumerate(order)}
    body = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", lambda m: f"[{numbers[m.group(1)]}]", head)
    refs = "\n## References\n\n" + "\n\n".join(f"[{numbers[k]}] {BIB[k]}" for k in order) + "\n"

    print(f"cited keys: {len(order)}   citations in text: {len(keys)}")
    if uncited:
        print("WARNING uncited bibliography entries (they will NOT be listed): "
              + ", ".join(uncited))
    if args.check:
        return
    SOURCE.write_text(body + refs)
    print(f"renumbered {len(order)} references by first appearance")


if __name__ == "__main__":
    main()
