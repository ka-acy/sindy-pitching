#!/usr/bin/env python3
"""Build docs/technical_paper.md into a PDF via pandoc + pdflatex.

Requires `pdflatex` (texlive) on PATH. If `pandoc` is missing, install the bundled binary
into a scratch directory with:

    pip install --target <dir> pypandoc-binary

and pass --pandoc <dir>/pypandoc/files/pandoc.

Three things need handling that a bare `pandoc -o out.pdf` gets wrong here:

1. Only pdflatex is assumed available, not xelatex, so the paper's Unicode (minus sign,
   arrows, Greek, subscripts) is mapped to LaTeX equivalents first.
2. Long file paths in inline code cannot hyphenate and overflowed the text block by up to
   329pt. A Lua filter escapes code spans itself and inserts \\allowbreak after path
   separators -- no visible hyphen, which matters because a hyphen would be
   indistinguishable from a real character in a path or flag.
3. Pandoc emits non-wrapping l/r columns for tables that look narrow in Markdown source,
   which overflows for the tables holding prose. A second Lua filter assigns relative
   column widths so LaTeX wraps them.

Usage:
    python scripts/build_paper_pdf.py [--pandoc PATH] [-o OUT]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "technical_paper.md"

TITLE = "Sparse Dynamical Modeling of Baseball Pitching Kinematics from Vision-Derived Pose"
AUTHOR = "Alex Yu"

# pdflatex-safe replacements. The em dash (U+2014) is left alone; pandoc handles it.
UNICODE_MAP = {
    "−": "-",              # minus sign
    "–": "--",             # en dash -> LaTeX en dash
    "→": r"$\rightarrow$",
    "φ": r"$\varphi$",
    "τ": r"$\tau$",
    "₁": r"$_1$",
    "Δ": r"$\Delta$",
}

BREAK_TT_LUA = r"""
-- Escape LaTeX specials in code spans ourselves, then insert explicit break
-- opportunities after path separators. \allowbreak adds no visible hyphen.
local esc = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}",
  ["$"] = "\\$", ["&"] = "\\&", ["#"] = "\\#", ["%"] = "\\%",
  ["_"] = "\\_", ["^"] = "\\^{}", ["~"] = "\\textasciitilde{}",
  ["<"] = "\\textless{}", [">"] = "\\textgreater{}",
}
function Code(el)
  local out = {}
  for ch in el.text:gmatch(utf8 and utf8.charpattern or ".") do
    out[#out + 1] = esc[ch] or ch
    if ch == "/" or ch == "_" or ch == "." or ch == "-" or ch == "," then
      out[#out + 1] = "\\allowbreak{}"
    end
  end
  return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
end
"""

TABLE_WIDTHS_LUA = r"""
-- Give tables with prose cells relative column widths so LaTeX wraps them.
local stringify = pandoc.utils.stringify
local function col_maxlens(tbl)
  local n, m = #tbl.colspecs, {}
  for i = 1, n do m[i] = 0 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= n then
          local len = #stringify(cell.contents)
          if len > m[i] then m[i] = len end
        end
      end
    end
  end
  for _, h in ipairs(tbl.head.rows) do scan({ h }) end
  for _, body in ipairs(tbl.bodies) do scan(body.body) end
  return m
end
function Table(tbl)
  local m = col_maxlens(tbl)
  local longest, total = 0, 0
  for _, v in ipairs(m) do
    if v > longest then longest = v end
    total = total + v
  end
  if longest <= 40 or total == 0 then return nil end
  for i = 1, #tbl.colspecs do
    tbl.colspecs[i][2] = 0.97 * m[i] / total
  end
  return tbl
end
"""

PREAMBLE = r"""
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{etoolbox}
\usepackage{xurl}
\usepackage{fvextra}
\usepackage{microtype}
\usepackage{titlesec}

% Result tables are wide; shrink rather than run into the margin.
\AtBeginEnvironment{longtable}{\footnotesize}
\AtBeginEnvironment{tabular}{\footnotesize}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}

\fvset{fontsize=\small,breaklines=true,breakanywhere=true}
\setlength{\emergencystretch}{3em}
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{0.7em}{}
"""


def preprocess(text: str) -> str:
    """Drop the H1 (the title comes from \\maketitle) and map Unicode to LaTeX."""
    first, rest = text.split("\n", 1)
    if not first.startswith("# "):
        raise SystemExit(f"expected an H1 on line 1, got: {first[:60]!r}")
    out = rest.lstrip("\n")
    for src, dst in UNICODE_MAP.items():
        out = out.replace(src, dst)
    leftover = sorted({c for c in out if ord(c) > 126 and c != "—"})
    if leftover:
        raise SystemExit(
            "unmapped non-ASCII characters would break pdflatex: "
            + ", ".join(f"U+{ord(c):04X} {c!r}" for c in leftover)
            + "\nAdd them to UNICODE_MAP."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pandoc", default=shutil.which("pandoc"), help="Path to the pandoc binary.")
    parser.add_argument("-o", "--output", default=str(PROJECT_ROOT / "technical_paper.pdf"))
    parser.add_argument("--date", default="", help="Date on the title page; defaults to the source mtime.")
    parser.add_argument("--keep-tex", action="store_true", help="Copy the generated .tex next to the PDF.")
    args = parser.parse_args()

    if not args.pandoc:
        raise SystemExit("pandoc not found; pass --pandoc PATH (see module docstring)")
    if not shutil.which("pdflatex"):
        raise SystemExit("pdflatex not found; install texlive")

    import datetime

    date = args.date or datetime.date.fromtimestamp(SOURCE.stat().st_mtime).strftime("%B %-d, %Y")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "breaktt.lua").write_text(BREAK_TT_LUA)
        (tmp_path / "tablewidths.lua").write_text(TABLE_WIDTHS_LUA)
        (tmp_path / "preamble.tex").write_text(PREAMBLE)
        (tmp_path / "paper.md").write_text(preprocess(SOURCE.read_text()))

        subprocess.run(
            [
                args.pandoc, str(tmp_path / "paper.md"), "-o", str(tmp_path / "paper.tex"),
                "--from=gfm+tex_math_dollars+smart", "--standalone", "--toc", "--toc-depth=2",
                f"--lua-filter={tmp_path / 'breaktt.lua'}",
                f"--lua-filter={tmp_path / 'tablewidths.lua'}",
                "-V", "geometry:a4paper,margin=2cm", "-V", "fontsize=10pt",
                "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
                "-V", "toccolor=black",
                "-V", f"title={TITLE}", "-V", f"author={AUTHOR}", "-V", f"date={date}",
                "-H", str(tmp_path / "preamble.tex"),
            ],
            check=True,
        )

        # Twice, so the table of contents resolves.
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )

        pdf = tmp_path / "paper.pdf"
        if not pdf.exists():
            log = (tmp_path / "paper.log").read_text(errors="replace")[-3000:]
            raise SystemExit("pdflatex produced no PDF. Tail of log:\n" + log)

        log = (tmp_path / "paper.log").read_text(errors="replace")
        overfull = log.count("Overfull \\hbox")
        pages = log.split("Output written on paper.pdf (")[-1].split(" pages")[0]

        out = Path(args.output)
        shutil.copy(pdf, out)
        if args.keep_tex:
            shutil.copy(tmp_path / "paper.tex", out.with_suffix(".tex"))

    print(f"wrote {out}  ({pages} pages, {overfull} overfull hbox warnings)")


if __name__ == "__main__":
    main()
