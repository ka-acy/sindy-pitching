#!/usr/bin/env python3
"""Build docs/technical_paper.md into an IEEE-format PDF via pandoc + pdflatex.

Requires `pdflatex` (texlive, including the ieeetran package) on PATH. If `pandoc` is
missing, install a bundled binary with

    pip install --target <dir> pypandoc-binary

and pass --pandoc <dir>/pypandoc/files/pandoc.

Two styles are available:

  --style ieee     IEEEtran, two-column, Roman-numeral sections, abstract block,
                   Index Terms, no table of contents. This is the IEEE paper format.
  --style article  single-column article class with a table of contents. Easier to read
                   on screen and much friendlier to this paper's wide tables and long
                   shell commands, but it is not IEEE format.

Things a bare `pandoc -o out.pdf` gets wrong here, all handled below:

1. Only pdflatex is assumed available, not xelatex, so the paper's Unicode (minus sign,
   en dash, arrow, Greek, subscript) is mapped to LaTeX equivalents first. Unmapped
   non-ASCII is a hard error rather than a silently mangled PDF.
2. Long file paths in inline code cannot hyphenate and overflow the text block -- by up to
   329pt in single column, and far worse in a two-column measure. A Lua filter escapes code
   spans itself and inserts \\allowbreak after path separators. No visible hyphen, which
   matters because a hyphen is indistinguishable from a real character in a path or flag.
3. Pandoc emits `longtable`, which does not work in two-column mode at all. For the IEEE
   style every table is converted to a `table`/`table*` float with `tabular`; tables with
   four or more columns or long cells span both columns via `table*`.
4. IEEEtran numbers sections itself as Roman numerals, so the manual numbers in the
   Markdown headings are stripped and the in-text cross-references are rewritten from
   "Section 6.5" to IEEE's "Section VI-E" form.

Usage:
    python scripts/build_paper_pdf.py [--style ieee|article] [--pandoc PATH] [-o OUT]
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "technical_paper.md"

TITLE = "Sparse Dynamical Modeling of Baseball Pitching Kinematics from Vision-Derived Pose"
AUTHOR = "Alex Yu"
KEYWORDS = (
    "sparse identification of nonlinear dynamics, SINDy, human pose estimation, "
    "sports biomechanics, model evaluation, baseline controls, time-series forecasting"
)

# Math symbols are emitted via \ensuremath rather than $...$ on purpose. Pandoc will not
# parse a closing "$" that is immediately followed by a digit, so "$\pm$6773" was escaped to
# literal "\$\pm\$6773", putting \pm into text mode and producing a 207pt runaway box.
# \ensuremath works in either mode and is immune to what follows it.
UNICODE_MAP = {
    "−": "-",              # minus sign
    "–": "--",             # en dash
    "→": r"\ensuremath{\rightarrow}",
    "φ": r"\ensuremath{\varphi}",
    "τ": r"\ensuremath{\tau}",
    "₁": r"\ensuremath{_1}",
    "Δ": r"\ensuremath{\Delta}",
    "±": r"\ensuremath{\pm}",
}

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
         "XIII", "XIV", "XV", "XVI"]

BREAK_TT_LUA = r"""
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

PREAMBLE_COMMON = r"""
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{array}
\usepackage{etoolbox}
\usepackage{fvextra}
\fvset{breaklines=true,breakanywhere=true}
\setlength{\emergencystretch}{3em}
"""

PREAMBLE_IEEE = PREAMBLE_COMMON + r"""
\usepackage{url}
\AtBeginEnvironment{tabular}{\scriptsize}
% IEEEtran sets \labelsep etc. for its own lists; keep pandoc's tightlist harmless.
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

% Long shell commands and identifiers do not fit a two-column measure. \fvset alone does
% not reach these: pandoc fixes options when it defines the environments, so redefine both
% after the fact with explicit breaking. breakanywhere is required because the long tokens
% are single unbroken paths and flags with no spaces to break at.
\usepackage{fancyvrb}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{%
  breaklines=true,breakanywhere=true,fontsize=\scriptsize}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{%
  commandchars=\\\{\},breaklines=true,breakanywhere=true,fontsize=\scriptsize}
"""

PREAMBLE_ARTICLE = PREAMBLE_COMMON + r"""
\usepackage{longtable}
\usepackage{xurl}
\usepackage{microtype}
\usepackage{titlesec}
\AtBeginEnvironment{longtable}{\footnotesize}
\AtBeginEnvironment{tabular}{\footnotesize}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\fvset{fontsize=\small}
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{0.7em}{}
"""


def build_section_map(text: str) -> dict[str, str]:
    """Map Arabic heading numbers in the source to the Roman/letter labels IEEEtran emits."""
    mapping: dict[str, str] = {}
    section = 0
    for line in text.splitlines():
        m2 = re.match(r"^## (\d+)\. ", line)
        if m2:
            section = int(m2.group(1))
            mapping[m2.group(1)] = ROMAN[section]
            sub = 0
            continue
        m3 = re.match(r"^### (\d+)\.(\d+) ", line)
        if m3:
            sub = int(m3.group(2))
            mapping[f"{m3.group(1)}.{m3.group(2)}"] = f"{ROMAN[int(m3.group(1))]}-{chr(64 + sub)}"
    return mapping


def to_ieee_structure(text: str) -> tuple[str, str]:
    """Strip manual heading numbers, Romanize cross-references, split out the abstract."""
    secmap = build_section_map(text)

    # Longest keys first so "6.5" is replaced before "6".
    for key in sorted(secmap, key=len, reverse=True):
        val = secmap[key]
        text = re.sub(rf"\bSection {re.escape(key)}\b", f"Section {val}", text)
        text = re.sub(rf"\bSections {re.escape(key)}\b", f"Sections {val}", text)
    # "Sections VI-D and 6.7" -> fix the trailing member of a pair
    for key in sorted(secmap, key=len, reverse=True):
        text = re.sub(rf"(Sections [IVX]+(?:-[A-Z])? and ){re.escape(key)}\b",
                      rf"\g<1>{secmap[key]}", text)

    # IEEEtran numbers sections itself.
    text = re.sub(r"^## \d+\. ", "## ", text, flags=re.M)
    text = re.sub(r"^### \d+\.\d+ ", "### ", text, flags=re.M)

    # Pull the abstract out of the body; IEEEtran wants it in an abstract environment.
    m = re.search(r"^## Abstract\n(.*?)(?=^## )", text, flags=re.M | re.S)
    if not m:
        raise SystemExit("could not locate the '## Abstract' section")
    abstract = " ".join(p.strip() for p in m.group(1).strip().split("\n\n"))
    text = text[: m.start()] + text[m.end():]
    return text, abstract


def _match_brace(s: str, open_idx: int) -> int:
    """Index of the brace matching the one at open_idx. Column specs contain @{} and p{},
    so a non-greedy regex captures only '@{' -- the depth has to be tracked."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise SystemExit("unbalanced braces in a longtable column spec")


def longtable_to_float(tex: str) -> str:
    """Convert pandoc longtables to IEEE-compatible table/table* floats.

    longtable does not work in two-column mode. Tables with four or more columns, or with
    proportional p{} columns from the width filter, span both columns via table*.
    """
    marker = "\\begin{longtable}[]{"
    out, pos, count, wide_count = [], 0, 0, 0
    while True:
        i = tex.find(marker, pos)
        if i == -1:
            out.append(tex[pos:])
            break
        # optional pandoc wrapper immediately before the environment
        start = i
        wrapper = tex.rfind("{\\def\\LTcaptype{none}", max(0, i - 200), i)
        if wrapper != -1 and tex[wrapper:i].strip().startswith("{\\def"):
            start = wrapper
        out.append(tex[pos:start])

        brace = i + len(marker) - 1
        close = _match_brace(tex, brace)
        spec = tex[brace + 1:close]

        end = tex.find("\\end{longtable}", close)
        if end == -1:
            raise SystemExit("unterminated longtable")
        body = tex[close + 1:end]
        after = end + len("\\end{longtable}")
        if after < len(tex) and tex[after] == "}":
            after += 1
        pos = after

        # A captioned longtable carries \caption plus a duplicated \endfirsthead block.
        # IEEE puts the caption above the table, so lift it out of the tabular entirely.
        caption = ""
        cidx = body.find("\\caption{")
        if cidx != -1:
            cstart = cidx + len("\\caption")
            cend = _match_brace(body, cstart)
            caption = body[cstart + 1:cend]
            body = body[:cidx] + body[cend + 1:]
            body = body.replace("\\tabularnewline", "", 1)
            # drop the repeated header that longtable needs and tabular does not
            fh = body.find("\\endfirsthead")
            if fh != -1:
                eh = body.find("\\endhead", fh)
                if eh != -1:
                    body = body[:fh] + body[eh + len("\\endhead"):]

        body = body.replace("\\endhead", "")
        body = re.sub(r"\\bottomrule\\noalign\{\}\s*\\endlastfoot", "", body)
        body = body.replace("\\endfirsthead", "").strip()
        if not body.endswith("\\bottomrule\\noalign{}"):
            body += "\n\\bottomrule\\noalign{}"

        ncols = len(re.findall(r"p\{|>\{", spec)) or len(re.findall(r"[lcr]", re.sub(r"@\{[^}]*\}", "", spec)))
        proportional = "p{" in spec
        # Column count alone misjudges narrow tables holding prose: a 3-column table of
        # sentences still will not fit a ~3.5in IEEE column. Estimate the natural width
        # from the widest row, with LaTeX markup stripped, and compare against roughly
        # what fits at \scriptsize in one column.
        plain = re.sub(r"\\[a-zA-Z]+\s*|[{}]|\\allowbreak", "", body)
        rows = [r.strip() for r in plain.split(chr(92) * 2) if r.strip()]
        widest = max((len(r) for r in rows), default=0)
        wide = ncols >= 4 or proportional or widest > 55
        env = "table*" if wide else "table"
        count += 1
        wide_count += 1 if wide else 0
        cap = f"\\caption{{{caption}}}\n" if caption else ""
        out.append(
            f"\\begin{{{env}}}[!t]\n\\centering\n{cap}"
            f"\\begin{{tabular}}{{{spec}}}\n{body}\n\\end{{tabular}}\n"
            f"\\end{{{env}}}"
        )
    print(f"  converted {count} longtable(s): {wide_count} full-width (table*), {count - wide_count} single-column")
    return "".join(out)


def preprocess(text: str, style: str) -> tuple[str, str]:
    first, rest = text.split("\n", 1)
    if not first.startswith("# "):
        raise SystemExit(f"expected an H1 on line 1, got: {first[:60]!r}")
    out = rest.lstrip("\n")

    abstract = ""
    if style == "ieee":
        out, abstract = to_ieee_structure(out)

    for src, dst in UNICODE_MAP.items():
        out = out.replace(src, dst)
        abstract = abstract.replace(src, dst)

    leftover = sorted({c for c in out + abstract if ord(c) > 126 and c != "—"})
    if leftover:
        raise SystemExit(
            "unmapped non-ASCII would break pdflatex: "
            + ", ".join(f"U+{ord(c):04X} {c!r}" for c in leftover)
            + "\nAdd them to UNICODE_MAP."
        )
    return out, abstract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", choices=("ieee", "article"), default="ieee")
    parser.add_argument("--pandoc", default=shutil.which("pandoc"))
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--keep-tex", action="store_true")
    args = parser.parse_args()

    if not args.pandoc:
        raise SystemExit("pandoc not found; pass --pandoc PATH (see module docstring)")
    if not shutil.which("pdflatex"):
        raise SystemExit("pdflatex not found; install texlive")

    out_path = Path(args.output) if args.output else PROJECT_ROOT / (
        "technical_paper_ieee.pdf" if args.style == "ieee" else "technical_paper.pdf"
    )
    date = args.date or datetime.date.fromtimestamp(SOURCE.stat().st_mtime).strftime("%B %d, %Y")
    body, abstract = preprocess(SOURCE.read_text(), args.style)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "breaktt.lua").write_text(BREAK_TT_LUA)
        (tmp_path / "tablewidths.lua").write_text(TABLE_WIDTHS_LUA)
        (tmp_path / "preamble.tex").write_text(
            PREAMBLE_IEEE if args.style == "ieee" else PREAMBLE_ARTICLE
        )
        if args.style == "ieee":
            def yaml_block(text: str) -> str:
                return "\n".join("  " + line for line in text.split("\n"))
            front = (
                "---\n"
                "abstract: |\n" + yaml_block(abstract) + "\n"
                "keywords: |\n" + yaml_block(KEYWORDS) + "\n"
                "---\n\n"
            )
            (tmp_path / "paper.md").write_text(front + body)
        else:
            (tmp_path / "paper.md").write_text(body)

        cmd = [
            args.pandoc, str(tmp_path / "paper.md"), "-o", str(tmp_path / "paper.tex"),
            # gfm has no table-caption syntax; pandoc markdown supports ": caption" after a table.
            # autolink_bare_uris keeps the bare DOI/arXiv URLs in the references clickable.
            "--from=markdown+tex_math_dollars+smart+autolink_bare_uris", "--standalone",
            # Syntax highlighting wraps every token in a macro, and fvextra cannot break
            # inside macro arguments -- long shell commands then overflow the column.
            # IEEE papers are monochrome regardless.
            "--syntax-highlighting=none",
            # The H1 is dropped, so '##' would still map to \subsection and lose IEEEtran's
            # Roman section numbering. Shift so '##' -> \section, '###' -> \subsection.
            "--shift-heading-level-by=-1",
            f"--lua-filter={tmp_path / 'breaktt.lua'}",
            "-V", f"title={TITLE}", "-V", f"author={AUTHOR}", "-V", f"date={date}",
            f"--lua-filter={tmp_path / 'tablewidths.lua'}",
            "-H", str(tmp_path / "preamble.tex"),
        ]
        if args.style == "ieee":
            cmd += [
                # Pandoc sets secnumdepth to 0 unless asked, which suppresses IEEEtran's
                # Roman section numbers. The manual numbers were stripped above, so this
                # is what restores "I. INTRODUCTION" / "VI-E".
                "--number-sections",
                "-V", "documentclass=IEEEtran",
                "-V", "classoption=journal",
                "-V", "classoption=10pt",
            ]
        else:
            cmd += [
                "--toc", "--toc-depth=2",
                "-V", "geometry:a4paper,margin=2cm", "-V", "fontsize=10pt",
                "-V", "colorlinks=true", "-V", "linkcolor=blue",
                "-V", "urlcolor=blue", "-V", "toccolor=black",
            ]
        subprocess.run(cmd, check=True)

        tex = (tmp_path / "paper.tex").read_text()
        if args.style == "ieee":
            tex = longtable_to_float(tex)
            # IEEEtran wants Index Terms right after the abstract.
            tex = tex.replace(
                "\\end{abstract}",
                "\\end{abstract}\n\n\\begin{IEEEkeywords}\n" + KEYWORDS + "\n\\end{IEEEkeywords}",
                1,
            )
        (tmp_path / "paper.tex").write_text(tex)

        for _ in range(3):  # IEEEtran + floats need extra passes to settle
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )

        pdf = tmp_path / "paper.pdf"
        log = (tmp_path / "paper.log").read_text(errors="replace")
        if not pdf.exists():
            raise SystemExit("pdflatex produced no PDF. Tail of log:\n" + log[-4000:])

        overfull = log.count("Overfull \\hbox")
        worst = max((float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)), default=0.0)
        pages = log.split("Output written on paper.pdf (")[-1].split(" pages")[0]

        shutil.copy(pdf, out_path)
        if args.keep_tex:
            shutil.copy(tmp_path / "paper.tex", out_path.with_suffix(".tex"))

    print(f"wrote {out_path}")
    print(f"  style={args.style}  pages={pages}  overfull={overfull}  worst={worst:.1f}pt")


if __name__ == "__main__":
    main()
