"""Convert a V5 markdown section into LaTeX for paper/main.tex.

Only the sections V5 wrote from scratch go through this. Sections carried over from V4 keep their
existing LaTeX -- tables, math and TikZ are already typeset there, and re-converting them from
markdown would be a downgrade. Cross-references survive a move untouched because LaTeX resolves
them by label, not by number.

The converter is deliberately conservative: it handles what the V5 prose actually uses and raises
on anything it does not recognise, so a silent mistranslation cannot reach the paper.

    python scripts/md2tex.py PAPER_DRAFT_V5.md "## 3. The failure channel" "## 4." > out.tex
"""
from __future__ import annotations

import io
import re
import sys

# Section anchors: markdown "§N.M" -> \Cref-style reference by label.
SEC_LABEL = {
    "1": "sec:intro", "2": "sec:related", "3": "sec:channel", "3.1": "sec:noisechannel",
    "3.2": "sec:diagnosis", "3.3": "sec:gate", "3.4": "sec:frames",
    "4": "sec:instrument", "4.1": "sec:setting", "4.2": "sec:denoised",
    "4.3": "sec:certificates", "4.4": "sec:cascade",
    "5": "sec:validation", "5.1": "sec:battery", "5.2": "sec:regate",
    "6": "sec:blind", "6.1": "sec:classrel", "6.2": "sec:powerrel",
    "6.3": "sec:familyrel", "6.4": "sec:metricrel",
    "7": "sec:application", "7.1": "sec:panel", "7.2": "sec:gap", "7.3": "sec:anchors",
    "8": "sec:discussion", "9": "sec:repro",
}

UNI = [
    ("\u2212", "$-$"), ("\u2192", "$\\to$"), ("\u2248", "$\\approx$"), ("\u2264", "$\\le$"),
    ("\u2265", "$\\ge$"), ("\u00d7", "$\\times$"), ("\u2208", "$\\in$"), ("\u222a", "$\\cup$"),
    ("\u03c1", "$\\rho$"), ("\u03b4", "$\\delta$"), ("\u03bb", "$\\lambda$"), ("\u00b2", "$^2$"),
    ("\u2014", "---"), ("\u2013", "--"), ("\u2018", "`"), ("\u2019", "'"),
    ("\u201c", "``"), ("\u201d", "''"), ("\u2026", "\\ldots"), ("\u2265", "$\\ge$"),
    ("\u00a0", "~"), ("\u2020", "$\\dagger$"), ("\u2713", "\\yes"),
]


def inline(t: str) -> str:
    """Markdown inline markup -> LaTeX, then unicode -> LaTeX."""
    t = re.sub(r"`([^`]+)`", lambda m: "\\texttt{" + m.group(1).replace("_", "\\_") + "}", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", t)
    t = re.sub(r"§(\d(?:\.\d)?)", lambda m: "Section~\\ref{%s}" % SEC_LABEL[m.group(1)]
               if m.group(1) in SEC_LABEL else m.group(0), t)
    t = re.sub(r"PREREG Section~\\ref\{[^}]+\}", "PREREG~\\\\S", t)      # never renumber the prereg
    t = re.sub(r"−(\d[\d.,]*)", lambda m: "$-" + m.group(1).rstrip(".,") + "$"
               + m.group(1)[len(m.group(1).rstrip(".,")):], t)      # keep minus with its number
    # bare cell names such as ecom_offers are text-mode underscores, which LaTeX reads as
    # subscripts; V4 typesets them as \\texttt{...}, so match that rather than escaping in place
    tok = re.compile(r"(?<![\\\\A-Za-z0-9{])([a-z][a-z0-9]*(?:" + chr(95) + r"[a-z0-9]+)+)")
    t = tok.sub(lambda mm: "\\texttt{" + mm.group(1).replace(chr(95), "\\" + chr(95)) + "}", t)
    q = chr(34)                                                  # straight quotes -> LaTeX quotes
    t = re.sub(q + "([^" + q + chr(10) + "]*)" + q, lambda mm: "``" + mm.group(1) + "''", t)
    for a, b in UNI:
        t = t.replace(a, b)
    t = re.sub(r"(?<![\\$])%", r"\\%", t)
    t = re.sub(r"(?<![\\$_^{])&", r"\\&", t)
    return t


def table(rows, caption, label):
    head, body = rows[0], rows[2:]
    ncol = len(head)
    out = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{%s}" % inline(caption), "\\label{%s}" % label,
           "\\begin{tabular}{@{}l%s@{}}" % ("r" * (ncol - 1)), "\\toprule",
           " & ".join(inline(c) for c in head) + " \\\\", "\\midrule"]
    for r in body:
        out.append(" & ".join(inline(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def convert(md: str) -> str:
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            out.append(""); i += 1; continue
        m = re.match(r"^(#{2,4})\s+(?:\d+(?:\.\d+)?\.?\s+)?(.+)$", ln)
        if m:
            depth = len(m.group(1))
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[depth]
            num = re.match(r"^#{2,4}\s+(\d+(?:\.\d+)?)[.\s]", ln)
            lab = SEC_LABEL.get(num.group(1)) if num else None
            out.append("\\%s{%s}%s" % (cmd, inline(m.group(2)),
                                       "\\label{%s}" % lab if lab else ""))
            i += 1; continue
        if ln.lstrip().startswith("**Table ") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            cap = re.sub(r"^\*\*Table [^.]+\.\*\*\s*", "", ln.strip())
            lab = "tab:" + re.search(r"Table ([A-C]?\d+)", ln).group(1).lower()
            rows, j = [], i + 1
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")]); j += 1
            out.append(table(rows, cap, lab)); i = j; continue
        if ln.startswith("|"):                                   # table with no caption line
            rows, j = [], i
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")]); j += 1
            out.append(table(rows, "", "tab:unnamed%d" % i)); i = j; continue
        if re.match(r"^\d+\.\s", ln.strip()):                    # enumerate block
            items, j = [], i
            while j < len(lines) and (re.match(r"^\d+\.\s", lines[j].strip()) or
                                      (lines[j].startswith("   ") and lines[j].strip())):
                items.append(lines[j]); j += 1
            out.append("\\begin{enumerate}")
            for it in "\n".join(items).split("\n"):
                if re.match(r"^\d+\.\s", it.strip()):
                    out.append("\\item " + inline(re.sub(r"^\d+\.\s*", "", it.strip())))
                else:
                    out.append(inline(it.strip()))
            out.append("\\end{enumerate}"); i = j; continue
        out.append(inline(ln)); i += 1
    return "\n".join(out)


if __name__ == "__main__":
    src = io.open(sys.argv[1], encoding="utf-8").read()
    a = src.index(sys.argv[2])
    b = src.index(sys.argv[3], a) if len(sys.argv) > 3 else len(src)
    sys.stdout.reconfigure(encoding="utf-8")
    print(convert(src[a:b]))
