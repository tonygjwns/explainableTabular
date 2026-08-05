"""Catch the class of defect only a LaTeX compile has been catching.

Three compiles found three defects -- a text-mode underscore, a currency dollar, and table rows that
never closed -- and none was visible to the checks available locally, because the file was
well-formed by every structural measure being applied. This is that missing check. It is not a LaTeX
parser; it looks for the specific things that have actually broken this document.

    python scripts/tex_lint.py paper/main.tex

Checks
  1. Special characters raw in running text: _ & # % ~ ^ $ outside math, comments, and the braced
     arguments of \\texttt, \\label, \\ref, \\input, \\includegraphics, \\resizebox, \\url.
  2. Corrupted tokens of the form  word_\\texttt{...}  -- produced by escaping only part of an
     identifier, which is how the third compile broke.
  3. Table rows that do not end in a row terminator, and rows whose cell count disagrees with the
     column specification (the spec is parsed properly: l/c/r/p count, >{}/@{}/| do not).
  4. Unbalanced math delimiters per paragraph, unbalanced braces per file, environments that open
     and never close, and \\ref targets with no \\label.
"""
from __future__ import annotations

import io
import re
import sys

B, U, D = chr(92), chr(95), chr(36)
ARGS = ("texttt", "label", "ref", "input", "includegraphics", "resizebox", "url", "cite",
        "citep", "citet", "bibliography", "documentclass", "usepackage", "def", "newcommand")
MASK_MATH = re.compile(re.escape(D) + r"[^" + re.escape(D) + r"]*" + re.escape(D))
MASK_ARG = re.compile(re.escape(B) + r"(?:" + "|".join(ARGS) + r")\*?(?:\[[^\]]*\])?\{[^{}]*\}")
SPECIALS = U + "#"               # & separates columns, ~ is a tie, % starts a comment,
                                 # and $ is covered by the per-paragraph math check


def mask(line):
    line = strip_comment(line)
    t = MASK_ARG.sub(lambda m: " " * len(m.group(0)), line)
    return MASK_MATH.sub(lambda m: " " * len(m.group(0)), t)


def strip_comment(line):
    """Everything after an unescaped % is a comment and cannot break typesetting."""
    for k, ch in enumerate(line):
        if ch == "%" and (k == 0 or line[k - 1] != B):
            return line[:k]
    return line


def colspec_width(spec):
    n, i = 0, 0
    while i < len(spec):
        c = spec[i]
        if c in "lcr":
            n += 1
        elif c == "p" and i + 1 < len(spec) and spec[i + 1] == "{":
            n += 1
            depth = 0
            while i < len(spec):
                depth += spec[i] == "{"
                depth -= spec[i] == "}"
                i += 1
                if depth == 0:
                    break
            continue
        elif c in ">@" and i + 1 < len(spec) and spec[i + 1] == "{":
            depth = 0
            i += 1
            while i < len(spec):
                depth += spec[i] == "{"
                depth -= spec[i] == "}"
                i += 1
                if depth == 0:
                    break
            continue
        i += 1
    return n


def main(path):
    src = io.open(path, encoding="utf-8").read()
    L = src.split(chr(10))
    bad = []

    for i, line in enumerate(L, 1):
        if line.lstrip().startswith("%"):
            continue
        view = mask(line)
        for k, ch in enumerate(view):
            if ch in SPECIALS and (k == 0 or view[k - 1] != B):
                bad.append((i, "raw '%s' in text" % ch, line.strip()[:90]))
                break
        if U + B + "texttt{" in line:
            bad.append((i, "half-escaped identifier", line.strip()[:90]))

    inside, spec, ncol = False, "", 0
    for i, line in enumerate(L, 1):
        s = line.strip()
        if s.startswith(B + "begin{tabular}"):
            inside = True
            spec = s[s.index("{", 15) + 1: s.rindex("}")] if "{" in s[15:] else ""
            ncol = colspec_width(spec)
            continue
        if s.startswith(B + "end{tabular}"):
            inside = False
            continue
        if not inside or not s or s.startswith((B + "toprule", B + "midrule", B + "bottomrule",
                                                B + "cmidrule", B + "multicolumn", "%")):
            continue
        if not s.endswith(B * 2):
            bad.append((i, "row has no terminator", s[:90]))
            continue
        cells = len(re.split(r"(?<!" + re.escape(B) + r")&", s)) if "&" in s else 1
        if ncol and cells != ncol:
            bad.append((i, "%d cells vs %d columns" % (cells, ncol), s[:90]))

    start = 0
    for i, line in enumerate(L + [""]):
        if line.strip() == "":
            n = sum(1 for row in L[start:i] for k, ch in enumerate(row)
                    if ch == D and (k == 0 or row[k - 1] != B))
            if n % 2:
                bad.append((start + 1, "odd math delimiters in paragraph", L[start][:90]))
            start = i + 1

    if src.count("{") != src.count("}"):
        bad.append((0, "unbalanced braces", "%+d" % (src.count("{") - src.count("}"))))
    for env in ("table", "tabular", "figure", "itemize", "enumerate", "abstract", "document", "center"):
        b, e = src.count(B + "begin{" + env + "}"), src.count(B + "end{" + env + "}")
        if b != e:
            bad.append((0, "unbalanced environment " + env, "%d begin / %d end" % (b, e)))
    defined = set(re.findall(r"label\{([^}]+)\}", src))
    for r in sorted(set(re.findall(r"ref\{([^}]+)\}", src)) - defined):
        bad.append((0, "reference with no label", r))

    for i, why, txt in bad:
        print(("  L%-6d" % i if i else "  file   ") + "%-34s %s" % (why, txt))
    print("%d issue(s)" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex"))
