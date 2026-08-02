"""Check the manuscript's numeric claims against the committed artifacts.

Motivation, concretely: a sentence reading "three of four unidentifiable cells fail the
learnability gate" survived a revision round that corrected the same claim in three other
sections, and was only caught by reading the artifacts back. Prose drifts from data silently;
this makes the drift loud.

What it does
  1. Reads every committed run artifact under results/ and prereg_results/ and builds a lookup of
     (dataset, model, seed_base, span, injection family/carrier) -> row.
  2. Extracts every number in the three manuscript files that looks like an instrument reading
     (staleness / denoised / D / recovery / delta / gate), with its surrounding line.
  3. Reports each as MATCH (an artifact carries that value), NEAR (within rounding of one), or
     UNMATCHED (no artifact within tolerance).
  4. Cross-checks the three files against each other on the numbers they share.

UNMATCHED is not automatically an error -- the paper legitimately quotes derived quantities
(ratios, bias-corrected gaps, aggregates) and numbers from lenses outside these artifacts. The
output is a worklist, not a verdict. Read it, do not obey it.

    python scripts/audit_paper_numbers.py
    python scripts/audit_paper_numbers.py --tol 0.0005 --show matched
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["PAPER_DRAFT_V4.md", "PAPER_DRAFT_V4_KO.md", "paper/main.tex"]
FIELDS = ("staleness_harm", "denoised_staleness", "D_strip", "D_full", "injected_staleness",
          "delta_staleness", "noise_ratio", "recency_gain", "decay", "injection_learn_score")

NUM = re.compile(r"[+−-]?\d+\.\d{3,5}")          # 3-5 dp: instrument readings, not years


def load_artifacts():
    vals = {}          # rounded value -> list of "dataset.field (context)"
    n_files = 0
    for pat in ("results/**/*.json", "prereg_results/**/*.json",
                "audit_artifacts_2026-07-04/**/*.json", "audit_repair_2026-07-18/*.json",
                "*.json"):
        for f in glob.glob(str(ROOT / pat), recursive=True):
            if "_raw" in f:
                continue
            try:
                blob = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            rows = blob.get("rows")
            if not isinstance(rows, list):
                continue
            n_files += 1
            meta = blob.get("meta", {})
            a = " ".join(meta.get("argv", []))
            tag = []
            for m in ("rf", "linear", "knn", "mlp"):
                if f"--model {m}" in a:
                    tag.append(m)
            if "--seed-base 100" in a:
                tag.append("sb100")
            if "--tabred-span full" in a:
                tag.append("full")
            for k in ("--inj-family", "--inj-cols", "--metric", "--mi-k"):
                if k in a:
                    tag.append(f"{k[2:]}={a.split(k)[1].split()[0]}")
            ctx = ",".join(tag) or "default"
            for r in rows:
                if not isinstance(r, dict):
                    continue
                ds = r.get("dataset", "?")
                for fld in FIELDS:
                    v = r.get(fld)
                    if isinstance(v, (int, float)):
                        vals.setdefault(round(float(v), 4), []).append(f"{ds}.{fld} [{ctx}]")
                for fld in ("staleness_harm_ci", "denoised_staleness_ci", "injected_staleness_ci"):
                    ci = r.get(fld)
                    if isinstance(ci, list):
                        for v in ci:
                            if isinstance(v, (int, float)):
                                vals.setdefault(round(float(v), 4), []).append(f"{ds}.{fld} [{ctx}]")
    return vals, n_files


def doc_numbers(path):
    out = []
    for i, line in enumerate(open(ROOT / path, encoding="utf-8"), 1):
        if line.lstrip().startswith(">"):            # revision-note blockquotes
            continue
        if "doi" in line.lower() or "arxiv" in line.lower():   # 10.1007/... is not a reading
            continue
        line = re.sub(r"(\d)--(\d)", r" to ", line)   # LaTeX en-dash range, not a sign
        for m in NUM.finditer(line):
            raw = m.group().replace("−", "-")
            try:
                v = float(raw)
            except ValueError:
                continue
            if abs(v) > 100:                          # sample sizes, years
                continue
            out.append((i, v, line.strip()[:118]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0005)
    ap.add_argument("--show", default="unmatched", choices=["unmatched", "matched", "all"])
    args = ap.parse_args()

    vals, n_files = load_artifacts()
    keys = sorted(vals)
    print(f"artifacts: {n_files} run files, {len(keys)} distinct instrument values\n")

    per_doc = {}
    for d in DOCS:
        nums = doc_numbers(d)
        per_doc[d] = nums
        exact = near = unmatched = 0
        misses = []
        for ln, v, ctx in nums:
            r = round(v, 4)
            if r in vals:
                exact += 1
                if args.show in ("matched", "all"):
                    print(f"  MATCH  {d}:{ln}  {v:+.4f}  <- {vals[r][0]}")
                continue
            cand = [k for k in keys if abs(k - v) <= args.tol]
            if cand:
                near += 1
                if args.show in ("matched", "all"):
                    print(f"  NEAR   {d}:{ln}  {v:+.4f} ~ {cand[0]:+.4f}  <- {vals[cand[0]][0]}")
            else:
                unmatched += 1
                misses.append((ln, v, ctx))
        print(f"{d}:  {len(nums)} numbers | exact {exact} | near {near} | unmatched {unmatched}")
        if args.show in ("unmatched", "all"):
            for ln, v, ctx in misses:
                print(f"    ? {d}:{ln}  {v:+.4f}   {ctx}")
        print()

    print("=" * 78)
    print("cross-file: numbers present in one manuscript file but not the others")
    sets = {d: {round(v, 4) for _, v, _ in per_doc[d]} for d in DOCS}
    for d in DOCS:
        others = set().union(*[sets[o] for o in DOCS if o != d])
        only = sorted(sets[d] - others)
        print(f"  {d}: {len(only)} unique" + (f"  {only[:12]}" if only else ""))
    print("\nNOTE: UNMATCHED covers derived quantities (ratios, bias-corrected gaps, aggregates)")
    print("and readings from lenses whose artifacts live elsewhere. This is a worklist to read,")
    print("not a verdict to obey.")


if __name__ == "__main__":
    main()
