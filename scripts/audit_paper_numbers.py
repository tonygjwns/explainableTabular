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

A value match alone is weak evidence -- with ~5k readings rounded to 4 dp, coincidence is real.
So a match is only reported as CONFIRMED when the artifact that carries the value also carries a
label (dataset, battery row, cell name) that appears in the manuscript line quoting it. VALUE-ONLY
means the number exists in some artifact but not one the sentence is talking about; read those.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["PAPER_DRAFT_V4.md", "PAPER_DRAFT_V4_KO.md", "paper/main.tex"]
FIELDS = ("staleness_harm", "denoised_staleness", "D_strip", "D_full", "injected_staleness",
          "delta_staleness", "noise_ratio", "recency_gain", "decay", "injection_learn_score")

# Artifacts that predate the row schema (the battery, the model-class matrices, the regression
# controls, whyshift, gap hygiene) store readings under their own key names at arbitrary depth.
# Walk those too, keeping only leaves whose key names an instrument reading.
READING_KEY = re.compile(
    r"(stale|denois|^den(_ci)?$|^raw(_ci)?$|^D_|recenc|decay|delta|gap|mean$|auc|score|^r2$"
    r"|ratio|recover|floor|alpha|theta|overlap|ess_pct|bias_corrected|corr|placebo|true_|"
    r"^vals$|^band$|_ci$)", re.I)
SKIP_KEY = re.compile(r"(^n$|^n_|_n$|seed|epoch|wall|feats|pairs|cuts|min_per_half|cols_used"
                      r"|windows|^tests$|^best_|_pct_n)", re.I)
LABEL_KEY = ("dataset", "row", "task", "name", "cell", "variant", "arm")

NUM = re.compile(r"[+−-]?\d+\.\d{3,5}")          # 3-5 dp: instrument readings, not years
TOKEN = re.compile(r"[a-z][a-z0-9]{3,}")         # label words worth matching against a sentence


def add(vals, v, label):
    """Record one reading under its rounded value, with the label a sentence must echo."""
    vals.setdefault(round(float(v), 4), []).append((label, frozenset(TOKEN.findall(label.lower()))))


def walk(vals, node, label, key=""):
    """Collect instrument-looking numeric leaves from a schema we do not know in advance."""
    if isinstance(node, dict):
        own = next((str(node[k]) for k in LABEL_KEY if isinstance(node.get(k), str)), "")
        sub = f"{label}/{own}" if own else label
        for k, v in node.items():
            walk(vals, v, sub, k)
    elif isinstance(node, list):
        for v in node:
            walk(vals, v, label, key)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if READING_KEY.search(key) and not SKIP_KEY.search(key) and abs(node) <= 100:
            add(vals, node, f"{label}.{key}")


def load_artifacts():
    vals = {}          # rounded value -> [(label, tokens)]
    n_files = n_walked = 0
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
            rows = blob.get("rows") if isinstance(blob, dict) else None
            if not isinstance(rows, list):
                # Pre-row schema (battery, model-class matrices, regression controls, whyshift,
                # gap hygiene). These carry manuscript numbers too; the old loader dropped them.
                n_walked += 1
                walk(vals, blob, Path(f).stem)
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
                        add(vals, v, f"{ds}.{fld} [{ctx}]")
                for fld in ("staleness_harm_ci", "denoised_staleness_ci", "injected_staleness_ci"):
                    ci = r.get(fld)
                    if isinstance(ci, list):
                        for v in ci:
                            if isinstance(v, (int, float)):
                                add(vals, v, f"{ds}.{fld} [{ctx}]")
    return vals, n_files, n_walked


KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+\.\d+)")
BRACKET = re.compile(r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]")


def load_logs(vals):
    """Some readings only ever existed in a run log -- the ACS positive rates, for one. Harvest
    key=value pairs and bracketed CIs, labelled by the identifiers on their own line."""
    n = 0
    for f in glob.glob(str(ROOT / "logs/**/*.log"), recursive=True):
        n += 1
        stem = Path(f).stem
        for line in open(f, encoding="utf-8", errors="replace"):
            ids = [t for t in TOKEN.findall(line.lower()) if not t.isdigit()]
            if not ids:
                continue
            label = f"{stem}: {' '.join(ids[:4])}"
            for k, v in KV.findall(line):
                if abs(float(v)) <= 100:
                    add(vals, float(v), f"{label}.{k}")
            for lo, hi in BRACKET.findall(line):
                for v in (lo, hi):
                    add(vals, float(v), f"{label}.ci")
    return n


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
    try:                                   # Windows consoles default to cp949 here
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0005)
    ap.add_argument("--show", default="unmatched", choices=["unmatched", "matched", "all"])
    args = ap.parse_args()

    vals, n_files, n_walked = load_artifacts()
    n_logs = load_logs(vals)
    keys = sorted(vals)
    print(f"artifacts: {n_files} row files + {n_walked} walked + {n_logs} logs, "
          f"{len(keys)} distinct instrument values\n")

    per_doc = {}
    for d in DOCS:
        nums = doc_numbers(d)
        per_doc[d] = nums
        confirmed = value_only = near = unmatched = 0
        misses = []
        for ln, v, ctx in nums:
            cand = [k for k in ([round(v, 4)] if round(v, 4) in vals
                                # +1e-9: a 3-dp quote of x.xxx5 sits exactly on the tolerance
                                else [k for k in keys if abs(k - v) <= args.tol + 1e-9])]
            if not cand:
                unmatched += 1
                misses.append((ln, v, ctx))
                continue
            words = set(TOKEN.findall(ctx.lower()))
            hit = next(((k, lab) for k in cand for lab, toks in vals[k] if toks & words), None)
            exact = round(v, 4) in vals
            if hit:
                confirmed += 1
                if args.show in ("matched", "all"):
                    kind = "CONF " if exact else "CONF~"
                    print(f"  {kind}  {d}:{ln}  {v:+.4f}  <- {hit[1]}")
            else:
                value_only += 1
                if args.show in ("matched", "all"):
                    kind = "VALUE" if exact else "NEAR "
                    print(f"  {kind}  {d}:{ln}  {v:+.4f}  <- {vals[cand[0]][0][0]}")
            near += 0 if exact else 1
        print(f"{d}:  {len(nums)} numbers | confirmed {confirmed} | value-only {value_only} "
              f"| unmatched {unmatched}   (of which {near} matched only within {args.tol})")
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
