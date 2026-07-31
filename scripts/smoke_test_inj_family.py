"""Smoke test for the injection sweep axes (PREREG §16).

Fills the gap recorded as INJFAMILY_SWEEP_PLAN E.2-1: the v1 `--inj-family` smoke test left NO
artifact — its only trace is prose in commit ffe1e56 and REVIEW_ROUND6:13, so nothing about it is
checkable. This one writes a commit-stamped JSON.

It checks four things, in this order (a failure of 1 blocks everything downstream):

  1. PARITY. The injected LABELS for (topvar|lowvar|subpop, auto) must be BIT-IDENTICAL to the v1
     implementation, inlined below as `_inject_concept_v1`. If this fails, the committed battery
     and the committed map are no longer reproducible and the change must be reverted.
     `interaction` is intentionally exempt (v1's axes were dependent; it has never been executed
     on real data, so no committed artifact depends on it). The reported feature tuple is
     informational and is allowed to GROW (subpop now also logs its gating column) -- it is not
     read by any computation.
  2. INDEPENDENCE. v1's `interaction` rotated z(f0*f1) against z(f1) -- an axis against one of its
     own factors. The two axes must now be (near-)uncorrelated.
  3. CARRIER SEPARATION. On a geometry built to reproduce the diagnosed real-data failure mode
     (heavy-tailed top-variance columns; audit L4):
        ASSERT  every rule geometry at `lo` PASSES the learnability gate
        ASSERT  the PURE-carrier geometries (topvar, subpop) at `hi` FAIL it
        REPORT  `interaction` at `hi` -- NOT asserted. The fixed interaction pulls a third column
                in as its second axis, so it is a HYBRID carrier: it can stay learnable while
                f0/f1 are degenerate. This is a property of the geometry, not a defect, but it
                means an interaction recovery is not attributable to the nominal carrier alone.
                Recorded so the sweep read-out can say so.
     Without this separation interaction/subpop inherit topvar's carrier and a non-recovery
     cannot be attributed -- the hardening step returns a false negative.
  4. GATE UNCHANGED. The learnability thresholds are the committed constants, not per-family.

Usage:  python scripts/smoke_test_inj_family.py
Writes: results/phase1/deployment_decay/smoke_inj_family_<git>.json
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_deployment_decay import (                       # noqa: E402
    _inject_concept, _injection_learnable, _z, INJ_STRENGTH, LEARN_AUC, LEARN_R2, LEARN_ACC,
)

FAMILIES = ("topvar", "lowvar", "interaction", "subpop")
N, K = 6000, 10


# --------------------------------------------------------------------------- v1 reference
def _inject_concept_v1(X, t, task, strength=2.5, seed=0, family="topvar"):
    """Verbatim copy of the pre-change implementation (commit 606eedb), for the parity check."""
    rng = np.random.default_rng(seed)
    Xf = np.nan_to_num(np.asarray(X, float))
    tn = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-12)
    order = np.argsort(-Xf.std(0))
    if family == "lowvar":
        nz = [int(j) for j in order[::-1] if Xf[:, int(j)].std() > 1e-9]
        f0, f1 = (nz[0], nz[1]) if len(nz) >= 2 else (
            (int(order[0]), int(order[1])) if Xf.shape[1] >= 2 else (0, 0))
    else:
        f0, f1 = (int(order[0]), int(order[1])) if Xf.shape[1] >= 2 else (0, 0)
    a, b = _z(Xf[:, f0]), _z(Xf[:, f1])
    if family == "interaction":
        a = _z(a * b)
    ang = strength * tn
    if family == "subpop":
        f2 = int(order[2]) if Xf.shape[1] >= 3 else f1
        ang = ang * (_z(Xf[:, f2]) > 0).astype(float)
    score = np.cos(ang) * a + np.sin(ang) * b + rng.normal(0, .3, len(tn))
    if task == "regression":
        return score.astype(float), (f0, f1)
    if task == "multiclass":
        s2 = np.cos(ang + 2.0) * a + np.sin(ang + 2.0) * b
        return np.stack([score, s2, -score - s2], axis=1).argmax(1), (f0, f1)
    return (score > np.median(score)).astype(int), (f0, f1)


# --------------------------------------------------------------------------- geometries
def _geom_heavytail(seed=0):
    """The diagnosed real-data failure mode (audit L4, and the mechanism behind ecom /
    homecredit / weather injection-vacuity): the two HIGHEST-variance columns are heavy-tailed,
    so z-scoring parks ~99% of rows near 0 and the N(0,.3) noise term dominates the planted
    score -> the rule is unlearnable even in-window. The LOW-variance columns are clean
    indicator-like features that carry a rule fine. A carrier-blind sweep cannot tell these
    apart; that is the whole point of --inj-cols."""
    rng = np.random.default_rng(seed)
    heavy = rng.standard_cauchy((N, 2)) * 50.0          # huge variance, ~all mass at ~0 after z
    mid = rng.normal(0, 3.0, (N, 3))
    lo = rng.normal(0, 0.05, (N, 4))                    # low variance, well-behaved
    X = np.column_stack([heavy, mid, lo])
    t = np.linspace(0, 1, N)
    return X, t


def _geom_clean(seed=0):
    """Well-behaved geometry: every column Gaussian, variance ordering meaningful. Both carriers
    should be learnable here -- this is the control that says a `hi` failure above is about the
    tails, not about the carrier machinery."""
    rng = np.random.default_rng(seed)
    sd = np.linspace(3.0, 0.3, 8)
    X = rng.normal(0, 1, (N, 8)) * sd
    t = np.linspace(0, 1, N)
    return X, t


def _axes(X, family, cols):
    """Reconstruct the two rotation axes the injector uses, for the independence check."""
    from scripts.run_deployment_decay import _carrier_pool
    Xf = np.nan_to_num(np.asarray(X, float))
    c = ("lo" if family == "lowvar" else "hi") if cols == "auto" else cols
    pool = _carrier_pool(Xf, c)
    a, b = _z(Xf[:, pool[0]]), _z(Xf[:, pool[1]])
    if family == "interaction":
        a = _z(a * b)
        if Xf.shape[1] >= 3:
            b = _z(Xf[:, pool[2]])
    return a, b


def _axes_v1(X, family):
    Xf = np.nan_to_num(np.asarray(X, float))
    order = np.argsort(-Xf.std(0))
    a, b = _z(Xf[:, order[0]]), _z(Xf[:, order[1]])
    if family == "interaction":
        a = _z(a * b)
    return a, b


# --------------------------------------------------------------------------- checks
def main():
    out = {"meta": {}, "parity": [], "independence": [], "carrier": [], "gate": {}}
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        git = ""
    out["meta"] = {"utc": datetime.now(timezone.utc).isoformat(), "git": git,
                   "python": sys.version.split()[0], "numpy": np.__version__,
                   "inj_strength": INJ_STRENGTH, "n": N, "K": K}

    # ---- 1. PARITY: (topvar|lowvar, auto) must be bit-identical to v1 ----------------------
    print("\n== 1. PARITY (auto must reproduce v1 bit-for-bit) ==")
    parity_ok = True
    for geom_name, geom in (("heavytail", _geom_heavytail), ("clean", _geom_clean)):
        X, t = geom()
        for task in ("binclass", "regression", "multiclass"):
            for fam in FAMILIES:
                yn, fn = _inject_concept(X, t, task, strength=INJ_STRENGTH, family=fam, cols="auto")
                yo, fo = _inject_concept_v1(X, t, task, strength=INJ_STRENGTH, family=fam)
                same = bool(np.array_equal(yn, yo))            # LABELS only; feats may grow
                # interaction is INTENTIONALLY changed (v1 axes were dependent); it has never
                # been executed on real data, so no committed artifact depends on it.
                expect = (fam != "interaction")
                ok = (same == expect)
                # the first two carrier columns must never move, whatever else is logged
                ok &= (tuple(fn)[:2] == tuple(fo)[:2]) or fam == "interaction"
                parity_ok &= ok
                out["parity"].append({"geom": geom_name, "task": task, "family": fam,
                                      "labels_identical_to_v1": same, "expected": expect, "ok": ok,
                                      "feats_new": list(map(int, fn)),
                                      "feats_v1": list(map(int, fo))})
                print(f"   {geom_name:<10} {task:<11} {fam:<12} labels_identical={str(same):<5} "
                      f"expect={str(expect):<5} feats {list(map(int, fo))}->{list(map(int, fn))} "
                      f"{'OK' if ok else 'FAIL'}")
    print(f"   -> PARITY {'PASS' if parity_ok else 'FAIL'}")

    # ---- 2. INDEPENDENCE: interaction axes must not be a factor of each other --------------
    print("\n== 2. INDEPENDENCE of the interaction rotation axes ==")
    indep_ok = True
    for geom_name, geom in (("heavytail", _geom_heavytail), ("clean", _geom_clean)):
        X, _ = geom()
        for cols in ("hi", "lo"):
            a, b = _axes(X, "interaction", cols)
            r_new = float(abs(np.corrcoef(a, b)[0, 1]))
            av, bv = _axes_v1(X, "interaction")
            r_v1 = float(abs(np.corrcoef(av, bv)[0, 1]))
            ok = r_new < 0.20
            indep_ok &= ok
            out["independence"].append({"geom": geom_name, "cols": cols, "abs_corr_new": r_new,
                                        "abs_corr_v1": r_v1, "ok": ok})
            print(f"   {geom_name:<10} cols={cols:<3} |corr| new={r_new:.4f} "
                  f"(v1={r_v1:.4f})  {'OK' if ok else 'FAIL (axes still dependent)'}")
    print(f"   -> INDEPENDENCE {'PASS' if indep_ok else 'FAIL'}")

    # ---- 3. CARRIER SEPARATION -------------------------------------------------------------
    print("\n== 3. CARRIER SEPARATION (the reason family x cols must be crossed) ==")
    print("   heavy-tail geometry: hi carrier must FAIL the gate, lo carrier must PASS")
    carrier_ok = True
    HYBRID = {"interaction"}          # second axis is a third column -> carrier is not pure
    for geom_name, geom, expect_hi in (("heavytail", _geom_heavytail, False),
                                       ("clean", _geom_clean, True)):
        X, t = geom()
        for fam in FAMILIES:
            for cols in ("hi", "lo"):
                y_inj, feats = _inject_concept(X, t, "binclass", strength=INJ_STRENGTH,
                                               family=fam, cols=cols)
                lrn, score, kind = _injection_learnable(X, y_inj, t, "binclass", K, False, seed=0)
                asserted = not (cols == "hi" and fam in HYBRID and not expect_hi)
                exp = (expect_hi if cols == "hi" else True) if asserted else None
                ok = (bool(lrn) == exp) if asserted else True
                carrier_ok &= ok
                out["carrier"].append({"geom": geom_name, "family": fam, "cols": cols,
                                       "carrier_cols_used": list(map(int, feats)),
                                       "learnable": bool(lrn), "score": score, "kind": kind,
                                       "asserted": asserted, "expected": exp, "ok": ok})
                mark = ("OK" if ok else "FAIL") if asserted else "report-only (hybrid carrier)"
                print(f"   {geom_name:<10} {fam:<12} cols={cols:<3} learnable={str(bool(lrn)):<5} "
                      f"{kind}={score if score is None else round(score, 4)!s:<8} "
                      f"cols_used={list(map(int, feats))!s:<12} {mark}")
    print(f"   -> CARRIER SEPARATION {'PASS' if carrier_ok else 'FAIL'}")

    # ---- 4. GATE UNCHANGED -----------------------------------------------------------------
    gate_ok = (LEARN_AUC, LEARN_R2, LEARN_ACC) == (0.65, 0.20, 0.10)
    out["gate"] = {"LEARN_AUC": LEARN_AUC, "LEARN_R2": LEARN_R2, "LEARN_ACC": LEARN_ACC,
                   "ok": gate_ok, "note": "global constants, applied identically to every "
                                          "(family, cols) combination"}
    print(f"\n== 4. GATE constants {(LEARN_AUC, LEARN_R2, LEARN_ACC)} "
          f"{'PASS' if gate_ok else 'FAIL (thresholds moved)'}")

    ok = parity_ok and indep_ok and carrier_ok and gate_ok
    out["verdict"] = "PASS" if ok else "FAIL"
    d = Path("results/phase1/deployment_decay"); d.mkdir(parents=True, exist_ok=True)
    p = d / f"smoke_inj_family_{git or 'nogit'}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  SMOKE {out['verdict']}   wrote {p}")
    print("  NOTE: this smoke test does NOT substitute for the PREREG §4 battery gate.")
    print("        Run `--synth` on the server and require a 14/14 match before any real data.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
