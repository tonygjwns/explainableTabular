"""Denoised-staleness estimator + noise-profile gate: candidate repair for the F1 kill
(noise-drift mints DEPLOYMENT-CONCEPT under a FIXED conditional mean).

CANDIDATE (as specified):
  denoised_staleness = score(recent N) - score(recent N  U  (X_old, g_old(X_old)) N), eval future,
  where g_old = HGB cross-fitted WITHIN the old window (2-fold: fit half A -> pseudo-label half B,
  and vice versa; pseudo-labels are strictly out-of-fold).
  Pseudo-label variant chosen: HARD labels via m.predict() -- for binclass this is the
  0.5-threshold on predict_proba, for multiclass the argmax, for regression the conditional-mean
  prediction. Simplest sound variant: HGB log-loss on hard Bayes labels conveys the rule; soft
  labels via sample duplication add nothing here and double the fit cost.

  Under NOISE-ONLY drift (fixed mean rule): g_old estimates the SAME mean -> pseudo-labels are
  denoised approximately-correct labels -> adding them must not hurt -> denoised <= floor.
  Under CONCEPT (rule changed): g_old encodes the OLD rule -> pseudo-labels still contradict the
  current rule -> harm persists -> denoised > floor.

NOISE-PROFILE GATE (cheaper alternative):
  per-window noise proxy with a fresh HGB per window (70/30 held-out):
    regression  -> held-out mean squared residual (y is z-scored, so unit-free)
    binclass    -> 1 - held-out AUC
    multiclass  -> 1 - held-out accuracy
  gate statistic = mean over seeds of proxy(old window) / median proxy(recent-pool windows).
  PRE-COMMITTED before any battery run: gate fires (positive staleness filed
  NOISE-DRIFT-CONFOUNDED instead of CONCEPT) if the ratio > 1.5. Stable controls must
  calibrate BELOW 1.5 for the threshold to stand (reported).

RNG DISCIPLINE (fidelity to the instrument): the raw arm consumes the IDENTICAL rng stream as
dd._per_seed (same subsample -> same window jitter -> same permutation -> same _sample calls in
the same order); all NEW draws (cross-fit fold split, noise-proxy splits, old_cap) use a separate
rng seeded seed+100000. Therefore raw staleness must reproduce dd's numbers bit-for-bit
(checked against exp-reg: early_noisy +0.021373, reg_concept +0.5459).
Decay/recency fits are skipped (they draw no rng; scores are not needed here).

Verdict logic of the REPAIRED estimator (pre-committed):
  den_fire  = denoised CI-lower > 0 AND mean > FLOOR_GAIN(0.02)
  raw_fire  = same rule on raw staleness
  if den_fire                     -> CONCEPT            (gate reported but NOT allowed to veto:
                                                          concept+noise-drift must still read CONCEPT)
  elif raw_fire and gate fired    -> NOISE-DRIFT-CONFOUNDED
  elif raw_fire                   -> RAW-ONLY-POSITIVE (unresolved; NOT concept under repair)
  else                            -> no concept signal
"""
import sys, json, importlib.util
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\joon\Desktop\ExplainableTab")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "dd", str(REPO / "scripts" / "run_deployment_decay.py"))
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)

GATE_THRESH = 1.5          # pre-committed noise-ratio threshold (validated on stable controls)


# --------------------------------------------------------------------- model access (shim-able)
def _clf(seed):
    return dd.HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed)


def _reg(seed):
    return dd.HistGradientBoostingRegressor(max_iter=200, early_stopping=False, random_state=seed)


def _fit_predict(Xtr, ytr, Xte, task, seed):
    """Hard predictions of a fresh model; None on degenerate train."""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    keep = dd._nonconstant_mask(Xtr)
    if not keep.any():
        return None
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if task == "regression":
        return _reg(seed).fit(Xtr, ytr).predict(Xte)
    u = np.unique(ytr)
    if len(u) < 2:
        return np.full(len(Xte), u[0])          # single-class fold -> constant pseudo-label
    return _clf(seed).fit(Xtr, ytr).predict(Xte)


def crossfit_pseudo(Xs, ys, pool, task, seed, rng2):
    """Out-of-fold pseudo-labels for rows `pool` (2-fold cross-fit within the pool).
    Returns array aligned with pool, or None if it failed."""
    m = len(pool)
    perm = rng2.permutation(m)
    half = m // 2
    foldA, foldB = pool[perm[:half]], pool[perm[half:]]
    pseudo = np.empty(m, dtype=float)
    posA, posB = perm[:half], perm[half:]
    try:
        pA = _fit_predict(Xs[foldB], ys[foldB], Xs[foldA], task, seed)   # B trains -> labels A
        pB = _fit_predict(Xs[foldA], ys[foldA], Xs[foldB], task, seed)   # A trains -> labels B
    except Exception:
        return None
    if pA is None or pB is None:
        return None
    pseudo[posA] = pA
    pseudo[posB] = pB
    if task != "regression":
        pseudo = pseudo.astype(ys.dtype)
    return pseudo


def window_noise_proxy(Xs, ys, idx, task, seed, rng2):
    """Per-window noise proxy: held-out irreducible-error estimate with a fresh model.
    regression -> held-out mean squared residual; binclass -> 1-AUC; multiclass -> 1-acc."""
    if len(idx) < 60:
        return None
    p = rng2.permutation(idx)
    cut = int(len(p) * 0.7)
    tr, ho = p[:cut], p[cut:]
    if not dd._ok_train(ys[tr], task):
        return None
    try:
        if task == "regression":
            pred = _fit_predict(Xs[tr], ys[tr], Xs[ho], task, seed)
            return float(np.mean((ys[ho] - pred) ** 2))
        s = dd._fit_score(Xs[tr], ys[tr], Xs[ho], ys[ho], task, seed)
        return None if s is None else float(1.0 - s)
    except Exception:
        return None


# --------------------------------------------------------------------- per-seed core
def per_seed(X, y, t, task, K, max_train, seed, old_cap=None):
    """Mirrors dd._per_seed's rng stream EXACTLY for the raw arm; adds the denoised arm and the
    per-window noise proxies using a separate rng (seed+100000)."""
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 100000)
    n = len(y)
    sub = rng.choice(n, int(0.9 * n), replace=False) if n > 200 else np.arange(n)
    Xs, ys, ts = X[sub], y[sub], t[sub]
    win, Keff = dd._assign_windows(ts, K, False, rng=rng)
    keep = win >= 0
    Xs, ys, win = Xs[keep], ys[keep], win[keep]
    by_w = [np.where(win == k)[0] for k in range(Keff)]
    old_w = next((k for k in range(Keff)
                  if len(by_w[k]) >= dd.ROW_FLOOR and dd._ok_train(ys[by_w[k]], task)), None)
    if old_w is None:
        return None
    old_pool = by_w[old_w]
    # raw-arm rng parity: dd draws permutation(old_pool) then _sample for the decay fit
    w0 = rng.permutation(old_pool); cut = int(len(w0) * 0.7)
    _tr0 = dd._sample(w0[:cut], max_train, rng)      # draw consumed for parity; decay fit skipped
    # ---- B4c knob: cut the old window down (uses rng2 -> raw stream untouched) ----
    if old_cap is not None and len(old_pool) > old_cap:
        old_pool = rng2.choice(old_pool, old_cap, replace=False)
    # ---- cross-fitted pseudo-labels for the WHOLE old pool, once per seed ----
    pseudo = crossfit_pseudo(Xs, ys, old_pool, task, seed, rng2)
    pos_of = {int(i): k for k, i in enumerate(old_pool)}     # row index -> position in pseudo
    # ---- noise proxies ----
    noise_old = window_noise_proxy(Xs, ys, old_pool, task, seed, rng2)
    recent_windows = sorted({j - 1 for j in range(max(old_w + 1, Keff // 2), Keff)
                             if j - 1 != old_w})
    noise_recent = [v for v in (window_noise_proxy(Xs, ys, by_w[k], task, seed, rng2)
                                for k in recent_windows) if v is not None]
    noise_recent_med = float(np.median(noise_recent)) if noise_recent else None

    raw_stales, den_stales = [], []
    for j in range(max(old_w + 1, Keff // 2), Keff):
        te = by_w[j]
        if len(te) < 20 or (task != "regression" and len(np.unique(ys[te])) < 2):
            continue
        recent_pool = by_w[j - 1]
        N = min(len(recent_pool), len(old_pool), max_train)
        if N < 50:
            continue
        recent = dd._sample(recent_pool, N, rng)
        old = dd._sample(old_pool, N, rng)
        if not (dd._ok_train(ys[recent], task) and dd._ok_train(ys[old], task)):
            continue
        recent_old = np.concatenate([recent, old])
        sRec = dd._fit_score(Xs[recent], ys[recent], Xs[te], ys[te], task, seed)
        sRecOld = dd._fit_score(Xs[recent_old], ys[recent_old], Xs[te], ys[te], task, seed)
        if None not in (sRec, sRecOld):
            raw_stales.append(sRec - sRecOld)
        if pseudo is not None and sRec is not None:
            y_den_old = pseudo[[pos_of[int(i)] for i in old]]
            y_comb = np.concatenate([ys[recent], y_den_old])
            sRecOldD = dd._fit_score(Xs[recent_old], y_comb, Xs[te], ys[te], task, seed)
            if sRecOldD is not None:
                den_stales.append(sRec - sRecOldD)
    mean = lambda a: float(np.mean(a)) if a else None
    return {"raw": mean(raw_stales), "den": mean(den_stales),
            "noise_old": noise_old, "noise_recent": noise_recent_med, "Keff": Keff}


# --------------------------------------------------------------------- assess wrapper
def _fire(m, ci):
    return (m is not None and ci[0] is not None and ci[0] > 0 and m > dd.FLOOR_GAIN)


def assess_denoised(name, X, y, t, task, K=10, n_seeds=5, max_train=6000, old_cap=None):
    X = dd._sanitize(X)
    y = np.asarray(y); t = np.asarray(t, float)
    good = np.isfinite(t)
    if task == "regression":
        good = good & np.isfinite(y.astype(float))
    X, y, t = X[good], y[good], t[good]
    if task == "regression":                    # z-score target, exactly as dd.assess
        yf = np.asarray(y, float); mu, sd = float(np.mean(yf)), float(np.std(yf)) + 1e-12
        y = (yf - mu) / sd
    keep_px = dd._proxy_mask(X, y, t, task)     # proxy-strip, exactly as dd.assess
    if not keep_px.all() and keep_px.any():
        X = X[:, keep_px]

    rows = [per_seed(X, y, t, task, K, max_train, s, old_cap=old_cap) for s in range(n_seeds)]
    rows = [r for r in rows if r is not None]
    raw, raw_ci = dd._ci95([r["raw"] for r in rows])
    den, den_ci = dd._ci95([r["den"] for r in rows])
    ratios = [r["noise_old"] / r["noise_recent"] for r in rows
              if r["noise_old"] is not None and r["noise_recent"] not in (None, 0)]
    ratio, ratio_ci = dd._ci95(ratios)
    gate = ratio is not None and ratio > GATE_THRESH
    raw_fire, den_fire = _fire(raw, raw_ci), _fire(den, den_ci)
    if den_fire:
        verdict = "CONCEPT"
    elif raw_fire and gate:
        verdict = "NOISE-DRIFT-CONFOUNDED"
    elif raw_fire:
        verdict = "RAW-ONLY-POSITIVE"
    else:
        verdict = "NO-CONCEPT-SIGNAL"
    return {"dataset": name, "task": task, "n_seeds_ok": len(rows), "old_cap": old_cap,
            "raw_staleness": raw, "raw_ci": raw_ci, "raw_fire": raw_fire,
            "denoised_staleness": den, "denoised_ci": den_ci, "den_fire": den_fire,
            "noise_ratio_old_over_recent": ratio, "noise_ratio_ci": ratio_ci,
            "gate_thresh": GATE_THRESH, "noise_gate_fired": bool(gate),
            "repaired_verdict": verdict,
            "per_seed": rows}


# --------------------------------------------------------------------- generators
def gen(kind, seed=0, n=12000, d=10):
    """Regression controls mirror exp-reg/run_reg_controls.py style; binclass controls call
    dd._synth verbatim."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d))
    if kind == "reg_early_noisy":               # F1 killer: fixed mean, noise 1.5 -> 0.3
        sd = 1.5 - 1.2 * t
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    if kind == "reg_early_noisy_extreme":       # KILL-TEST of the repair itself: noise 3.0 -> 0.3
        sd = 3.0 - 2.7 * t                      # pseudo-label estimation error scales with noise;
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd  # does denoised cross the 0.02 floor?
        return X, y, t, "regression"
    if kind == "reg_early_noisy_xtreme2":       # boundary mapping: noise 6.0 -> 0.3
        sd = 6.0 - 5.7 * t
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    if kind == "reg_concept_xtremenoisy":       # power at the boundary: concept + noise 6.0 -> 0.3
        ang = 3.0 * t
        sd = 6.0 - 5.7 * t
        y = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    if kind == "reg_late_noisy":                # B4d: noise 0.3 -> 1.5 (direction that passed)
        sd = 0.3 + 1.2 * t
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    if kind == "reg_concept":                   # B2: rotating rule, const noise
        ang = 3.0 * t
        y = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, 0.3, n)
        return X, y, t, "regression"
    if kind == "reg_concept_earlynoisy":        # B4a: concept + noise-decay simultaneously
        ang = 3.0 * t
        sd = 1.5 - 1.2 * t
        y = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    if kind == "reg_xdep_noise":                # B4b: x-DEPENDENT noise, scale drifts (early-noisy)
        sd = (1.5 - 1.2 * t) * (0.5 + 1.0 / (1.0 + np.exp(-2.0 * X[:, 1])))
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    if kind == "reg_stable":                    # gate calibration
        y = 3 * X[:, 0] + rng.normal(0, 0.3, n)
        return X, y, t, "regression"
    # binclass / multiclass ground-truth controls: use the repo's own generators verbatim
    return dd._synth(kind, seed=seed, n=n, d=d)


# --------------------------------------------------------------------- shims (B5)
def apply_shim(kind):
    sys.path.insert(0, str(Path(__file__).parent.parent / "exp-modelclass"))
    from shims import make_shims
    C, R = make_shims(kind)
    dd.HistGradientBoostingClassifier = C
    dd.HistGradientBoostingRegressor = R


# --------------------------------------------------------------------- CLI
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kinds", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--old-cap", type=int, default=None)
    ap.add_argument("--model", default="hgb", choices=["hgb", "rf", "knn", "linear"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.model != "hgb":
        apply_shim(args.model)
    out = Path(__file__).parent / "battery_results.json"
    prev = json.loads(out.read_text())["rows"] if out.exists() else []
    for kind in args.kinds:
        X, y, t, task = gen(kind)
        name = kind + (f"__{args.model}" if args.model != "hgb" else "") \
                    + (f"__oldcap{args.old_cap}" if args.old_cap else "") \
                    + (f"__{args.tag}" if args.tag else "")
        r = assess_denoised(name, X, y, t, task, K=10, n_seeds=args.seeds,
                            max_train=6000, old_cap=args.old_cap)
        r["model"] = args.model
        print(json.dumps({k: r[k] for k in (
            "dataset", "repaired_verdict", "raw_staleness", "raw_ci", "denoised_staleness",
            "denoised_ci", "noise_ratio_old_over_recent", "noise_gate_fired", "n_seeds_ok")},
            default=float), flush=True)
        prev.append(r)
        out.write_text(json.dumps({"rows": prev}, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
