"""V5 — validate the broad negative with the DEPLOYMENT lens, not the within-overlap lens.

After 8 within-overlap negatives, the earned claim is "unmeasurable by the within-overlap
instrument," NOT "concept is absent." That instrument (median early/late split + common-support
gap) is structurally blind to a changed rule that lives in the NEW, non-overlapping region: it
abstains exactly there (DISDE term-ii, chosen early, never questioned). Malware is the smoking
gun — concept drift is literature-established (TESSERACT) yet our EMBER read concept~0 — pointing
at the tool/split, not the phenomenon.

So we drop that lens and ask the question a deployed model actually faces: train on the PAST,
predict the FUTURE. Three quantities, rolling-origin, imbalance kept:

  decay          : a model trained on the OLDEST window loses how much AUC on future windows
                   (baseline = its own held-out early score).  Necessary, not sufficient (covariate
                   shift alone decays a model).
  recency_gain   : on the SAME future window W_j, a model trained on the most-RECENT past window
                   beats one trained on the OLDEST window (both size-matched).  Controls for window
                   difficulty (the INSECTS achievable-accuracy trap), but NOT for covariate
                   coverage (recent X is simply closer to W_j) -> conflates coverage + rule-change.
  staleness_harm : recent-only(N)  vs  recent+old (the SAME N recent samples + N OLD ones), eval
                   on W_j.  THE concept-isolating discriminator.  The recent portion is IDENTICAL
                   in both, so there is NO covariate-density / coverage confound — the ONLY
                   difference is the N extra OLD samples.  Under a FIXED rule old data carries
                   correct labels -> adding it helps or is neutral -> staleness_harm <= 0.  Under
                   concept drift old labels are WRONG now -> adding them pollutes the fit -> the
                   future score drops -> staleness_harm > 0 = the rule P(y|x) changed = CONCEPT.

v3 DECISION RULE (frozen in PREREG_DEPLOYMENT_V2.md; scores — binclass: AUC; multiclass: acc;
regression: -RMSE on z-scored y). Two staleness arms per comparison:
  RAW       staleness on logged labels (as v2).
  DENOISED  old labels replaced by 2-fold cross-fitted within-old-window model predictions.
            Under NOISE-ONLY drift (fixed rule, label-noise level changes) the pseudo-labels are
            approximately-correct denoised labels -> harm vanishes; under a CHANGED rule they
            still encode the OLD rule -> harm persists. (Audit F1: noise decay alone minted a
            +0.021 'concept' under a provably fixed rule — the raw arm alone is not sound.)
plus a NOISE GATE: per-window held-out noise proxy (regression: MSE; binclass: 1-AUC; multiclass:
1-acc), statistic = old/median(recent); fires > GATE_THRESH, denoiser validity envelope ends at
GATE_ENVELOPE (beyond it the denoiser's own bias can cross the floor -> abstain).
  DEPLOYMENT-CONCEPT        denoised fires (CI>0 & mean>floor) and noise ratio within envelope
  NOISE-AMBIGUOUS           denoised fires but ratio > envelope -> abstain (denoiser bias zone)
  NOISE-DRIFT-CONFOUNDED    raw fires, denoised null, gate fired -> label-noise drift, not rule
  RAW-ONLY-POSITIVE         raw fires, denoised null, gate quiet -> unresolved; NOT concept
  UNIDENTIFIABLE-{EXPLOITABLE,INERT} / INJECTION-RECOVERED  staleness null + separability gate
                            D>=D* -> learnability-gated injection decides earned/recovered/vacuous
  SUBFLOOR-CONCEPT-SIGNAL / NO-STRONG-CONCEPT / DEPLOYMENT-DECAY-COVARIATE / INCONCLUSIVE / NO-DATA
SCOPE (audit F2/F3, executed): D measures WINDOW SEPARABILITY, not support disjointness (group-
aware split + random subsample fix the duplicate/cohort saturation channel; semantics stay
separability). All verdicts are relative to the tree-ensemble hypothesis class — kNN/linear
false-fire CONCEPT under fixed rules (+0.098/+0.026); do not transfer verdicts across classes.

Discipline (lessons baked in): size-matched training, same-window gain comparison, denoised arm +
noise gate (F1), group-aware D (F2), class scoping (F3), learnability-gated injection (L4),
multi-seed CIs with per-seed window jitter, and a synthetic ground-truth battery (binclass +
regression + adversarial noise-drift kinds) that MUST pass before any real-data read is trusted.

Model-light (sklearn). Generic: point it at any time-ordered tabular CSV/parquet.

    python scripts/run_deployment_decay.py --synth                     # ground-truth controls (run FIRST)
    python scripts/run_deployment_decay.py --csv ember.parquet --target label --time appeared --by-value
    python scripts/run_deployment_decay.py --csv baf.parquet --target fraud_bool --time month
    python scripts/run_deployment_decay.py --tabred sberbank_housing homecredit_default --config configs/phase1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error  # noqa: E402

FLOOR_GAIN = 0.02     # a gain/harm below this (in AUC/acc units) is noise
FLOOR_DECAY = 0.02    # a decay below this is not "the model meaningfully ages"
DSTAR = 0.96          # identifiability gate: D_strip >= DSTAR => old data can't reach the future
                      # covariate region, so a staleness null is UNIDENTIFIABLE (not "no concept").
                      # Derived from the power curve (staleness alive to AUC~0.985, dead by ~0.999).
INJ_STRENGTH = 2.5    # reference concept strength for the injection control (clears the floor in
                      # overlapping geometry — validated: nuisance_proxy staleness ~+0.2 at 2.5).
ROW_FLOOR = 200       # a window/old-anchor needs >= this many rows to be usable (NEW-4: a sparse
                      # early window shrinks N and biases staleness toward the null).
MIN_POS = 10          # a usable binclass window/sample needs >= this many of the minority class
GATE_THRESH = 1.5     # noise gate: old/recent noise-proxy ratio above this = noise-drift present
                      # (pre-committed; stable controls calibrate at ~0.75-0.99, well below).
GATE_ENVELOPE = 4.7   # denoiser validity envelope: beyond this ratio the cross-fitted pseudo-labels
                      # carry enough estimation bias to cross the 0.02 floor on a pure null
                      # (executed: den +0.026 at ratio 5.7) -> abstain (NOISE-AMBIGUOUS).
INJ_SEEDS = 10        # injection control seeds — same power as the real read (audit L4: was 4).
LEARN_AUC = 0.65      # injection learnability gates: the injected rule must be learnable IN-WINDOW
LEARN_R2 = 0.20       # (held-out AUC / R^2 / acc-over-majority margin) or the injection null is
LEARN_ACC = 0.10      # VACUOUS (audit L4: junk top-variance features -> in-window AUC 0.506 ->
                      # false 'unident-earned'). Unlearnable -> flag injection-vacuous, not earned.
D_SEEDS = 5           # D is now a multi-seed median (audit L2: was a single seed-0 point estimate)


# ----------------------------------------------------------------------------- scoring
_FIT_WARNED = set()
_DEBUG_RAISE = False     # set by --debug-raise: re-raise HGB errors with full traceback


def _fit_score(Xtr, ytr, Xte, yte, task, seed):
    """Higher-is-better score of a fresh HGB trained on (Xtr,ytr), eval on (Xte,yte).
    Returns None if the window can't yield a valid score (single-class train/test, fit error).
    HGB handles NaN natively; imbalance is preserved (no resampling). Any fit/predict error is
    swallowed to None (one-time warning) so a single degenerate window can't kill the batch."""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    # Drop columns constant WITHIN this training subset (not just globally). sklearn>=1.9's
    # bin-mapper computes midpoints between consecutive distinct values via sliding_window_view(.,2),
    # which raises "window shape cannot be larger than input array shape" on a single-distinct-value
    # column. A within-subset constant carries no signal, so dropping it is lossless and removes the
    # crash. (Bootstrap/sparse windows make this common on e.g. sberbank's 392 features.)
    keep = _nonconstant_mask(Xtr)
    if not keep.any():
        return None
    if not keep.all():
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    # early_stopping=False (NOT the default 'auto', which turns ON only when n_samples>10000):
    # the staleness comparison trains on N and 2N, and at the real regime (N~6000) only the 2N arm
    # would cross the 'auto' threshold -> asymmetric regularization that biases the comparison
    # independent of any rule change (red-team Flaw 1). Fixed regularization for every fit removes it.
    HGB = dict(max_iter=200, early_stopping=False, random_state=seed)
    try:
        if task == "regression":
            m = HistGradientBoostingRegressor(**HGB).fit(Xtr, ytr)
            return -float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))   # -RMSE: higher better
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return None
        m = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr)
        if task == "binclass":
            pos = list(m.classes_).index(m.classes_[-1])
            return float(roc_auc_score(yte, m.predict_proba(Xte)[:, pos]))
        return float(accuracy_score(yte, m.predict(Xte)))
    except Exception as e:                                  # version-specific binning quirks, etc.
        if _DEBUG_RAISE:
            raise
        key = type(e).__name__
        if key not in _FIT_WARNED:
            _FIT_WARNED.add(key)
            print(f"    [warn] HGB fit/score skipped a window: {key}: {e} "
                  f"(Xtr={np.asarray(Xtr).shape})", file=sys.stderr)
        return None


def _nonconstant_mask(X):
    """Columns with >= 2 distinct non-NaN values. Computed WITHOUT triggering numpy's
    'Degrees of freedom <= 0' warning: nanstd is evaluated only on columns that have at least one
    non-NaN value (so N>=1, ddof=0 -> N>0), never on an all-NaN column (N=0, the warning's cause).
    The all-NaN columns are simply excluded up front. Result is identical, output is clean."""
    X = np.asarray(X, float)
    not_all_nan = ~np.all(np.isnan(X), axis=0)
    keep = np.zeros(X.shape[1], dtype=bool)
    if not_all_nan.any():
        with np.errstate(all="ignore"):
            keep[not_all_nan] = np.nanstd(X[:, not_all_nan], axis=0) > 0
    return keep


def _fit_predict(Xtr, ytr, Xte, task, seed):
    """Hard predictions of a fresh model (denoiser building block, ported from the audited
    harness). None on degenerate train; single-class fold -> constant pseudo-label."""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    keep = _nonconstant_mask(Xtr)
    if not keep.any():
        return None
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    HGB = dict(max_iter=200, early_stopping=False, random_state=seed)
    if task == "regression":
        return HistGradientBoostingRegressor(**HGB).fit(Xtr, ytr).predict(Xte)
    u = np.unique(ytr)
    if len(u) < 2:
        return np.full(len(Xte), u[0])
    return HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr).predict(Xte)


def _crossfit_pseudo(Xs, ys, pool, task, seed, rng2):
    """Out-of-fold pseudo-labels for rows `pool` (2-fold cross-fit WITHIN the pool): fit half A ->
    label half B, and vice versa. Under a fixed rule these are denoised approximately-correct
    labels; under a changed rule they still encode the pool's (old) rule. Returns array aligned
    with pool, or None on failure. (Validated: kills the noise-decay false positive +0.021 ->
    +0.004 while retaining rotating-rule power +0.541; battery in audit_artifacts_2026-07-04.)"""
    m = len(pool)
    perm = rng2.permutation(m)
    half = m // 2
    foldA, foldB = pool[perm[:half]], pool[perm[half:]]
    pseudo = np.empty(m, dtype=float)
    posA, posB = perm[:half], perm[half:]
    try:
        pA = _fit_predict(Xs[foldB], ys[foldB], Xs[foldA], task, seed)
        pB = _fit_predict(Xs[foldA], ys[foldA], Xs[foldB], task, seed)
    except Exception:
        return None
    if pA is None or pB is None:
        return None
    pseudo[posA] = pA
    pseudo[posB] = pB
    if task != "regression":
        pseudo = pseudo.astype(ys.dtype)
    return pseudo


def _window_noise_proxy(Xs, ys, idx, task, seed, rng2):
    """Per-window noise proxy: held-out irreducible-error estimate with a fresh model.
    regression -> held-out mean squared residual (y z-scored, unit-free); binclass -> 1-AUC;
    multiclass -> 1-acc. Feeds the noise gate (old / median(recent))."""
    if len(idx) < 60:
        return None
    p = rng2.permutation(idx)
    cut = int(len(p) * 0.7)
    tr, ho = p[:cut], p[cut:]
    if not _ok_train(ys[tr], task):
        return None
    try:
        if task == "regression":
            pred = _fit_predict(Xs[tr], ys[tr], Xs[ho], task, seed)
            return None if pred is None else float(np.mean((ys[ho] - pred) ** 2))
        s = _fit_score(Xs[tr], ys[tr], Xs[ho], ys[ho], task, seed)
        return None if s is None else float(1.0 - s)
    except Exception:
        return None


def _row_groups(X, decimals=1):
    """Group ids for the group-aware D split: rows identical after z-scoring + rounding share a
    group (exact duplicates and tight near-duplicates cluster; honest iid rows stay singleton —
    at d>=10 the collision probability of distinct N(0,1) rows is ~0). Validated fix for the
    duplicate/cohort D-saturation channel (audit F2: dup m=5 D 0.994 -> 0.504 grouped; cohorts
    1.000 -> 0.525) while honest drift D is unchanged."""
    X = np.nan_to_num(np.asarray(X, float))
    sd = X.std(0); sd[sd == 0] = 1.0
    Z = np.round((X - X.mean(0)) / sd, decimals)
    _, groups = np.unique(Z, axis=0, return_inverse=True)
    return groups


def _proxy_mask(X, y, t, task, max_n=20000, seed=0):
    """Return a KEEP mask (True = non-proxy). A time-PROXY feature is one that, ALONE, strongly
    separates early from late (single-feature early/late AUC > 0.70) AND has ~no task-predictive
    value (marginal MI < 5% of the top feature's MI). Such a feature (a clock / row-id / drifting nuisance) does
    two bad things, both validated on synth: it inflates the disjointness measure D toward 1.0, and
    — critically — it BLINDS the staleness model (a drifting nuisance made staleness read +0.0004;
    stripping it recovered the true +0.236). A time-separating feature that IS predictive is REAL
    covariate drift and is KEPT. Predictive value is MUTUAL INFORMATION, not rank-correlation — a
    non-monotonic predictive feature (a sin rule) has ~0 rank-corr but high MI and must NOT be stripped."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    X = np.asarray(X, float); n, d = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    Xs = np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0)
    ts_ = np.asarray(t, float)[idx]; yv = np.asarray(y)[idx]
    late = (ts_ > np.median(ts_)).astype(int)
    keep = np.ones(d, bool)
    if len(np.unique(late)) < 2:
        return keep
    tsep = np.full(d, 0.5)                                             # single-feat time separation
    for j in range(d):
        col = Xs[:, j]
        if np.std(col) == 0:
            continue
        a = roc_auc_score(late, col); tsep[j] = max(a, 1 - a)
    try:                                                              # single-feat predictive value (MI)
        mi = (mutual_info_regression(Xs, yv.astype(float), random_state=seed) if task == "regression"
              else mutual_info_classif(Xs, yv.astype(int), random_state=seed))
    except Exception:
        return keep                                                  # MI failed -> strip nothing (safe)
    mi_cut = max(0.01, 0.05 * float(np.max(mi))) if mi.size else 0.01  # < 5% of the top feature's MI
    keep[(tsep > 0.70) & (mi < mi_cut)] = False
    return keep


def _disjointness(X, win, Keff, seed=0, groups=None):
    """D = median over back-half windows W_j of held-out AUC(oldest-window vs W_j) on the
    (proxy-stripped) feature space. SEMANTICS (audit F2): D measures WINDOW SEPARABILITY, not
    support disjointness — a single predictive drifting feature saturates it; do not read D>=D*
    as 'old data cannot reach the future covariate region'. Two executed fixes vs v2:
    (1) windows are size-matched by RANDOM subsample, not head-slice old[:nn]/fut[:nn] — the
        head-slice on time-sorted rows made the time-SHUFFLE control read D>0.6 with no static
        feature (truncation artifact, audit L3);
    (2) the train/test split is GROUP-aware (groups = duplicate/near-dup clusters via _row_groups,
        or caller-supplied entity ids) — a row-level split let HGB memorize duplicate/cohort rows
        and saturate D to 1.000 with ZERO covariate shift (audit F2, executed)."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    rng = np.random.default_rng(seed)
    by_w = [np.where(win == k)[0] for k in range(Keff)]
    old = by_w[0]
    if len(old) < 40:
        return None
    aucs = []
    for j in range(max(1, Keff // 2), Keff):
        fut = by_w[j]
        if len(fut) < 40:
            continue
        nn = min(len(old), len(fut), 5000)
        o = old if len(old) == nn else rng.choice(old, nn, replace=False)
        f = fut if len(fut) == nn else rng.choice(fut, nn, replace=False)
        sel = np.r_[o, f]
        Xd = np.nan_to_num(X[sel])
        yd = np.r_[np.zeros(nn), np.ones(nn)]
        try:
            if groups is not None:
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                                random_state=seed).split(Xd, yd, groups[sel]))
                if len(np.unique(yd[tr])) < 2 or len(np.unique(yd[te])) < 2:
                    continue
                Xa, ya, Xb, yb = Xd[tr], yd[tr], Xd[te], yd[te]
            else:
                Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.3, random_state=seed,
                                                  stratify=yd)
            m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                               random_state=seed).fit(Xa, ya)
            aucs.append(roc_auc_score(yb, m.predict_proba(Xb)[:, 1]))
        except Exception:
            continue
    return float(np.median(aucs)) if aucs else None


def _z(v):
    v = np.asarray(v, float); s = np.std(v)
    return (v - np.mean(v)) / (s + 1e-12)


def _inject_concept(X, t, task, strength=2.5, seed=0, family="topvar"):
    """Replace y with a KNOWN time-ROTATING rule of controlled strength, keeping this dataset's
    REAL covariate geometry (X, t). Probe for the injection control: run staleness on
    (X, y_injected). If it stays null on a candidate-UNIDENTIFIABLE dataset, the blindness is
    DEMONSTRATED on real geometry (UNIDENTIFIABLE-EARNED); if it recovers, D was misleading and
    the real null was informative (the dataset was identifiable after all).

    family (reviewer-2 R3 — the certificate is relative to the planted signal class; sweep it):
      topvar      : rotation on the 2 highest-variance features (the pre-registered reference)
      lowvar      : rotation carried by the 2 LOWEST-variance non-degenerate features
      interaction : rotation between an interaction term z(f0)*z(f1) and a main effect
      subpop      : the rotation runs only inside the half-population z(f2)>0 (local rule change;
                    the rule is static outside)
    The learnability gate applies unchanged, so an unlearnable family yields VACUOUS for that
    family rather than a fake certificate."""
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
        a = _z(a * b)                                   # the rule lives on an interaction term
    ang = strength * tn
    if family == "subpop":
        f2 = int(order[2]) if Xf.shape[1] >= 3 else f1
        ang = ang * (_z(Xf[:, f2]) > 0).astype(float)   # rule change only inside the subpopulation
    score = np.cos(ang) * a + np.sin(ang) * b + rng.normal(0, .3, len(tn))
    if task == "regression":
        return score.astype(float), (f0, f1)
    if task == "multiclass":
        s2 = np.cos(ang + 2.0) * a + np.sin(ang + 2.0) * b
        return np.stack([score, s2, -score - s2], axis=1).argmax(1), (f0, f1)
    return (score > np.median(score)).astype(int), (f0, f1)


def _injection_learnable(X, y_inj, t, task, K, by_value, seed=0):
    """Is the injected rule learnable IN-WINDOW at all? Without this check an injection null is
    VACUOUS (audit L4, executed: junk heavy-tailed top-variance features -> in-window AUC 0.506
    -> false 'unident-earned'; learnable control AUC 0.964 recovers +0.195). Fit on 70% of the
    LARGEST window, score the held-out 30%. Returns (learnable_bool, score, kind)."""
    win, Keff = _assign_windows(t, K, by_value)
    sizes = [(int((win == k).sum()), k) for k in range(Keff)]
    _, kbig = max(sizes)
    idx = np.where(win == kbig)[0]
    rng = np.random.default_rng(seed)
    p = rng.permutation(idx); cut = int(len(p) * 0.7)
    tr, ho = p[:cut], p[cut:]
    if len(ho) < 30 or not _ok_train(y_inj[tr], task):
        return False, None, "degenerate"
    if task == "regression":
        pred = _fit_predict(X[tr], y_inj[tr], X[ho], task, seed)
        if pred is None:
            return False, None, "r2"
        ss = float(np.var(y_inj[ho])) + 1e-12
        r2 = 1.0 - float(np.mean((y_inj[ho] - pred) ** 2)) / ss
        return r2 >= LEARN_R2, r2, "r2"
    s = _fit_score(X[tr], y_inj[tr], X[ho], y_inj[ho], task, seed)
    if s is None:
        return False, None, "score"
    if task == "binclass":
        return s >= LEARN_AUC, float(s), "auc"
    maj = float(np.bincount(y_inj[ho].astype(int)).max()) / len(ho)
    return (s - maj) >= LEARN_ACC, float(s), "acc-over-majority"


def _injection_recovers(X, t, task, K, by_value, max_train, n_seeds=INJ_SEEDS, seed_base=0,
                        family="topvar"):
    """Inject a reference-strength concept into (X, t) and test whether staleness fires.
    v3: learnability-gated (an unlearnable injection cannot 'earn' blindness), n_seeds matches the
    real read (audit L4: was 4), injected features logged. Returns
    (recovered, injected_staleness_mean, injected_staleness_ci, learnable, learn_score, feats).
    The CI is returned so the strict-rule shadow cascade can judge recovery under its own rule
    (PREREG §15: the shadow verdict must traverse the injection stage too, not stop before it)."""
    y_inj, feats = _inject_concept(X, t, task, strength=INJ_STRENGTH, family=family)
    learnable, lscore, _ = _injection_learnable(X, y_inj, t, task, K, by_value, seed=seed_base)
    st = []
    for s in range(seed_base, seed_base + n_seeds):
        out = _per_seed(X, y_inj, t, task, K, by_value, max_train, s)
        if out is not None and out["stale"] is not None:
            st.append(out["stale"])
    m, ci = _ci95(st)
    recovered = m is not None and ci[0] is not None and ci[0] > 0 and m > FLOOR_GAIN
    return recovered, m, ci, learnable, lscore, feats


def _sanitize(X):
    """inf -> NaN (HGB tolerates NaN, not always inf across versions) and drop all-NaN / constant
    columns (a zero-variance or all-missing column breaks the parallel bin-mapper on some sklearn
    builds — the cause of the TabReD 'sliding_window_view' crash). Mirrors covariate_shift_auc."""
    X = np.asarray(X, float)
    X[~np.isfinite(X)] = np.nan
    keep = _nonconstant_mask(X)
    return X[:, keep] if keep.any() else X


def _ok_train(y, task):
    if task == "regression":
        return len(y) >= 50
    if len(np.unique(y)) < 2:
        return False
    _, c = np.unique(y, return_counts=True)
    return c.min() >= MIN_POS


# ----------------------------------------------------------------------------- windows
def _assign_windows(t, K, by_value, rng=None):
    """Return per-row window id in [0..K-1] (or fewer). by_value: one window per unique time
    value (e.g. EMBER YYYYMM months = TESSERACT-faithful). else: K bins by TIMESTAMP-VALUE rank so
    all rows sharing a timestamp land in the SAME window (ties never straddle the past/future
    boundary — red-team Flaw 5). When rng is given, the K-1 window BOUNDARIES are jittered by a few
    unique-value ranks (block-bootstrap over timestamp values): this varies the partition across
    seeds even on COARSE timestamps, where a 90% row subsample leaves np.unique(t) unchanged and
    would otherwise FREEZE the partition -> under-dispersed CI (red-team NEW-3, reproduced)."""
    t = np.asarray(t, float)
    if by_value:
        uniq = np.unique(t[np.isfinite(t)])
        remap = {v: i for i, v in enumerate(uniq)}
        return np.array([remap.get(v, -1) for v in t], int), len(uniq)
    uniq, inv = np.unique(t, return_inverse=True)      # inv = rank of each row's timestamp value
    U = len(uniq)
    if rng is not None and U > K:
        edges = (np.arange(1, K) * U / K)              # K-1 interior boundaries in value-rank space
        # jitter = a FEW unique timestamps (capped 1..3): meaningful when timestamps are chunky
        # (coarse: ±1 value moves a whole block -> partition varies) and negligible when fine
        # (±few rows). Scaling with U over-jittered fine datasets (±25% of a window).
        b = int(np.clip(U // (4 * K), 1, 3))
        edges = np.clip(np.sort(edges + rng.integers(-b, b + 1, size=K - 1)), 1, U - 1)
        w = np.searchsorted(edges, inv, side="right")
    else:
        w = (inv * K // U).clip(0, K - 1)
    return w.astype(int), K


def _sample(idx, n, rng):
    """Draw n training indices. When the pool is larger than n, subsample WITHOUT replacement
    (varies per seed). When the pool is <= n, BOOTSTRAP (with replacement) to size n so that
    seeds still differ -> the across-seed CI reflects genuine sampling uncertainty instead of
    collapsing to width 0 (which would make 'CI excludes 0' vacuously true)."""
    if len(idx) <= n:
        return rng.choice(idx, n, replace=True)
    return rng.choice(idx, n, replace=False)


# ----------------------------------------------------------------------------- core
def _per_seed(X, y, t, task, K, by_value, max_train, seed):
    """One seed. A per-seed row SUBSAMPLE (90%) is windowed fresh, so the partition itself varies
    across seeds -> the across-seed CI captures partition + data-draw uncertainty, not only the
    training bootstrap (red-team Flaw 3). Returns (decay, recency_gain, staleness_harm, Keff).

    staleness_harm = score(recent N) − score(recent N ∪ old N): does ADDING old data on top of the
    SAME recent set hurt the future? >0 = old data is net-harmful = old labels contradict the current
    rule = concept. Under a fixed rule old data (correct labels) only adds coverage -> stale <= 0.
    NOTE on the red-team's proposed 'fixed-size' variant (recent∪old vs recent∪recent', both 2N): we
    implemented and ground-truth-tested it and REJECTED it — it leaks a covariate-coverage confound
    (recent' covers the future region better than old, so it reads stale>0 under a FIXED rule; the
    covariate synth went +0.000 -> +0.005). The N-vs-2N size effect it was meant to remove is
    empirically negligible (a d=10..700 bias sweep: N-vs-2N ≈ fixed-size on concept data), and
    early_stopping=False removes the only sharp asymmetry, so the original design is kept."""
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 100000)  # ALL v3 additions draw from rng2 ONLY, so the
    #                                              raw arm's stream stays bit-identical to v2
    #                                              (parity proven in the audited harness).
    n = len(y)
    sub = rng.choice(n, int(0.9 * n), replace=False) if n > 200 else np.arange(n)
    Xs, ys, ts = X[sub], y[sub], t[sub]
    win, Keff = _assign_windows(ts, K, by_value, rng=rng)   # rng => per-seed boundary jitter (NEW-3)
    keep = win >= 0
    Xs, ys, win = Xs[keep], ys[keep], win[keep]
    by_w = [np.where(win == k)[0] for k in range(Keff)]
    # robust "old" anchor: the EARLIEST window with enough ROWS (>=ROW_FLOOR, NEW-4) AND a valid
    # training set — skips a sparse historical tail whose tiny N would bias staleness toward null.
    old_w = next((k for k in range(Keff)
                  if len(by_w[k]) >= ROW_FLOOR and _ok_train(ys[by_w[k]], task)), None)
    if old_w is None:
        return None
    old_pool = by_w[old_w]
    # --- old-window model for the DECAY curve (its own held-out early baseline) ---
    w0 = rng.permutation(old_pool); cut = int(len(w0) * 0.7)
    tr0, ho0 = _sample(w0[:cut], max_train, rng), w0[cut:]
    base = (_fit_score(Xs[tr0], ys[tr0], Xs[ho0], ys[ho0], task, seed)
            if _ok_train(ys[tr0], task) and len(ho0) >= 20 else None)
    # --- v3: cross-fitted pseudo-labels for the whole old pool (once per seed) + noise proxies ---
    pseudo = _crossfit_pseudo(Xs, ys, old_pool, task, seed, rng2)
    pos_of = {int(i): k for k, i in enumerate(old_pool)}
    noise_old = _window_noise_proxy(Xs, ys, old_pool, task, seed, rng2)
    recent_ws = sorted({j - 1 for j in range(max(old_w + 1, Keff // 2), Keff) if j - 1 != old_w})
    noise_rec = [v for v in (_window_noise_proxy(Xs, ys, by_w[k], task, seed, rng2)
                             for k in recent_ws) if v is not None]
    noise_rec_med = float(np.median(noise_rec)) if noise_rec else None

    decays, recs, stales, dens = [], [], [], []
    for j in range(max(old_w + 1, Keff // 2), Keff):       # future windows, after the old anchor
        te = by_w[j]
        if len(te) < 20 or (task != "regression" and len(np.unique(ys[te])) < 2):
            continue
        if base is not None and _ok_train(ys[tr0], task):
            s = _fit_score(Xs[tr0], ys[tr0], Xs[te], ys[te], task, seed)
            if s is not None:
                decays.append(base - s)
        recent_pool = by_w[j - 1]
        N = min(len(recent_pool), len(old_pool), max_train)
        if N < 50:
            continue
        recent = _sample(recent_pool, N, rng)
        old = _sample(old_pool, N, rng)
        if not (_ok_train(ys[recent], task) and _ok_train(ys[old], task)):
            continue
        recent_old = np.concatenate([recent, old])     # same N recent + N old
        sRec = _fit_score(Xs[recent], ys[recent], Xs[te], ys[te], task, seed)
        sOld = _fit_score(Xs[old], ys[old], Xs[te], ys[te], task, seed)
        sRecOld = _fit_score(Xs[recent_old], ys[recent_old], Xs[te], ys[te], task, seed)
        if None not in (sRec, sOld):
            recs.append(sRec - sOld)
        if None not in (sRec, sRecOld):
            stales.append(sRec - sRecOld)              # >0 = adding old HURT = concept OR noise drift
        if pseudo is not None and sRec is not None:    # v3 denoised arm: same rows, pseudo old-labels
            y_den = np.concatenate([ys[recent], pseudo[[pos_of[int(i)] for i in old]]])
            sRecOldD = _fit_score(Xs[recent_old], y_den, Xs[te], ys[te], task, seed)
            if sRecOldD is not None:
                dens.append(sRec - sRecOldD)           # >0 here = the RULE changed (noise removed)
    mean = lambda a: float(np.mean(a)) if a else None
    return {"decay": mean(decays), "rec": mean(recs), "stale": mean(stales),
            "den": mean(dens), "noise_old": noise_old, "noise_rec": noise_rec_med, "Keff": Keff}


def _ci95(a):
    """Mean ± t(0.975, n-1)·SE. Uses the Student-t multiplier (n=5 -> 2.776, not 1.96) so a small
    seed count does not produce an anti-conservative interval (red-team Flaw 3)."""
    a = np.asarray([v for v in a if v is not None], float)
    if len(a) == 0:
        return None, [None, None]
    if len(a) < 2:
        return float(a[0]), [float(a[0]), float(a[0])]
    from scipy.stats import t as _student_t
    m, se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
    tm = float(_student_t.ppf(0.975, len(a) - 1))
    return m, [m - tm * se, m + tm * se]


def _delta(vals, alpha=0.05, power=0.8):
    """Minimum staleness the test could DETECT at the given power (one-sided). Reported so a null
    means 'staleness > delta is excluded' in the dataset's OWN units — metric-commensurate, so the
    single 0.02 floor is not the only thing standing behind a 'no strong concept' read (NEW-5)."""
    a = [v for v in vals if v is not None]
    if len(a) < 2:
        return None
    from scipy.stats import norm
    se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * se)   # ~2.49 * SE


def assess(name, X, y, t, task, K=10, by_value=False, n_seeds=5, max_train=6000, seed_base=0,
           inj_family="topvar"):
    """seed_base: exploratory runs use 0 (seeds 0..n-1); the pre-registered CONFIRMATORY rerun
    uses 100 (seeds 100..). D/shuffle diagnostics stay at their fixed seeds (deterministic
    geometry reads, identical across runs by design)."""
    X = _sanitize(X)
    y = np.asarray(y)
    t = np.asarray(t, float)
    # drop rows with a non-finite target (regression) or non-finite time
    good = np.isfinite(t)
    if task == "regression":
        good = good & np.isfinite(y.astype(float))
    X, y, t = X[good], y[good], t[good]
    # Regression: standardize the target (z-score) so -RMSE is in std units. The CONCEPT/STABLE
    # floor (0.02) is then unit-invariant — without this a target rescaled x100 would flip the
    # verdict (red-team Flaw 4). For AUC/accuracy the scores are already in [0,1].
    if task == "regression":
        yf = np.asarray(y, float); mu, sd = float(np.mean(yf)), float(np.std(yf)) + 1e-12
        y = (yf - mu) / sd

    # PROXY-STRIP (load-bearing, validated on synth): remove time-proxy features (single-feature
    # early/late AUC>0.70 AND ~no predictive value) from the MODEL space AND from D. A drifting
    # nuisance feature both inflates D->1 and BLINDS the staleness model (nuisance made staleness
    # read +0.0004; stripping recovered +0.236). Real (predictive) covariate drift is kept.
    keep_px = _proxy_mask(X, y, t, task)
    n_proxy = int((~keep_px).sum())
    X_full = X
    if not keep_px.all() and keep_px.any():
        X = X[:, keep_px]

    win, Keff = _assign_windows(t, K, by_value)            # single windowing, for diagnostics + D
    keep_w = win >= 0
    Xd, yd, td, win = X[keep_w], y[keep_w], t[keep_w], win[keep_w]
    # separability on stripped (D_strip, gate) and full (D_full) space. v3: group-aware split +
    # random size-matching (audit F2/L3) and a multi-seed median instead of a seed-0 point
    # estimate (audit L2). D_full>=D* but D_strip<D* => PROXY-SENSITIVE (a clock, not covariate).
    groups = _row_groups(Xd)
    D_list = [v for v in (_disjointness(Xd, win, Keff, seed=s, groups=groups)
                          for s in range(D_SEEDS)) if v is not None]
    D_strip = float(np.median(D_list)) if D_list else None
    D_spread = ([float(np.min(D_list)), float(np.max(D_list))] if D_list else [None, None])
    dup_frac = float(np.mean(np.bincount(groups) [groups] > 1))   # rows sharing a near-dup group
    D_full = _disjointness(X_full[keep_w], win, Keff, seed=0, groups=_row_groups(X_full[keep_w]))
    # time-shuffle control (now ALWAYS computed, any branch may earn the flag — audit L3: v2 only
    # ran it in the UNIDENTIFIABLE branch, so sberbank's 0.887 went unflagged). With the random
    # size-matching fix a high D_shuffle can no longer be the head-truncation artifact.
    sh = np.random.default_rng(0).permutation(len(td))
    wsh, Ksh = _assign_windows(td[sh], Keff, False)
    D_shuffle = _disjointness(Xd, wsh, Ksh, seed=0, groups=groups)
    min_window_n = int(min((int((win == k).sum()) for k in range(Keff) if (win == k).any()),
                           default=0))                        # NEW-4: smallest window row count

    # ---- domain guards / trust diagnostics (so a degenerate time axis or pure serial
    #      correlation can't be read as a real deployment verdict) ----
    uniq_t = int(np.unique(td).size)
    cov_el = None                                          # covariate movement first->last window
    try:
        fw, lw = Xd[win == win.min()], Xd[win == win.max()]
        if len(fw) >= 20 and len(lw) >= 20:
            from src.analysis.drift_measure import covariate_shift_auc
            cov_el = covariate_shift_auc(fw, lw).get("auc")
    except Exception:
        cov_el = None
    ys = yd[np.argsort(td, kind="stable")].astype(float)   # label autocorrelation in time order
    ylag = (float(np.corrcoef(ys[1:], ys[:-1])[0, 1])
            if ys.size > 2 and np.std(ys) > 0 else None)
    # tie-overlap: fraction of adjacent windows whose [min_t,max_t] overlap (a tied timestamp split
    # across the past/future boundary -> potential leakage the few-unique-times guard misses, Flaw 5)
    rngs = [(td[win == k].min(), td[win == k].max()) for k in range(Keff) if (win == k).any()]
    tie_ov = (sum(1 for a, b in zip(rngs, rngs[1:]) if a[1] >= b[0]) / max(len(rngs) - 1, 1)
              if len(rngs) > 1 else 0.0)

    decay_s, rec_s, stale_s, den_s, ratio_s = [], [], [], [], []
    for s in range(seed_base, seed_base + n_seeds):
        out = _per_seed(X, y, t, task, K, by_value, max_train, s)
        if out is None:
            continue
        Keff = out["Keff"]
        decay_s.append(out["decay"]); rec_s.append(out["rec"]); stale_s.append(out["stale"])
        den_s.append(out["den"])
        if out["noise_old"] is not None and out["noise_rec"] not in (None, 0):
            ratio_s.append(out["noise_old"] / out["noise_rec"])
    decay, decay_ci = _ci95(decay_s)
    rec, rec_ci = _ci95(rec_s)
    stale, stale_ci = _ci95(stale_s)
    den, den_ci = _ci95(den_s)
    noise_ratio, noise_ratio_ci = _ci95(ratio_s)

    # ---- trust flags (informational; never override the verdict) ----
    flags = []
    if uniq_t < Keff:
        flags.append("few-unique-times")                   # can't form the windows we claim
    if cov_el is not None and cov_el < 0.55:
        flags.append("no-covariate-movement")              # the 'time' axis barely moves
    if task == "binclass" and ylag is not None and abs(ylag) > 0.5:
        flags.append("autocorr-risk")                      # serial labels (elec2) fake decay/recency
    if decay is not None and decay < -FLOOR_DECAY:
        flags.append("future-easier")                      # negative decay -> time axis may not be chronological
    if tie_ov > 0.3:
        flags.append("tie-split")                          # tied timestamps straddle window boundaries
    if n_proxy:
        flags.append(f"proxy-stripped:{n_proxy}")

    # v3 fire rules. Rule A (pre-committed since e680960, applied to every arm uniformly):
    # CI-lower > 0 AND mean > floor. Rule B (strict sensitivity reading): CI-lower > floor.
    # Both are always reported (verdict / verdict_strict) — audit L1.
    fire_a = lambda m, ci: (m is not None and ci[0] is not None
                            and ci[0] > 0 and m > FLOOR_GAIN)
    fire_b = lambda m, ci: (m is not None and ci[0] is not None and ci[0] > FLOOR_GAIN)

    rec_present = fire_a(rec, rec_ci)
    raw_fire, den_fire = fire_a(stale, stale_ci), fire_a(den, den_ci)
    gate_fired = noise_ratio is not None and noise_ratio > GATE_THRESH
    env_exceeded = noise_ratio is not None and noise_ratio > GATE_ENVELOPE
    den_sub = (den is not None and den_ci[0] is not None and den_ci[0] > 0
               and not fire_a(den, den_ci))                           # 0 < denoised lower_CI <= floor
    stale_null = (stale is not None and stale_ci[1] is not None
                  and stale_ci[1] <= FLOOR_GAIN)                      # raw upper_CI <= floor
    den_null = den is None or (den_ci[1] is not None and den_ci[1] <= FLOOR_GAIN)
    disjoint = D_strip is not None and D_strip >= DSTAR
    if D_full is not None and D_strip is not None and D_full >= DSTAR and D_strip < DSTAR:
        flags.append("proxy-sensitive")                    # high separability was a clock, not covariate
    if den is None and stale is not None:
        flags.append("denoiser-failed")                    # cross-fit failed -> CONCEPT unreachable
    n_stale_ok = len([v for v in stale_s if v is not None])
    measured = (decay is not None and rec is not None and stale is not None
                and n_stale_ok >= 2)                       # count non-None seeds (audit landmine fix)

    def _cascade(rf, df):
        """v3 decision cascade (frozen in PREREG_DEPLOYMENT_V2.md). CONCEPT requires the DENOISED
        arm — the raw arm alone is not sound under label-noise drift (audit F1, executed)."""
        if not measured:
            return "NO-DATA"
        if df and not env_exceeded:
            return "DEPLOYMENT-CONCEPT"
        if df:
            return "NOISE-AMBIGUOUS"                       # denoiser bias zone -> abstain
        if rf and gate_fired:
            return "NOISE-DRIFT-CONFOUNDED"                # label-noise drift, not a rule change
        if rf:
            return "RAW-ONLY-POSITIVE"                     # unresolved; NOT concept under repair
        if disjoint:
            return "UNIDENTIFIABLE-EXPLOITABLE" if rec_present else "UNIDENTIFIABLE-INERT"
        if den_sub:
            return "SUBFLOOR-CONCEPT-SIGNAL"
        if stale_null and den_null:
            return "DEPLOYMENT-DECAY-COVARIATE" if rec_present else "NO-STRONG-CONCEPT"
        return "INCONCLUSIVE"

    verdict = _cascade(raw_fire, den_fire)
    verdict_strict = _cascade(fire_b(stale, stale_ci), fire_b(den, den_ci))   # rule-B sensitivity
    if verdict == "DEPLOYMENT-CONCEPT":
        if gate_fired:
            flags.append("noise-drift-present")            # concept + noise drift coexist (B4a)
        if disjoint:
            flags.append("d-gate-invalid")                 # fired where the gate declared blindness

    # ---- injection control. v3: learnability-gated, n_seeds=INJ_SEEDS, runs on the CONCEPT
    #      branch too as a positive control (v2 left the only positive cell uncontrolled).
    inj_stale = inj_learnable = inj_lscore = None; inj_feats = None
    inj_ci = None; inj_recovered_strict = None
    _needs_inj = lambda v: v.startswith("UNIDENTIFIABLE") or v == "DEPLOYMENT-CONCEPT"
    # PREREG §15: injection also runs when only the STRICT verdict routes to it (e.g. rule A reads
    # NOISE-DRIFT-CONFOUNDED but rule B lands UNIDENTIFIABLE) so the shadow cascade is complete.
    if measured and (_needs_inj(verdict) or _needs_inj(verdict_strict)):
        recovered, inj_stale, inj_ci, inj_learnable, inj_lscore, inj_feats = _injection_recovers(
            X, t, task, K, by_value, max_train, seed_base=seed_base, family=inj_family)
        if verdict.startswith("UNIDENTIFIABLE"):
            if not inj_learnable:
                flags.append("injection-vacuous")          # unlearnable probe: the null proves nothing
            elif recovered:                                # geometry HAD power -> real null informative
                verdict = "INJECTION-RECOVERED"; flags.append("injection-recovered")
            else:                                          # blindness demonstrated on real geometry
                flags.append("unident-earned")
        elif verdict == "DEPLOYMENT-CONCEPT":              # CONCEPT positive control
            if not inj_learnable:
                flags.append("injection-vacuous")
            elif not recovered:
                flags.append("injection-no-recover-on-concept")
        # strict shadow traverses the same stage under its own rule (B: CI lower bound > floor);
        # before this, any INJECTION-RECOVERED cell trivially diverged from its shadow (the gap
        # behind the cooking/delivery rule-sensitive mislabels, PREREG §15).
        inj_recovered_strict = (inj_ci is not None and inj_ci[0] is not None
                                and inj_ci[0] > FLOOR_GAIN)
        if verdict_strict.startswith("UNIDENTIFIABLE") and inj_learnable and inj_recovered_strict:
            verdict_strict = "INJECTION-RECOVERED"
    if D_shuffle is not None and D_shuffle > 0.6:          # unconditional (audit L3: was branch-scoped)
        flags.append("d-gate-suspect")
    if min_window_n and min_window_n < ROW_FLOOR:
        flags.append(f"sparse-window:{min_window_n}")      # NEW-4
    trust = "ok" if not flags else ";".join(flags)
    return {"dataset": name, "task": task, "n": int(len(y)), "n_windows": int(Keff),
            "n_seeds_ok": n_stale_ok,
            "decay": decay, "decay_ci": decay_ci,
            "recency_gain": rec, "recency_gain_ci": rec_ci,
            "staleness_harm": stale, "staleness_harm_ci": stale_ci,
            "denoised_staleness": den, "denoised_staleness_ci": den_ci,
            "noise_ratio": noise_ratio, "noise_ratio_ci": noise_ratio_ci,
            "noise_gate_fired": bool(gate_fired), "noise_envelope_exceeded": bool(env_exceeded),
            "gate_thresh": GATE_THRESH, "gate_envelope": GATE_ENVELOPE,
            "D_strip": D_strip, "D_spread": D_spread, "D_full": D_full, "D_shuffle": D_shuffle,
            "dup_group_frac": dup_frac,
            "injected_staleness": inj_stale, "injected_staleness_ci": inj_ci,
            "injection_learnable": inj_learnable,
            "injection_recovered_strict": inj_recovered_strict,
            "injection_learn_score": inj_lscore, "injection_features": inj_feats,
            "injection_family": inj_family,
            "delta_staleness": _delta(stale_s), "n_proxy_stripped": n_proxy,
            "min_window_n": min_window_n, "Dstar": DSTAR,
            "n_unique_t": uniq_t, "cov_auc_early_late": cov_el,
            "y_lag1_autocorr": ylag, "tie_overlap": float(tie_ov), "trust": trust,
            "verdict": verdict, "verdict_strict": verdict_strict,
            # per-seed raw values (reviewer request): enables alternative CI constructions
            # (block bootstrap, jackknife, split-half) downstream without a rerun.
            "per_seed": {"stale": stale_s, "den": den_s, "decay": decay_s, "rec": rec_s,
                         "noise_ratio": ratio_s}}


# ----------------------------------------------------------------------------- loaders
def _load_csv(path, target, time, drop, max_n):
    import pandas as pd
    from pandas.api import types as pt
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df = df.dropna(subset=[target, time])
    if max_n and len(df) > max_n:                       # keep time order; even thinning
        df = df.sort_values(time).iloc[:: max(1, len(df) // max_n)].reset_index(drop=True)

    def to_num(col):
        if pt.is_numeric_dtype(col):
            return pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")
        if pt.is_datetime64_any_dtype(col):
            return col.view("int64").to_numpy().astype("float64")
        return col.astype("category").cat.codes.to_numpy().astype("float64")

    y_raw = df[target]
    tnum = to_num(df[time])
    drop_cols = set([target, time] + list(drop or []))
    feats = [c for c in df.columns if c not in drop_cols]
    X = np.column_stack([to_num(df[c]) for c in feats])
    nuniq = len(pd.unique(y_raw.dropna()))
    task = ("binclass" if nuniq == 2 else
            "regression" if (pt.is_float_dtype(y_raw) and nuniq > 20) else "multiclass")
    if task != "regression":
        cls = {c: i for i, c in enumerate(sorted(pd.unique(y_raw.dropna())))}
        y = np.array([cls.get(v, 0) for v in y_raw.to_numpy()])
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype="float64")
    return X, y, tnum, task, len(feats)


def _load_tabred(ds, cfg, span="train"):
    from src.data.tabred_loader import load_tabred
    from src.analysis.drift_measure import _stack
    data = load_tabred(ds, Path(cfg.data.root), split=cfg.experiment.split)
    if span == "full":
        # train+val+test concatenated on the shared normalized timestamp — the windows then span
        # the held-out deployment gap the official split holds out (reviewer-2 R1). Split labels
        # are irrelevant to the retrospective audit; time order is re-established by sort.
        parts = [data.train, data.val, data.test]
        X = np.concatenate([_stack(p.X_num, p.X_bin, p.X_cat) for p in parts], axis=0)
        y = np.concatenate([p.y for p in parts])
        t = np.concatenate([np.asarray(p.t, float) for p in parts])
        o = np.argsort(t, kind="stable")
        return X[o], y[o], t[o], data.task, X.shape[1]
    # default: the TRAIN portion (carries the within-train temporal axis); t = its timestamp
    X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    return X, data.train.y, data.train.t, data.task, X.shape[1]


def _load_stream(data):
    """elec2 / insects: the natural deployment timeline is the WHOLE stream, so concatenate
    train+val+test and order by t_raw (the global stream index each split carries). t_raw is the
    time axis for windowing (a clean monotone order)."""
    from src.analysis.drift_measure import _stack
    parts = [data.train, data.val, data.test]
    X = np.concatenate([_stack(p.X_num, p.X_bin, p.X_cat) for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts])
    tr = np.concatenate([p.t_raw for p in parts]).astype(float)
    o = np.argsort(tr, kind="stable")
    return X[o], y[o], tr[o], data.task, X.shape[1]


# ----------------------------------------------------------------------------- synthetic ground truth
def _synth(kind, seed=0, n=12000, d=10):
    """Three regimes the instrument MUST separate, over a continuous time axis (-> quantile windows):
       concept   : rule P(y|x) rotates with t, covariates stationary  -> DEPLOYMENT-CONCEPT
       covariate : rule FIXED, covariate mean drifts with t           -> DEPLOYMENT-DECAY-COVARIATE
       stable    : rule fixed, covariates stationary                  -> DEPLOYMENT-STABLE
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d))
    if kind == "concept":
        ang = 3.0 * t                                   # decision rule rotates over time
        score = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1]
        y = (3 * score + rng.normal(0, .4, n) > 0).astype(int)
    elif kind == "covariate":
        # P(x) drifts (X0 mean ramps out of the early support) but the rule in OBSERVED-feature
        # space is FIXED and NONLINEAR -> the old model must EXTRAPOLATE on the new region
        # (decay + recency_gain), yet old samples carry CORRECT labels -> staleness_harm <= 0.
        X[:, 0] = X[:, 0] + 6.0 * t
        y = (np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, .3, n) > 0).astype(int)
    elif kind == "nuisance_proxy":
        # strong concept on X0,X1 (stationary) + a DRIFTING NUISANCE feature X2 (not in the rule).
        # Without proxy-strip the nuisance inflates D->1 and blinds the model -> wrongly reads
        # UNIDENTIFIABLE/no-concept; WITH strip it must read DEPLOYMENT-CONCEPT. Validates the strip.
        X[:, 2] = X[:, 2] + 8.0 * t
        ang = 2.5 * t
        y = (np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, .3, n) > 0).astype(int)
    elif kind == "covariate_mild":
        # IDENTIFIABLE covariate drift (ramp small enough that D < D*): the staleness null must be
        # load-bearing here, not gated away — validates the regime that matters for any real
        # borderline positive (audit exp-synth-repro probe: NO-STRONG-CONCEPT at 2 generator seeds).
        X[:, 0] = X[:, 0] + 1.5 * t
        y = (np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, .3, n) > 0).astype(int)
    elif kind == "reg_stable":
        y = 3 * X[:, 0] + rng.normal(0, 0.3, n)
        return X, y, t, "regression"
    elif kind == "reg_concept":
        ang = 3.0 * t
        y = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, 0.3, n)
        return X, y, t, "regression"
    elif kind == "reg_cov_linear":                      # fixed LINEAR rule, drifting X
        X[:, 0] = X[:, 0] + 6.0 * t
        y = 3 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 0.3, n)
        return X, y, t, "regression"
    elif kind == "reg_cov_nonlinear":                   # fixed NONLINEAR rule, drifting X (sberbank twin)
        X[:, 0] = X[:, 0] + 6.0 * t
        y = np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, 0.3, n)
        return X, y, t, "regression"
    elif kind == "reg_early_noisy":
        # THE F1 KILLER (audit, executed): fixed conditional mean, label-noise std 1.5 -> 0.3.
        # v2 minted DEPLOYMENT-CONCEPT at +0.0214 here — matching sberbank's +0.0239. v3 must file
        # NOISE-DRIFT-CONFOUNDED (raw fires, denoised null +0.0037, gate ~3.5 fires).
        sd = 1.5 - 1.2 * t
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    elif kind == "reg_late_noisy":                      # noise GROWS 0.3 -> 1.5 (direction v2 passed)
        sd = 0.3 + 1.2 * t
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    elif kind == "reg_xdep_noise":
        # x-DEPENDENT noise whose scale decays — second raw false-positive channel found by the
        # audit (+0.0254 raw under a fixed mean); denoised must stay null.
        sd = (1.5 - 1.2 * t) * (0.5 + 1.0 / (1.0 + np.exp(-2.0 * X[:, 1])))
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    elif kind == "reg_concept_earlynoisy":              # concept + noise decay simultaneously:
        ang = 3.0 * t                                   # must STILL read CONCEPT (gate must not veto)
        sd = 1.5 - 1.2 * t
        y = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, 1, n) * sd
        return X, y, t, "regression"
    elif kind == "covariate_mc":
        # ADVERSARIAL control for the insects (multiclass / accuracy) read: P(x) drifts AND the
        # class PRIOR drifts with it, but P(y|x) is FIXED in observed-feature space. A true concept
        # discriminator must NOT call this CONCEPT — if staleness_harm fires here, the multiclass/
        # accuracy path is confounding prior-shift with a changed rule (the insects worry).
        X[:, 0] = X[:, 0] + 6.0 * t                     # covariate drift -> also shifts class prior
        logits = np.stack([np.sin(1.5 * X[:, 0]), 0.8 * X[:, 1], -0.7 * X[:, 2]], axis=1)
        y = (logits + rng.normal(0, .3, (n, 3))).argmax(1)        # FIXED 3-class rule
        return X, y, t, "multiclass"
    else:                                               # stable
        y = (3 * X[:, 0] + rng.normal(0, .4, n) > 0).astype(int)
    return X, y, t, "binclass"


# ----------------------------------------------------------------------------- model-class shims
def _apply_model_class(kind):
    """PREREG Phase 3: swap the probe model class module-wide (staleness, denoiser, D, injection
    all follow). Ported verbatim from the audited model-class matrix (audit F3; shims validated
    there: Pipeline delegates classes_; HGB-style kwargs absorbed). 'linear'/'knn' are CANARIES —
    they false-fire CONCEPT under fixed rules by design (misspecification channel); only
    tree-ensemble verdicts (hgb, rf) are decision-grade."""
    if kind == "hgb":
        return
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    def build(is_clf, seed):
        if kind == "linear":
            base = (LogisticRegression(max_iter=2000, random_state=seed) if is_clf
                    else Ridge(random_state=seed))
            steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", base)]
        elif kind == "rf":
            base = (RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1) if is_clf
                    else RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1))
            steps = [("imp", SimpleImputer(strategy="median")), ("m", base)]
        elif kind == "knn":
            base = KNeighborsClassifier(n_neighbors=25) if is_clf else KNeighborsRegressor(n_neighbors=25)
            steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", base)]
        elif kind == "mlp":
            # neural probe (reviewer request: the paper's motivation is deep tabular architecture,
            # so the class panel needs at least one NN class). sklearn MLP keeps the panel
            # dependency-free; 2-layer, early-stopping on for stability at 12k rows.
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            base = (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                                  random_state=seed) if is_clf
                    else MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                                      random_state=seed))
            steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", base)]
        else:
            raise ValueError(kind)
        return Pipeline(steps)

    class _Base:
        _is_clf = True

        def __init__(self, **kw):
            seed = kw.get("random_state", 0) or 0
            self._pipe = build(self._is_clf, int(seed))

        def fit(self, X, y):
            self._pipe.fit(np.asarray(X, float), y); return self

        def predict(self, X):
            return self._pipe.predict(np.asarray(X, float))

        def predict_proba(self, X):
            return self._pipe.predict_proba(np.asarray(X, float))

        @property
        def classes_(self):
            return self._pipe.classes_

    globals()["HistGradientBoostingClassifier"] = type(f"Shim_{kind}_clf", (_Base,), {"_is_clf": True})
    globals()["HistGradientBoostingRegressor"] = type(f"Shim_{kind}_reg", (_Base,), {"_is_clf": False})


# river 0.25.0's actual accepted list (server ValueError 2026-07-04 printed exactly these 7;
# the loader's historical 9-name tuple included names this river version rejects).
INSECTS_VARIANTS = ("abrupt_balanced", "abrupt_imbalanced", "gradual_balanced",
                    "gradual_imbalanced", "incremental_balanced",
                    "incremental_abrupt_balanced", "incremental_reoccurring_balanced")


# ----------------------------------------------------------------------------- main
def _run_meta(args):
    """Provenance stamp written into every output blob (audit 10H: the v2 headline artifact had
    no run metadata — rows from different instrument versions were indistinguishable)."""
    import datetime
    import platform
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except Exception:
        sha = None
    import sklearn
    return {"git": sha, "argv": sys.argv[1:],
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "instrument": "v3",
            "model": getattr(args, "model", "hgb"), "seed_base": getattr(args, "seed_base", 0)}


def _show(r):
    f = lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else "   -"
    ci = lambda c: f"[{f(c[0])},{f(c[1])}]" if c and c[0] is not None else "   -"
    trust = r.get("trust", "ok")
    tnote = "" if trust == "ok" else f"  [TRUST: {trust}]"
    dstr = r.get("D_strip"); df = f"{dstr:.3f}" if isinstance(dstr, (int, float)) else "  -"
    nr = r.get("noise_ratio")
    gstr = (f"{nr:.2f}{'!' if r.get('noise_gate_fired') else ''}"
            if isinstance(nr, (int, float)) else "  -")
    print(f"  {r['dataset'][:20]:20s} W={r['n_windows']:>2d} D={df} gate={gstr} "
          f"stale={f(r['staleness_harm'])}{ci(r['staleness_harm_ci'])} "
          f"den={f(r.get('denoised_staleness'))}{ci(r.get('denoised_staleness_ci'))} "
          f"=> {r['verdict']}{tnote}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default=None); ap.add_argument("--time", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--tabred", nargs="*", default=[])
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced",
                    help="one of the 8 designed variants, or 'all'")
    ap.add_argument("--river", nargs="*", default=None,
                    help="river synth panel stream names, or 'all' (PREREG Phase 4 anchors)")
    ap.add_argument("--model", default="hgb", choices=["hgb", "rf", "linear", "knn", "mlp"],
                    help="probe model class (PREREG Phase 3; linear/knn canaries, mlp = NN probe)")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="0 = exploratory (seeds 0..n-1); 100 = confirmatory rerun (PREREG §5)")
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--by-value", action="store_true",
                    help="one window per unique time value (e.g. EMBER YYYYMM months)")
    ap.add_argument("--windows", type=int, default=10, help="quantile windows when not --by-value")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-n", type=int, default=60000)
    ap.add_argument("--tabred-span", default="train", choices=["train", "full"],
                    help="train = official train segment (the paper's map); full = train+val+test "
                         "concatenated by timestamp - audits across the held-out deployment gap "
                         "(reviewer-2 R1). Dataset names get a _fullspan suffix.")
    ap.add_argument("--inj-family", default="topvar",
                    choices=["topvar", "lowvar", "interaction", "subpop"],
                    help="injection reference family (reviewer-2 R3 sweep): topvar = pre-registered "
                         "default; lowvar / interaction / subpop probe certificate sensitivity to "
                         "the planted signal class. Recorded per-row as injection_family.")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--debug-raise", action="store_true",
                    help="re-raise HGB fit errors with full traceback (diagnose the binning crash)")
    args = ap.parse_args()
    if args.debug_raise:
        globals()["_DEBUG_RAISE"] = True
    _apply_model_class(args.model)                      # no-op for hgb (default compute unchanged)
    out_dir = Path("results/phase1/deployment_decay"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.synth:
        MUST_CONCEPT = ("concept", "nuisance_proxy", "reg_concept", "reg_concept_earlynoisy")
        NEVER_CONCEPT = ("covariate", "stable", "covariate_mc", "covariate_mild", "reg_stable",
                         "reg_cov_linear", "reg_cov_nonlinear", "reg_early_noisy",
                         "reg_late_noisy", "reg_xdep_noise")
        MUST_NOISE_CONF = ("reg_early_noisy", "reg_xdep_noise")        # the gate must catch these
        print("\n==== SYNTH ground-truth battery (v3: binclass + regression + noise-drift adversarials) ====")
        print(f"  EXPECT CONCEPT:      {', '.join(MUST_CONCEPT)}")
        print(f"  EXPECT not-CONCEPT:  {', '.join(NEVER_CONCEPT)}")
        print(f"  EXPECT NOISE-DRIFT-CONFOUNDED: {', '.join(MUST_NOISE_CONF)}")
        print("  EXPECT covariate_mild != UNIDENTIFIABLE-* (identifiable-regime coverage)")
        for kind in MUST_CONCEPT + NEVER_CONCEPT:
            X, y, t, task = _synth(kind)
            r = assess(f"synth_{kind}", X, y, t, task, K=args.windows, n_seeds=args.n_seeds,
                       max_train=args.max_train)
            rows.append(r); _show(r)
        blob = {"meta": _run_meta(args), "rows": rows}
        (out_dir / "synth_summary.json").write_text(json.dumps(blob, indent=2, default=float))
        verdicts = {r["dataset"]: r["verdict"] for r in rows}
        C = lambda k: verdicts.get(f"synth_{k}", "")
        # v3 invariants (frozen in PREREG_DEPLOYMENT_V2.md): CONCEPT fires for every true-concept
        # kind incl. concept+noise-decay (gate must not veto); NEVER for any fixed-rule kind incl.
        # both noise-drift adversarials (which must be specifically NOISE-DRIFT-CONFOUNDED); the
        # identifiable mild-covariate cell must be decided by the staleness null, not gated away.
        ok = (all(C(k) == "DEPLOYMENT-CONCEPT" for k in MUST_CONCEPT)
              and all(C(k) != "DEPLOYMENT-CONCEPT" for k in NEVER_CONCEPT)
              and all(C(k) == "NOISE-DRIFT-CONFOUNDED" for k in MUST_NOISE_CONF)
              and not C("covariate_mild").startswith("UNIDENTIFIABLE"))
        print(f"\n  GROUND-TRUTH {'PASS' if ok else 'FAIL (verdicts above must match EXPECT)'}")
        print(f"  wrote {out_dir}/synth_summary.json")
        return

    jobs = []
    if args.csv and args.target and args.time:
        X, y, t, task, nf = _load_csv(args.csv, args.target, args.time, args.drop, args.max_n)
        jobs.append((args.name or Path(args.csv).stem, X, y, t, task, nf))
    if args.tabred:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        for ds in args.tabred:
            X, y, t, task, nf = _load_tabred(ds, cfg, span=args.tabred_span)
            tag = ds if args.tabred_span == "train" else f"{ds}_fullspan"
            jobs.append((tag, X, y, t, task, nf))
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        jobs.append(("elec2", *_load_stream(load_elec2(split="temporal", seed=0))))
    if args.insects:
        from src.data.insects_loader import load_insects
        variants = INSECTS_VARIANTS if args.insects_variant == "all" else (args.insects_variant,)
        for v in variants:                     # one bad variant must not kill the batch
            try:
                jobs.append((f"insects_{v}",
                             *_load_stream(load_insects(variant=v, split="temporal", seed=0))))
            except Exception as e:
                print(f"  [warn] insects variant {v!r} skipped: {type(e).__name__}: {e}",
                      file=sys.stderr)
    if args.river is not None:
        from src.data.river_streams import list_streams, load_river_stream
        names = list_streams() if (not args.river or args.river == ["all"]) else args.river
        for nm in names:
            try:
                jobs.append((f"river_{nm}", *_load_stream(load_river_stream(nm, split="temporal",
                                                                            seed=0))))
            except Exception as e:
                print(f"  [warn] river stream {nm!r} skipped: {type(e).__name__}: {e}",
                      file=sys.stderr)
    if not jobs:
        print("provide --csv --target --time, --tabred ..., --elec2/--insects/--river, or --synth")
        return

    print("\n==== DEPLOYMENT-DECAY probe (rolling-origin: train past -> predict future) ====")
    for name, X, y, t, task, nf in jobs:
        print(f"  [{name}] task={task} feats={nf} n={len(y)}")
        try:
            r = assess(name, X, y, t, task, K=args.windows, by_value=args.by_value,
                       n_seeds=args.n_seeds, max_train=args.max_train,
                       seed_base=args.seed_base, inj_family=args.inj_family)
        except Exception as e:                              # never let one dataset kill the batch
            import traceback; traceback.print_exc()
            r = {"dataset": name, "task": task, "verdict": "ERROR", "error": f"{type(e).__name__}: {e}",
                 "decay": None, "decay_ci": [None, None], "recency_gain": None,
                 "recency_gain_ci": [None, None], "staleness_harm": None,
                 "staleness_harm_ci": [None, None], "n_windows": 0}
        rows.append(r); _show(r)
    print("\n  v3 READ (PREREG_DEPLOYMENT_V2.md): DEPLOYMENT-CONCEPT needs the DENOISED arm within the")
    print("  noise envelope; raw-only positives are NOISE-DRIFT-CONFOUNDED or RAW-ONLY-POSITIVE, never")
    print("  concept. All verdicts are tree-ensemble-class-scoped; D = window separability, not overlap.")
    meta = _run_meta(args)
    stamp = meta["utc"].replace(":", "").replace("-", "")[:15]
    out_file = out_dir / f"summary_{stamp}_{meta['git'] or 'nogit'}.json"
    out_file.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, default=float))
    (out_dir / "summary_latest.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2, default=float))
    print(f"\n  wrote {out_file}  (+ summary_latest.json)  <-- send me this")


if __name__ == "__main__":
    main()
