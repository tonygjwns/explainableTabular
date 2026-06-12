"""Statistical utilities for rigorous model comparison.

Implements the methodology pre-committed in PRE_REGISTRATION.md / EXPERIMENT_PLAN.md §7:
  - Paired Wilcoxon signed-rank test (NOT Welch's t-test; data are paired by seed)
  - Benjamini-Hochberg FDR correction for multiple comparisons
  - Hedges' g effect size (small-sample-corrected Cohen's d)

These are fully implemented and environment-independent (no GPU/data needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


@dataclass
class ComparisonResult:
    """Result of comparing method A vs method B on one dataset."""
    dataset: str
    mean_a: float
    mean_b: float
    delta: float          # mean_a - mean_b (after sign-orientation so higher = better)
    p_value: float        # paired Wilcoxon, raw (pre-FDR)
    hedges_g: float
    n_pairs: int


def orient_higher_is_better(scores: np.ndarray, metric: str) -> np.ndarray:
    """Flip sign for metrics where lower is better (RMSE), so higher is always better.

    Args:
        scores: array of metric values.
        metric: one of {"auc", "accuracy", "rmse"} (case-insensitive).
    Returns:
        Oriented scores where larger = better.
    """
    metric = metric.lower()
    if metric in {"rmse", "mae", "mse", "logloss", "log_loss"}:
        return -np.asarray(scores, dtype=float)
    if metric in {"auc", "roc_auc", "roc-auc", "accuracy", "acc", "r2"}:
        return np.asarray(scores, dtype=float)
    raise ValueError(f"Unknown metric for orientation: {metric}")


def paired_wilcoxon(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Paired Wilcoxon signed-rank test p-value.

    scores_a, scores_b: per-seed scores for the SAME seeds (paired), already
    oriented so higher = better. Returns two-sided p-value.

    Note: if all paired differences are zero, scipy raises; we return 1.0.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diffs = a - b
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        stat, p = wilcoxon(a, b)  # paired form
    except ValueError:
        # e.g. too few non-zero differences
        return 1.0
    return float(p)


def hedges_g(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """UNPAIRED Hedges' g effect size (Cohen's d with small-sample correction).

    Positive g means method A > method B (on oriented scores). Uses pooled SD,
    treating the two as independent samples — for seed-paired comparisons this
    UNDERSTATES the effect; prefer `hedges_g_paired` there (audit 2026-06-12).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pooled == 0:
        return 0.0
    cohen_d = (np.mean(a) - np.mean(b)) / s_pooled
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return float(cohen_d * correction)


def hedges_g_paired(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """PAIRED Hedges' g (d_z with small-sample correction) — use with paired Wilcoxon.

    d_z = mean(a-b) / std(a-b, ddof=1), corrected by J(df=n-1) = 1 - 3/(4(n-1)-1).
    When the two arms share seed/data/init, between-seed variance is common to both,
    so the unpaired pooled-SD `hedges_g` UNDERSTATES the standardized paired effect
    (audit 2026-06-12). Use this wherever the comparison is paired by seed.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n = len(a)
    if n < 2:
        return float("nan")
    d = a - b
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    dz = float(d.mean() / sd)
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return float(dz * correction)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction.

    Returns (reject_flags, p_corrected) where reject_flags[i] is True if the
    i-th hypothesis is rejected at FDR=alpha.
    """
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    reject, p_corr, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return reject, p_corr


def compare_across_datasets(
    scores_a: dict[str, Sequence[float]],
    scores_b: dict[str, Sequence[float]],
    metric_by_dataset: dict[str, str],
    alpha: float = 0.05,
) -> tuple[list[ComparisonResult], np.ndarray]:
    """Compare method A vs B across multiple datasets with FDR correction.

    Args:
        scores_a: {dataset_name: [per-seed scores]} for method A
        scores_b: {dataset_name: [per-seed scores]} for method B (same seeds)
        metric_by_dataset: {dataset_name: metric_name} for orientation
        alpha: FDR level

    Returns:
        (results, reject_flags) where results is a list of ComparisonResult
        (raw p-values) and reject_flags aligns with results after BH correction.
    """
    datasets = list(scores_a.keys())
    results: list[ComparisonResult] = []
    raw_p: list[float] = []
    for ds in datasets:
        metric = metric_by_dataset[ds]
        a = orient_higher_is_better(scores_a[ds], metric)
        b = orient_higher_is_better(scores_b[ds], metric)
        p = paired_wilcoxon(a, b)
        g = hedges_g(a, b)
        results.append(
            ComparisonResult(
                dataset=ds,
                mean_a=float(np.mean(a)),
                mean_b=float(np.mean(b)),
                delta=float(np.mean(a) - np.mean(b)),
                p_value=p,
                hedges_g=g,
                n_pairs=len(a),
            )
        )
        raw_p.append(p)
    reject, _ = benjamini_hochberg(raw_p, alpha=alpha)
    return results, reject


def passes_test1(
    results: list[ComparisonResult],
    reject_flags: np.ndarray,
    min_datasets: int = 3,
    min_hedges_g: float = 0.3,
) -> bool:
    """Sanity-check Test 1 gate (EXPERIMENT_PLAN §8 / PRE_REGISTRATION §3.1).

    PASS if A > B is significant (post-FDR) AND Hedges' g >= threshold in at
    least `min_datasets` datasets, AND A < B is never significant.
    """
    wins = 0
    for res, rej in zip(results, reject_flags):
        if rej and res.delta < 0 and res.hedges_g <= -min_hedges_g:
            # significant negative effect -> automatic FAIL
            return False
        if rej and res.delta > 0 and res.hedges_g >= min_hedges_g:
            wins += 1
    return wins >= min_datasets


if __name__ == "__main__":
    # quick self-test
    rng = np.random.default_rng(0)
    a = {f"ds{i}": rng.normal(0.86, 0.01, size=5) + 0.01 for i in range(4)}
    b = {f"ds{i}": rng.normal(0.86, 0.01, size=5) for i in range(4)}
    metrics = {f"ds{i}": "auc" for i in range(4)}
    res, rej = compare_across_datasets(a, b, metrics)
    for r, j in zip(res, rej):
        print(f"{r.dataset}: delta={r.delta:+.4f} p={r.p_value:.4f} g={r.hedges_g:+.2f} reject={j}")
    print("Test1 PASS:", passes_test1(res, rej))
