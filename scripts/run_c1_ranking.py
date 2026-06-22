"""V3.2 C1 — does the deep/retrieval-vs-GBDT margin on TabReD track our diagnostics? (PLAN_V3 §C1)

Red-team R2/D4: the paper ASSERTS that "covariate dominance ⇒ no concept ⇒ deep methods
can't win" explains TabReD's "simple beats complex", but never DEMONSTRATES it on the
TabReD rankings. This script earns (or honestly retires) the link using TabReD's OWN
published per-dataset scores [Rubachev et al., ICLR 2025, Table 3] — so the tuning budget
is controlled by THEIR comparable-tuning protocol (we do not re-train deep methods).

Per dataset: margin = GBDT advantage over a deep/retrieval method, oriented so + = GBDT
wins, made scale-free (relative to the GBDT score; RMSE & AUC handled by direction).
Join with our cov_AUC / overlap_mass / within-overlap concept_gap and correlate.

Prediction of the thesis: margin should be LARGER where cov_AUC is high / overlap low /
concept ~0. We report Spearman & Pearson honestly. If the correlation is weak/absent
(e.g. a high-covariate dataset where retrieval WINS), §7 must concede the thesis does not
strongly predict the per-dataset rankings -> the puzzle has other drivers (budget,
architecture). NOTE: with public numbers we cannot add a budget/seed-variance covariate;
TabReD's protocol controls budget across methods, but this is a stated limitation.

    python scripts/run_c1_ranking.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402

# TabReD Table 3 (Rubachev et al., ICLR 2025). higher_better per metric.
# dataset: (metric_higher_better, GBDT, MLP, FTT, TabR, MNCA)
TABRED = {
    "homesite_insurance":  (True,  0.9601, 0.9500, 0.9622, 0.9487, 0.9514),  # AUC
    "ecom_offers":         (True,  0.5763, 0.6015, 0.5775, 0.5943, 0.5765),  # AUC
    "homecredit_default":  (True,  0.8670, 0.8545, 0.8571, 0.8501, 0.8531),  # AUC
    "sberbank_housing":    (False, 0.2482, 0.2508, 0.2440, 0.2820, 0.2593),  # RMSE
    "cooking_time":        (False, 0.4826, 0.4820, 0.4820, 0.4828, 0.4825),  # RMSE
    "delivery_eta":        (False, 0.5468, 0.5504, 0.5542, 0.5514, 0.5498),  # RMSE
    "maps_routing":        (False, 0.1616, 0.1622, 0.1625, 0.1639, 0.1625),  # RMSE
    "weather":             (False, 1.4625, 1.5470, 1.5104, 1.4666, 1.5062),  # RMSE
}


def rel_margin(hb, gbdt, other):
    """+ = GBDT better; relative to GBDT score (scale-free, metric-direction aware)."""
    raw = (gbdt - other) if hb else (other - gbdt)   # + = GBDT better in both cases
    return raw / abs(gbdt)


def main():
    diag = {r["dataset"]: r for r in
            json.load(open(Path("disde_degeneration_summary.json"), encoding="utf-8"))["rows"]}
    rows = []
    print("\n==== C1: TabReD margin vs our diagnostics (public TabReD Table 3) ====")
    print(f"  {'dataset':22s}{'cov_AUC':>8s}{'overlap':>8s}{'gap':>7s} | "
          f"{'GBDT-TabR':>10s}{'GBDT-MLP':>9s}{'GBDT-bestDL':>12s}")
    for ds, (hb, g, mlp, ftt, tabr, mnca) in TABRED.items():
        d = diag.get(ds, {})
        cov = d.get("cov_auc"); ov = d.get("disde", {}).get("overlap_mass")
        gap = d.get("concept_gap")
        m_tabr = rel_margin(hb, g, tabr)
        m_mlp = rel_margin(hb, g, mlp)
        best_dl = max([mlp, ftt, tabr, mnca]) if hb else min([mlp, ftt, tabr, mnca])
        m_dl = rel_margin(hb, g, best_dl)
        gt = f"{gap:+.3f}" if isinstance(gap, (int, float)) else "  -"
        print(f"  {ds:22s}{cov:8.3f}{ov:8.3f}{gt:>7s} | {m_tabr:+10.4f}{m_mlp:+9.4f}{m_dl:+12.4f}")
        rows.append({"dataset": ds, "cov_auc": cov, "overlap_mass": ov, "concept_gap": gap,
                     "margin_gbdt_tabr": m_tabr, "margin_gbdt_mlp": m_mlp,
                     "margin_gbdt_bestDL": m_dl})

    cov = np.array([r["cov_auc"] for r in rows])
    ov = np.array([r["overlap_mass"] for r in rows])
    print("\n  ==== correlations (margin vs diagnostic, n=8 TabReD) ====")
    for mkey, mlab in [("margin_gbdt_tabr", "GBDT-TabR (retrieval collapse)"),
                       ("margin_gbdt_bestDL", "GBDT-bestDeep")]:
        m = np.array([r[mkey] for r in rows])
        rs_c, ps_c = spearmanr(cov, m); rp_c, pp_c = pearsonr(cov, m)
        rs_o, ps_o = spearmanr(ov, m)
        print(f"  [{mlab}]")
        print(f"     Spearman(cov_AUC, margin)={rs_c:+.3f} (p={ps_c:.3f})  "
              f"Pearson={rp_c:+.3f} (p={pp_c:.3f})  [thesis: POSITIVE]")
        print(f"     Spearman(overlap, margin)={rs_o:+.3f} (p={ps_o:.3f})  [thesis: NEGATIVE]")

    print("\n  READ: ecom_offers is the key test — cov_AUC=1.0 (max covariate) yet TabR BEATS")
    print("  GBDT there (negative margin). If cov_AUC does NOT robustly predict the margin,")
    print("  s7 must concede the thesis explains the COMPARATIVE result but does NOT predict")
    print("  per-dataset TabReD rankings -> the puzzle has other drivers (budget/architecture).")
    out = Path("results/phase1/c1_ranking"); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
    print(f"\n  wrote {out}/summary.json")


if __name__ == "__main__":
    main()
