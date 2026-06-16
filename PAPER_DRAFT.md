# When Is Concept Drift Even Measurable? Covariate Dominance, the Limits of Time-Aware Tabular Models, and a Validated Diagnostic Toolkit

> DRAFT v0.1 (2026-06-17). Target: NeurIPS Datasets & Benchmarks (workshop-ready now).
> Language: English (submission). Verified numbers from RESULTS.md §1–16. [PENDING]/[FUTURE]
> tags mark not-yet-done items. Honest scope: Claim A leads; Claim B supports; the Cai&Ye
> adjudication is definitional (empirical reproduction is future work).

---

## Abstract

On tabular temporal benchmarks, simple models (GBDTs, MLP+time-feature) routinely match or
beat elaborate time-aware deep methods — a well-known but under-explained puzzle. We give a
mechanistic explanation grounded in *measurement*. First, we show that distribution shift in
real tabular temporal data is **overwhelmingly covariate** (P(x)): on 5 of 8 TabReD datasets a
past-vs-future classifier reaches AUC ≈ 1.0 and the early/late input supports become **disjoint**,
so concept drift (a change in P(y|x)) is **not identifiable** by the standard conditional or
importance-reweighting lens — including DISDE, whose density-ratio estimator degenerates
(effective sample size collapses to <1%). Second, we introduce a **within-overlap model-transfer**
frame that measures concept on the common support where it remains identifiable, and validate it
on a controlled covariate×concept synthetic grid: it recovers planted concept (Spearman 1.0 with
ground truth), emits ~0 where there is none, and **abstains** when support vanishes. On real data
it yields a clean three-way picture: high-covariate datasets are *unmeasurable*; low-covariate
datasets are *measurable but concept-free*; only Elec2 (+0.132 AUC) and INSECTS (+0.144 acc) carry
*measurable, substantial* concept. Third, where concept is measurable, we test whether a
time-indexed retrieval **structure** beats simply feeding time as a **feature**, under a confound-
controlled protocol (25 seeds, paired): it does **not** (−0.007 and −0.021, both 95% CI < 0), and
the mechanism is an *in-distribution* device (it helps under random splits, hurts under temporal
extrapolation). Finally, we reconcile recent positive results: a state-of-the-art temporal
modulation is **label-free by construction**, hence a covariate (X-side) adaptation that cannot
exploit concept — its gains do not contradict our account. We release the diagnostic toolkit.

---

## 1. Introduction

Tabular data under temporal distribution shift is the setting of most deployed ML (credit, demand,
pricing). The TabReD benchmark [Rubachev et al., ICLR 2025] showed that under time-based splits,
retrieval- and pretraining-based deep methods collapse while GBDTs and MLPs survive. *Why* simple
methods win has been attributed to protocol artifacts and missing periodicity [Cai & Ye, ICML 2025]
or addressed by feature-statistic modulation [Cai & Ye, NeurIPS 2025], but the **nature of the
shift itself** — covariate (P(x)) vs concept (P(y|x)) — has not been measured on these benchmarks.

We argue the puzzle is, at root, a **measurement** problem. Time-aware structure can only pay off
if the *rule* P(y|x) changes over time (concept drift); if only the *inputs* change (covariate),
adapting features suffices and a time feature already captures it. We ask: **does exploitable
concept drift exist on tabular temporal benchmarks, and can it even be measured?**

**Contributions.**
1. **A measurement frame** (within-overlap model-transfer gap) that identifies concept on the
   early/late common support, plus a diagnostic of *when standard reweighting (DISDE) degenerates*.
2. **Ground-truth validation** of the toolkit on a controlled covariate×concept synthetic grid,
   including its failure mode (abstention under disjoint support).
3. **An empirical finding** across 10 datasets: covariate dominance makes concept unmeasurable on
   high-covariate TabReD datasets; where measurable, it is ~0 (cooking, maps) except on Elec2
   (+0.132) and INSECTS (+0.144).
4. **A confound-controlled negative**: where concept is measurable, time-indexed retrieval
   *structure* does not beat a time *feature* (paired, 25 seeds), and is an in-distribution (not
   extrapolation) device.
5. **An adjudication** showing a recent positive (feature modulation) is label-free, hence X-side
   by construction — consistent with, not counter to, our account.
6. **A released toolkit** (covariate AUC, DISDE-degeneration, within-overlap transfer gap).

## 2. Related Work

**Shift decomposition.** DISDE [Cai, Namkoong, Yadlowsky, *Operations Research* 2025] attributes a
performance drop to (i) harder seen examples, (ii) a within-overlap Y|X change, (iii) unseen-region
X-shift, via a shared overlap distribution and density ratios. Our within-overlap gap is a
**temporal, model-transfer adaptation** of (ii); we additionally characterize *where DISDE's
estimator degenerates* and remains usable. WhyShift [Liu et al., NeurIPS 2023] decomposes shift on
5 *spatial* tabular datasets and finds Y|X-shift dominant — the **opposite axis** to our temporal
finding (a complementary contrast, not a conflict). The streams literature formalized covariate vs
conditional drift long ago [Gama et al. 2014; Webb et al. 2016].

**Tabular temporal methods.** TabReD [Rubachev et al., ICLR 2025] characterizes its shift only
holistically (ensemble-variance proxy) and does **not** decompose X vs Y|X. Cai & Ye [ICML 2025]
attribute failures to training lag / validation bias and add Fourier time embeddings; Cai & Ye
[NeurIPS 2025] modulate feature statistics over time and beat baselines on TabReD. Drift-Resilient
TabPFN [Helli et al., NeurIPS 2024] *succeeds* with an SCM mechanism-shift prior — consistent with
our discussion that design helps only by encoding assumptions where concept is present.

**Time-aware retrieval.** TabR [Gorishniy et al., ICLR 2024] is retrieval over instances with no
time coordinate. Time-aware instance selection exists in streaming (FISH [Žliobaitė 2011], SAM-kNN
[Losing et al. 2016]); we scope our "empty intersection" to *modern differentiable deep tabular*
retrieval. Elec2's autocorrelation critique [Žliobaitė 2013] (a no-change baseline ≈85%) is
reported with our results.

## 3. The Measurement Framework

**Within-overlap model-transfer gap.** To measure whether P(y|x) changed early→late without
contaminating by covariate extrapolation, we (i) fit an out-of-fold time classifier P(late|x) and
restrict to the overlap band P(late|x) ∈ [0.1, 0.9]; (ii) on a *fixed late-overlap test set*,
compare a model trained on early-overlap data (AUC_early) vs late-overlap data (AUC_late);
gap = AUC_late − AUC_early. Same test, same input region ⇒ difficulty and input distribution are
controlled, so the gap is concept, not covariate. (iii) We check stability across P(late|x)
tertiles (residual-covariate test).

**DISDE-degeneration diagnostic.** DISDE estimates within-support terms by reweighting source with
density ratios w(x)=P(late|x)/P(early|x). We quantify when this degenerates: **ESS** = (Σw)²/Σw²
(heavy-tail/variance mode) and **overlap_mass** = fraction with P(late|x)∈[0.1,0.9] (disjoint-
support/bias mode; ESS is misleading under perfect separation as weights collapse to the clip floor).

**Toolkit.** covariate_shift_auc (with drop-top-k pervasiveness), disde_iw_degeneration, and
concept_within_overlap. Model-light (gradient-boosted trees only). Released at [URL].

## 4. Validation on Ground Truth

We generate synthetic early/late data with two independently controlled knobs: mu_cov (covariate
shift along *non-rule* dimensions) and theta (rotation of the decision rule = concept). On a 4×4
grid the toolkit passes all four checks: (1) **recovery** Spearman(theta, gap)=+1.0 at low
covariate; (2) **no false positive** max|gap| over theta=0 cells = 0.002; (3) **degeneration
monotone** Spearman(mu, cov_AUC)=+1.0, Spearman(mu, overlap_mass)=−1.0; (4) **failure mode**: at
mu=3.0 (overlap 0.002) it reports *unmeasurable* for all theta — abstaining rather than emitting a
false concept. Crucially, at mu=0.70 the DISDE reweighting is effectively dead (ESS = 2.33%), yet
the within-overlap gap recovers the planted concept identically to the zero-covariate row — a
controlled proof of the real-data Elec2 phenomenon. (RESULTS §14.)

## 5. The Three-Way Dichotomy on Real Data

Across TabReD (8) + Elec2 + INSECTS (early/late by median train t):

| dataset | cov_AUC | overlap | ESS% | n_overlap | concept_gap | regime |
|---|---|---|---|---|---|---|
| sberbank / homesite / ecom / homecredit / weather | 1.00 | 0.000 | — | 0 | — | **unmeasurable (disjoint)** |
| delivery | 0.997 | 0.061 | 0.21 | 568 | −0.048 | heavy-tail degenerate |
| cooking | 0.753 | 0.880 | 44.9 | 16960 | −0.005 | measurable, **concept ≈ 0** |
| maps | 0.566 | 1.000 | 93.1 | 20000 | −0.003 | measurable, **concept ≈ 0** |
| **Elec2** | 0.993 | 0.438 | 0.55 | 4721 | **+0.132** | measurable concept (DISDE degenerate) |
| **INSECTS** (incremental) | 0.707 | 0.973 | 39.3 | 19000 | **+0.144** | measurable concept |

Three regimes: (i) high-covariate ⇒ disjoint support ⇒ concept **unmeasurable** by the conditional
lens; (ii) low-covariate ⇒ measurable but **concept ≈ 0**; (iii) genuine concept (Elec2, INSECTS),
where Elec2 needs the within-overlap frame because DISDE reweighting collapses (ESS 0.55%). INSECTS
is a *designed* concept-drift stream (ground truth = drift exists), and the frame recovers it large
(+0.144) — a real-data validation mirroring §4. (RESULTS §13.)

## 6. Does Time-Structured Retrieval Help Where Concept Exists?

We test, on the measurable-concept datasets, whether a time-indexed retrieval **structure**
(time-TabR) beats a time **feature** (MLP+τ(t)). To remove confounds we use five arms sharing one
encoder — mlp_t, tabr, **tabr_t** (retrieval + direct time feature), **time_tabr_t** (retrieval +
time hooks + direct feature) — so the **primary contrast time_tabr_t − tabr_t** isolates the
*structure* with the time feature held present in both arms. Non-degenerate value hook, scaled
similarity + key projection, full-train eval context, val-fair selection, 25 seeds, paired.

**Result (temporal split, paired 95% CI):**
- INSECTS incremental: −0.0067 [−0.012, −0.001], p=.006
- INSECTS incremental_abrupt: −0.0205 [−0.034, −0.008], p<.001

Both **significantly negative**: the structure does not beat the feature; it costs a little.
Decomposition: the retrieval substrate is now competitive (tabr_t − mlp_t ≈ 0; the pre-fix −0.038
deficit is gone), and time helps retrieval (time_tabr_t − tabr > 0), but the *feature* carries time
at least as well. **In-distribution vs extrapolation flip**: the time hook *helps* on random splits
(+0.005, +0.021) and *hurts* on temporal splits (−0.007, −0.021) — it is an in-distribution device,
not an extrapolation device, directly consistent with a representational-redundancy account (a
time-input network already represents in-distribution P(y|x,t)). (RESULTS §12.)

## 7. Adjudicating Recent Positive Results

Cai & Ye [NeurIPS 2025] modulate feature statistics over time and report beating baselines on
TabReD, calling the target "concept drift". Their modulation is x ↦ γ(t)·YeoJohnson(x, λ(t)) + β(t)
with γ, β, λ linear in a time embedding — **label-free**: no term depends on y. By construction it
is a time-indexed *covariate* normalization and **cannot** exploit a P(y|x) change; what they term
"concept drift" is feature-distribution (X-side) drift. Their positive results are therefore
**subsumed** by our account: time-aware methods can win, but by covariate adaptation, not by
exploiting concept (which is unmeasurable where the gains are largest). **[PENDING]** An empirical
reproduction with their tuned pipeline (to localize the gain to the X-shifted region) is future
work; our minimal reimplementation does not reproduce their gains and so is inconclusive on the
empirical half (the definitional argument is sufficient and load-bearing). (RESULTS §16.)

## 8. The Mechanism Is Faithful (the negative is not a broken method)

To rule out "the structure loses because the mechanism is broken," we verify functional
faithfulness on synthetic concept where the true drift direction w(t) is known: recovery =
mean_t cos(ŵ(t), w(t)), ŵ(t)=E_x[∂score/∂x] (gauge-fixed, no Procrustes). The mechanism recovers
0.991 (10/10 seeds) on a 90° rotation, and **0.988 (10/10) under a full 2π rotation with a matched
Fourier basis**, where the random floor collapses to 0.017 (ceiling 0.972) — i.e. faithful across a
wide dynamic range, not merely above a high floor. The negative is thus about the *data* and the
*in-distribution nature* of the structure, not a defective mechanism. (RESULTS §8, §15.)

## 9. Discussion: When Can Design Help?

The barrier is not model capacity. (i) Under disjoint support, the rule's change on future inputs
is **unidentifiable** — no architecture recovers a function's change where it has no paired
observations. (ii) Where concept ≈ 0, there is no signal. (iii) Where concept is real, handling it
under temporal shift is an **extrapolation** problem, and adding in-distribution structure confers
no extrapolation advantage (our flip; redundancy). Design *can* help only by **(a) encoding a drift
prior** (e.g., Drift-Resilient TabPFN's SCM-shift prior — assumption-driven, and only where concept
is present) or **(b) moving to an online protocol** where adaptation speed/sample efficiency, not
static extrapolation, is the metric. This delimits the design space — a constructive corollary of
the negative.

**Limitations.** Claim B rests on 2 clean datasets (+Elec2 as a noisy substrate); the measurement
frame overlaps DISDE (we position as a temporal adaptation + degeneration extension, not an
invention); the Cai&Ye empirical adjudication is pending a faithful reproduction; redundancy is an
in-distribution argument, not a proof.

## 10. Conclusion

On tabular temporal benchmarks, covariate dominance makes concept drift unmeasurable by the
standard conditional/reweighting lens; a within-overlap model-transfer frame recovers it where
common support exists, and validates that the puzzle of "simple beats complex" is, in large part,
the absence (or unmeasurability) of exploitable concept — and, where concept exists, the
in-distribution redundancy of time-structured retrieval over a time feature. We release a validated
diagnostic toolkit so the community can characterize *which* shift a benchmark actually contains
before choosing methods for it.

---

## Appendix / pointers (internal)
- Numbers: RESULTS.md §1–16. Evidence chain: FINDINGS.md. Literature (verified): REFERENCES.md §0.
- Code: run_disde_degeneration.py (§3,§5), run_toolkit_validation.py (§4), run_elec2_q2.py (§6),
  run_modulation_adjudication.py (§7), run_q1_faithfulness.py (§8). Decision rules: PREREG_V2.md.

## [FUTURE] before submission
- Widen method sweep (GBDT+t, kNN+t, no-change, reference TabR, Drift-Resilient TabPFN) as anchors.
- Faithful Cai&Ye reproduction (LAMDA repo) to complete §7 empirical half.
- (Main track) scale to 15–20 tabular-temporal datasets; multiple time-aware methods through the lens.
- Toolkit packaging: API, datasheet/Croissant, reproducible pipeline.
- INSECTS abrupt/gradual variants for Claim B breadth (note: abrupt fails val→test gate — report as finding).
