# Concept Drift Is Unmeasurable in the Representation You Deploy: A Positivity-Bounded Diagnostic for Tabular Temporal Shift

> DRAFT v3.0 (2026-06-17). Reformalized after a 7-agent adversarial review (RED_TEAM.md)
> and the V3.0 decisive-gate experiments (PLAN_V3.md G1–G4, all passed/re-scoped). Replaces
> PAPER_DRAFT.md (v0.1, which the red-team demolished). Numbers: RESULTS §1–16 + gap_controls
> /representation/toolkit_adversarial summaries. Honest scope: every universal claim is
> re-scoped to the *deployed representation* under an *overlap/reweighting lens*; the lead
> contribution is the formal identification boundary + a ground-truth-AND-adversarially
> validated abstaining diagnostic. [PENDING]/[FUTURE] mark not-yet-done items.

---

## Abstract

On tabular temporal benchmarks, simple models match or beat elaborate time-aware deep methods.
We give a measurement-theoretic account and, crucially, bound what it can claim. (1) **Formal
object.** Adapting DISDE [Cai, Namkoong, Yadlowsky 2025] to the time axis, we define the
concept-drift estimand θ as the change in P(y|x) integrated over the early/late *overlap*, and
show the standard conditional/reweighting lens degenerates exactly when the overlap vanishes
(a positivity failure). (2) **It is a property of the representation, not the world.** Whether
overlap positivity holds is a functional of the feature map: on TabReD, adding *one* engineered
time-proxy feature flips a measurable concept (+0.98) to "unmeasurable"; on 4 of 5 high-covariate
TabReD datasets, stripping the time-leaking features makes concept measurable again — and it is
then ≈0. So "concept is unmeasurable on tabular temporal data" is false as a universal; the true
statement is **"in the deployed 261-feature representation, concept is un-checkable by the overlap
lens, and where you can check it (a sparse predictive representation) it is genuinely ≈0 or the
support is irreducibly disjoint."** (3) **A validated, abstaining diagnostic.** Our toolkit
(adversarial-validation covariate AUC, density-ratio degeneration, within-overlap transfer gap)
recovers planted concept (Spearman 1.0), abstains under positivity failure, and we *adversarially*
characterize its two real failure modes (a ~0.03 conditional-entropy confound; representation
sensitivity to feature *addition*, not invertible re-coordinatization). The two benchmarks with
genuine concept (Elec2 +0.146, INSECTS +0.150) survive a permutation placebo (so the gap is not a
train/test home-field artifact) *and* representation change. (4) Where concept is genuinely
present, a time-indexed retrieval *structure* does not beat a time *feature* (paired, 25 seeds),
and is an in-distribution device. We release the diagnostic and state its identification boundary.

---

## 1. Introduction

Time-based splits on industrial tabular data (TabReD [Rubachev et al. 2025]) collapse retrieval/
pretraining deep methods while GBDTs survive. *Why* is contested (protocol artifacts [Cai & Ye
2025a]; feature modulation [Cai & Ye 2025b]) but the **nature of the shift** — covariate P(x) vs
concept P(y|x) — has not been measured there. Time-aware structure can only pay off if the rule
P(y|x) changes; otherwise a time feature suffices. We ask whether exploitable concept drift exists
on these benchmarks — and, finding the question itself ill-posed in the deployed representation,
we make the *measurability* the object of study.

**Contributions (honest, post-rebuild).**
1. **A formal estimand and identification boundary** (§3): concept drift as DISDE's overlap Y|X
   term on the time axis, with a positivity proposition that says *off-overlap concept is not
   identified without a prior* — which both licenses abstention and resolves the design space to
   {drift-prior, online}.
2. **Representation-relativity, demonstrated** (§6): the measurability verdict is a functional of
   the feature map; we show feature engineering manufactures "unmeasurability," and re-scope the
   empirical picture to the deployed vs. sparse representation.
3. **A ground-truth- AND adversarially-validated abstaining diagnostic** (§4–5): it recovers
   planted concept, abstains under positivity failure, and we publish its measured failure modes.
4. **Controls that the flagship positives are concept, not artifacts** (§5): a permutation placebo
   refutes the home-field confound (Elec2/INSECTS survive); noise/prior nulls bound the residual.
5. **A confound-controlled negative** (§8): where concept exists, time-indexed retrieval structure
   ≤ a time feature, with an in-distribution (not extrapolation) mechanism.
6. **Released toolkit** with its identification boundary stated.

## 2. Related Work

**Shift decomposition.** DISDE [Cai, Namkoong, Yadlowsky, *Oper. Res.* 2025] decomposes a
performance drop into (i) harder seen examples, (ii) a within-overlap Y|X term, (iii) an
unseen-region X term, via a shared overlap measure S and density ratios. **Our concept estimand is
their term (ii) on the time axis; our "unmeasurability" is the regime where their term (iii) mass →
1.** We cite DISDE as the source of the frame and contribute the temporal/model-transfer
operationalization, the *representation-relativity* of the verdict, and the validated abstention.
**Adversarial validation** (classifier-two-sample test; [Lopez-Paz & Oquab 2017]; [Rabanser et al.,
*Failing Loudly*, 2019]; Kaggle practice; credit-scoring [Pang et al. 2021]) is the lineage of our
covariate-AUC; we claim no novelty for the technique. **WhyShift** [Liu et al. 2023] reports Y|X-shift
dominant on *spatial* tabular settings under a *performance-degradation* measure. Running our
conditional, overlap-lens toolkit on the same ACS family (ACSIncome via folktables, 5 states × 2 years)
does **not** reproduce a spatial=Y|X / temporal=X contrast: both spatial (CA→{TX,NY,FL,PA}, 2018) and
temporal (2014→2018, per state) settings show within-overlap concept ≈ 0 (gap ≈ placebo), and *spatial*
was the **more** covariate-shifted axis (mean cov-AUC 0.94 vs temporal 0.68) — the opposite of the
contrast we initially expected. We therefore make **no spatial/temporal axis claim**; WhyShift's
Y|X finding rests on a different (loss-based) measure and is not in direct conflict with ours. The run
does confirm our instrument (covariate-AUC, within-overlap gap, permutation placebo) transfers cleanly
beyond TabReD to ACS, and that under folktables' ~10 raw features *both* axes stay measurable (overlap
survives) — corroborating §6 that unmeasurability tracks *feature engineering*, not a spatial/temporal
axis. Our novelty is the unmeasurability/positivity result, not any covariate-dominance observation. **Tabular temporal methods**: TabReD does not decompose X vs Y|X; Cai & Ye [2025a,b]
fix protocol issues / modulate features. **Drift-Resilient TabPFN** [Helli et al. 2024] succeeds with
an SCM drift prior — the existence proof our §3 corollary predicts. **Streams**: covariate-vs-concept
is the field's founding taxonomy [Gama et al. 2014; Webb et al. 2016]; time-aware instance retrieval
exists [Žliobaitė 2011; Losing et al. 2016], so we scope our "empty intersection" to modern
differentiable deep tabular retrieval. Elec2's autocorrelation critique [Žliobaitė 2013] is
controlled for (§5, no-change baseline).

## 3. Setup, Estimand, and the Identification Boundary

**Estimand.** Fix a reference measure S on the overlap O = supp(P_early) ∩ supp(P_late). Define
concept drift as
  θ := E_{x∼S}[ ℓ(P(y|x, late)) − ℓ(P(y|x, early)) ],
for a skill functional ℓ. We estimate θ by the **within-overlap model-transfer gap**: select O by an
out-of-fold time classifier P(late|x)∈[0.1,0.9]; train f_early, f_late on early/late-overlap; on a
held-out late-overlap test, gap = ℓ(f_late) − ℓ(f_early). Under (A1) overlap positivity (P_early,
P_late mutually absolutely continuous on O) and (A2) realizability of P(y|x) on O, the gap is
consistent for θ on O. We use ℓ = AUC (discrimination-relevant for a retrieval predictor) and report
robustness to ℓ ∈ {Brier, Bayes-risk, KL} as [PENDING]; the rank-preserving recalibration kernel is an
acknowledged blind spot.

**Proposition (positivity / off-overlap non-identification).** Let R = supp(P_late) \ supp(P_early).
Without restrictions on the conditional, P(y|x) on R is not nonparametrically identified from
{P_early, P_late}: any two conditionals agreeing on supp(P_early) but differing on R induce identical
observable distributions, since no early data fall in R. Hence the concept change off O is
set-identified with the full simplex (uninformative) absent an extrapolation prior. *Corollary.* A
correct prior (smoothness/SCM, as in Drift-Resilient TabPFN) identifies within its assumptions; the
design space for off-overlap concept is exactly {drift-prior, online adaptation}, **not** "no
architecture can help." This both justifies **abstention** when positivity fails and corrects the
naïve impossibility framing.

**Representation-relativity (stated upfront).** O, and thus whether positivity (A1) holds, is a
functional of the feature map φ. Adding features — especially engineered time-proxies — shrinks O.
We therefore report every diagnostic *relative to a representation*: the deployed feature set, and a
sparse predictive set (§6). "Unmeasurable" always means "by the overlap lens, in this representation."

## 4. The Diagnostic Toolkit and Its Validated Abstention

**Components.** covariate_shift_auc (adversarial validation, with a drop-top-k pervasiveness probe);
disde_iw_degeneration (ESS and overlap_mass — heavy-tail vs disjoint-support modes); the within-overlap
transfer gap. Model-light (gradient-boosted trees). Released at [URL].

**Ground-truth validation** (synthetic covariate×concept grid, RESULTS §14): recovers planted concept
(Spearman(θ_plant, gap)=1.0), emits ~0 with no concept (max|gap|=0.002), degeneration is monotone in
covariate (ρ=±1.0), and it **abstains** under disjoint support (4/4). At ESS%=2.33 (reweighting dead)
the within-overlap gap still recovers the planted concept — the controlled analogue of Elec2.

**Adversarial validation** (the honest part, toolkit_adversarial summary): we run the DGPs the grid
never plants. The toolkit catches a 16%-mass subregion rule-flip (gap +0.156); does **not** false-fire
when covariate is entangled with the rule axis (−0.004); its "abstain" on a disjoint-but-smooth
trajectory does **not** hide exploitable concept (a time-aware model gains only +0.001 there); and it
is **invariant to invertible re-coordinatization** (gap +0.355 vs rotated +0.332). Two real failure
modes remain and we report them: (a) a **conditional-entropy (label-noise) confound** of ~**+0.034**
(a fixed boundary with early-noisy/late-clean labels reads as a small false concept) — well below the
flagship +0.15, but a stated caveat; (b) **sensitivity to feature *addition*** (not coordinate choice),
which is the representation result of §6.

## 5. The Gap Measures Concept, Not a Home-Field Artifact

The transfer gap compares a late-trained model (tested on its own distribution) to an early-trained
one — a possible train/test "home-field" advantage. We test it with a **permutation placebo**: permute
the early/late label *within the overlap band* (same region, no real early→late structure); a real
concept gap must drop to ~0. Results (gap_controls summary, 15 seeds, 95% CI):

| dataset | true gap [CI] | placebo [CI] | concept (true − placebo) |
|---|---|---|---|
| **Elec2** | +0.146 [.140,.151] | −0.035 [−.038,−.033] | **+0.181** (survives) |
| **INSECTS** | +0.150 [.147,.152] | −0.017 [−.019,−.015] | **+0.167** (survives) |
| cooking | −0.009 | −0.012 | +0.003 (≈0) |
| maps | −0.003 | −0.003 | +0.000 (≈0) |

The two concept benchmarks' true-gap CIs lie far above their placebo CIs: the gap is **not** home-field
(the placebo floor is small and slightly negative). A synthetic positive control gives true +0.985 /
placebo +0.005; prior-shift and noise-drift nulls give +0.001 and +0.034 (the §4 caveat). Net: Elec2/
INSECTS retain ~+0.11–0.15 of concept after subtracting both the home-field floor and the noise caveat.

## 6. Feature Engineering Controls Measurability (representation result)

**Reverse demonstration.** A synthetic stream with real concept (gap +0.98, overlap 0.997) becomes
"unmeasurable" (overlap 0.027) by appending **one** time-proxy feature c = t + ε. Same data, same
concept. The dichotomy is a representation functional.

**On TabReD** (representation summary): the 5 "disjoint" datasets, recomputed under de-time-leaked /
sparse-MI(@5–50) representations:
- **4/5 become measurable with concept ≈ 0**: sberbank (sparse: cov 0.63, overlap 0.996, gap +0.02),
  homecredit (cov 0.50, overlap 1.0, gap +0.00), homesite (cov 0.91, gap −0.01), weather (de-time-leak:
  cov 0.92, overlap 0.72, gap +0.02).
- **ecom stays genuinely disjoint** at every representation (cov 1.0, overlap 0) — the irreducible case.
- The concept benchmarks **survive de-time-leaking**: Elec2 +0.132→+0.078, INSECTS +0.144→+0.108
  (concept is representation-robust, not a feature artifact).

**Re-scoped three-way picture** (replaces the old §13): *in the deployed representation*, 5/8 TabReD
datasets are un-checkable (positivity fails); *checked on a sparse representation*, they are concept≈0
(4) or irreducibly disjoint (1); cooking/maps are checkable and concept≈0 in either; Elec2/INSECTS carry
concept that survives the placebo (§5) and representation change. So the practitioner's situation is:
their own feature pipeline destroys the overlap needed to ask whether concept drift is present, and when
you reconstruct a checkable representation, there is (almost) nothing there.

**External cross-validation (ACS/folktables).** The representation account makes a falsifiable
out-of-benchmark prediction: a data family with *few raw* features should stay measurable even under
strong distribution shift, because unmeasurability is an artifact of feature engineering rather than of
the data or the shift axis. We test this on ACSIncome (folktables; ~10 raw demographic features; 5 states
× years 2014/2018), running the same toolkit (cov-AUC, within-overlap gap, permutation placebo) used on
TabReD. The prediction holds: **all** spatial (state→state, 2018) and temporal (2014→2018) settings remain
**measurable** (overlap survives) despite high covariate shift (spatial mean cov-AUC 0.94, temporal 0.68) —
in sharp contrast to TabReD's 261-feature pipelines, where 5/8 deployed representations are un-checkable.
This is direct evidence that *unmeasurability tracks the representation, not a spatial/temporal axis*, and
that the toolkit (including its placebo) transfers cleanly beyond TabReD. As a by-product, the
within-overlap concept gap is ≈ 0 on both axes here (gap ≈ placebo), so we do **not** reproduce the
spatial=Y|X / temporal=X contrast under our conditional measure (§2); WhyShift's Y|X result is a different,
loss-based measure and not in conflict.

## 7. What Our Account Explains — and What It Does NOT (the C1 test)

We tested whether covariate dominance predicts TabReD's per-dataset deep-vs-tree margin, using
TabReD's **own published scores** [Rubachev et al. 2025, Table 3] so the tuning budget is controlled by
their comparable-tuning protocol (run_c1_ranking.py). **It does not.** Across the 8 datasets,
Spearman(cov_AUC, GBDT−TabR relative margin) = +0.22 (p=.61), Pearson +0.13 (p=.76); for the best deep
method the sign even reverses (Spearman −0.41). The decisive counterexample is **ecom-offers**: maximal
covariate (cov_AUC 1.0, overlap 0) yet **TabR BEATS GBDT** there. The only large margin is sberbank, an
outlier. So our diagnostics do **not** predict the per-dataset TabReD rankings, and we **do not claim to
explain the TabReD puzzle**: "simple beats complex" has other drivers we cannot rule out (tuning-budget
asymmetry, deep-method optimization instability, preprocessing). What our account *does* support is the
narrower, conditional statement — where concept is genuinely ≈0 (most TabReD, checked on a sparse
representation; §6), a concept-targeting structure has no extra signal to exploit — which is a claim about
*exploitable signal*, not a prediction of the leaderboard. We also note the time feature's *efficacy* (as
opposed to the structure's redundancy) is covariate recalibration under misspecification [Shimodaira 2000],
an X-side mechanism (§9), not concept exploitation.

## 8. Time-Structure vs Time-Feature Where Concept Exists (re-scoped)

On the measurable-concept datasets, a 5-arm shared-encoder design isolates structure: the primary
contrast **time_tabr_t − tabr_t** (direct time feature held present in both) is significantly negative
(INSECTS incremental −0.0067 [−.012,−.001] p=.006; incremental_abrupt −0.0205 [−.034,−.008] p<.001; 25
seeds, paired). The substrate is competitive (tabr_t ≈ mlp_t), and the hook **helps in-distribution
(random split +0.005,+0.021) but hurts under temporal extrapolation (−0.007,−0.021)** — an
in-distribution device, consistent with representational redundancy. **Scope (honest):** this is
"time-indexed instance retrieval is in-distribution-redundant with a time feature on two concept
benchmarks," *not* a general law that time-aware structure cannot help — covariate recalibration helps
(§9), and our own §9 adjudication concedes a time-aware modulation winning under covariate shift.

## 9. Adjudicating a Recent Positive (a remark, not a refutation)

Cai & Ye [2025b] modulate feature statistics over time and beat baselines on TabReD, calling the target
"concept drift." Their modulation is label-free (no y in the transform), hence by construction a
covariate (X-side) normalization — what they call "concept drift" is feature-distribution drift. We
flag this as a **terminology** point, not a refutation: our own minimal reproduction does not reproduce
their gains, and in fact its modulation gain correlates with measured concept (ρ=−0.50 with covariate),
so we explicitly do **not** claim their gains are X-side empirically — only that the transform cannot
*directly* condition on labels. A faithful reproduction localizing their gain is [FUTURE].

## 10. The Mechanism Is Not Miswired (scoped)

To rule out an implementation bug as the reason a time-indexed mechanism underperforms, we verify
functional faithfulness on synthetic concept with a matched basis: recovery 0.991 (90° rotation) and
0.988 under a full 2π rotation (random floor 0.017, ceiling 0.972). **Scope:** this shows the mechanism
is not miswired in a basis-matched setting; it does not certify real-data optimization quality (a
separate concern), and the negative of §8 is about in-distribution redundancy, not a bug.

## 11. Discussion

The identification boundary (§3) makes the design space concrete: off-overlap concept is reachable only
by a **drift prior** (where concept exists and the prior matches — Drift-Resilient TabPFN) or by **online
adaptation** (where the metric is adaptation speed, not static extrapolation). On the benchmarks studied,
the binding fact is upstream of architecture: feature engineering destroys positivity, and the checkable
reality is near-zero concept.

**Limitations.** (a) ℓ=AUC blind to recalibration drift [robustness PENDING]; (b) median early/late split
aliases drift shape — a rolling-origin trajectory is [FUTURE]; (c) Claim B rests on 2 concept benchmarks;
(d) the within-overlap frame is a temporal adaptation of DISDE, not a new identification strategy; (e) the
A↔B link to TabReD rankings is asserted pending C1; (f) the §9 empirical adjudication is unreproduced; (g)
no BH-FDR across the contrast family yet [hygiene PENDING]; (h) the ACS cross-check (§6) used only
ACSIncome — the WhyShift tasks reported as *more* Y|X-driven (ACSPublicCoverage, ACSMobility) are left to
[FUTURE], so our "no spatial/temporal axis contrast" conclusion is scoped to ACSIncome under our measure.

## 12. Conclusion

Whether concept drift is *measurable* on tabular temporal benchmarks is not a fact about the data but
about the deployed feature representation: industrial feature engineering destroys the early/late overlap
that conditional measurement requires, and where you reconstruct a checkable representation the concept is
genuinely small. We formalize the estimand and its positivity boundary, validate an abstaining diagnostic
both on ground truth and adversarially, control the flagship positives against the home-field confound,
and show that where concept does exist, time-indexed retrieval structure is in-distribution-redundant with
a time feature. We release the toolkit with its identification boundary stated, so the community can ask —
before choosing methods — not only *which* shift a benchmark contains, but *whether its representation even
permits the question.*

---

## Appendix / pointers (internal)
- Numbers: RESULTS §1–16; gap_controls/representation/toolkit_adversarial/disde/toolkit_validation summaries.
- Red-team: RED_TEAM.md (7 agents). Rebuild plan + gate verdicts: PLAN_V3.md. Literature: REFERENCES §0.
- Code: run_gap_controls.py (§5), run_representation.py (§6), run_toolkit_adversarial.py (§4),
  run_disde_degeneration.py / run_toolkit_validation.py (§4), run_elec2_q2.py (§8), run_q1_faithfulness.py (§10).

## [FUTURE] before submission (priority)
1. C1: `margin ~ cov_AUC + budget + seed_var` on the TabReD leaderboard (earn §7's puzzle link).
2. ℓ-robustness (Brier/Bayes-risk/KL) + rolling-origin gap trajectory (§3, limitations a,b).
3. Faithful Cai & Ye reproduction (§9). No-change/GBDT+t/TabPFN anchors (§8).
4. ACSIncome done (§6: toolkit generalizes; no spatial/temporal contrast under our measure). Extend to
   ACSPublicCoverage/ACSMobility — the WhyShift tasks reported as more Y|X-driven.
5. Hygiene: BH-FDR; pre-register Claim-A thresholds / sensitivity grid; seed-CIs on every real gap.
6. Toolkit packaging: API, datasheet/Croissant, reproducible pipeline.
