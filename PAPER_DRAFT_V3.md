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
characterize its real failure modes; the ESS-floor abstention is *enforced* in the deployed rule
(the same rule the ground-truth test passes). The flagship within-overlap gaps survive a permutation
placebo, but a deeper decomposition shows **Elec2's gap is largely serial autocorrelation plus a
noise-difficulty drift** (it drops as a clean concept anchor); INSECTS' concept survives thinning,
and a synthetic river panel supplies ground-truth concept of dial-able size. (4) **The retrieval
structure has an identified niche.** Under *monotonic* drift a time-indexed retrieval structure does
not beat a time feature (the original negative, paired, 25 seeds); but under *reoccurring* drift —
where an old concept returns and recency fails — the **learned retrieval beats recency and a
fixed-metric k-NN** (struct−recency **+0.26**, CI excludes 0; real INSECTS-reoccurring **+0.21**),
recalling the returned concept. The within-overlap measure plus a reoccurrence signal thus predict
*which* adaptation pays. We release the diagnostic and state its identification boundary.

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
4. **Controls that separate concept from its confounds** (§5, §8): a permutation placebo refutes the
   home-field confound, and a decomposition (thinning, lagged-label, noise proxy) shows **Elec2's
   apparent concept is mostly serial autocorrelation + noise drift** (we retire it as a clean anchor),
   while INSECTS' concept survives — so the positive concept evidence is reported narrowly and honestly.
5. **A conditional method positive: the reoccurring-drift niche** (§8): the time-indexed retrieval
   structure is redundant under *monotonic* drift (the original negative, now scoped) but **beats
   recency and a fixed-metric k-NN under *reoccurring* drift** (learned struct−recency +0.26, CI>0;
   real INSECTS-reoccurring +0.21), recalling the returned concept — and the measure plus a
   reoccurrence signal predict *when* it is the right tool (the diagnostic is **generative**).
6. **Released toolkit** with its identification boundary stated.

## 2. Related Work

**Shift decomposition / attribution.** DISDE [Cai, Namkoong, Yadlowsky, *Oper. Res.* 2025] decomposes a
performance drop into (i) harder seen examples, (ii) a within-overlap Y|X term, (iii) an
unseen-region X term, via a shared overlap measure S and density ratios; related attribution work splits
a distribution change into causal mechanisms (marginal vs conditional) [Budhathoki et al., *AISTATS* 2021]
or attributes a performance change to specific shifts [Zhang et al., *ICML* 2023]. **Our concept estimand is
DISDE's term (ii) on the time axis; our "unmeasurability" is the regime where their term (iii) mass → 1.**
We claim no novelty for the decomposition frame itself — we cite it as the source and contribute the
*temporal/model-transfer operationalization*, the *representation-relativity* of the verdict, and the
*adversarially-validated abstention*. **Overlap/positivity.** That a high-dimensional covariate map can
*destroy* common support is a known theorem, not our discovery: [D'Amour, Ding, Feller, Lei & Sekhon,
*J. Econometrics* 2021] prove strict overlap fails as covariate dimension grows. Our §6 is the *empirical
demonstration* of that mechanism on the time/representation axis (industrial feature pipelines manufacture
the positivity failure), and our positivity boundary (§3) is the time-axis instance of their result — we
cite it as the formal basis and scope our §6 contribution to the demonstration plus the abstention, not a
new theorem.
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

*Two reporting points (honest accounting).* (a) **The placebo is slightly negative** (−0.035, −0.017), not
≥0 as a pure home-field advantage would predict; we read this as a small structural negative bias of the
band-restricted estimator, so we report the **raw** gap (+0.146 / +0.150) as the primary figure and the
bias-corrected gap (raw − placebo = +0.181 / +0.167) as an upper bound, not the headline. (b) **Floor
units:** the noise-drift null's raw gap is +0.034 but its *bias-corrected* value is +0.041 (= +0.0349 −
(−0.0065)); since the pre-registered rule thresholds the *bias-corrected* gap, the floor there is 0.041
(not 0.034), and Elec2/INSECTS clear it either way. When a raw per-cell gap is thresholded instead (§6,
no per-cell placebo), the correct floor is the raw 0.034.

**Hygiene (gap_hygiene summary).** The verdict is hardened along every axis a reviewer would probe.
*(i) Seed-CI:* the CIs above are over 15 seeds. *(ii) Loss-robustness:* the gap is positive with CI
excluding 0 not only for AUC/accuracy but for **Brier** (Elec2 +0.43, INSECTS +0.12) and **log-loss
(Bayes-risk)** (+1.41, +0.16), and the predictive laws move (mean KL late‖early 1.66, 0.73) — the concept
verdict does not hinge on the metric (resolving the recalibration-drift blind spot). *(iii) Rolling-origin
g(t):* sweeping the cut over time-quantiles {.3,.4,.5,.6,.7}, **every** cut-point is positive (Elec2 mean
+0.122 [.109,.135], gradual; INSECTS +0.157 [.095,.219], monotone-decreasing in the cut — drift is
front-loaded, so the median single number *understates* early-window concept). *(iv) Sensitivity grid:*
across band × min-per-half × classifier (HGB/logreg), the concept verdict holds in **18/18** cells for
each dataset. *(v) Multiplicity:* the one-sided paired Wilcoxon (true > placebo) survives Benjamini-Hochberg
across the contrast family (both BH-p ≈ 3×10⁻⁵). Under the **pre-registered** rule (CI > placebo ∧
bias-corrected > 0.034 noise floor ∧ BH-significant ∧ metric-invariant), **both Elec2 and INSECTS are
classified as genuine concept**; cooking/maps are not. *(Update, in progress: these numbers are on the
full deployed representation; §6 shows Elec2-full is un-checkable once the IW-ESS floor is enforced, so the
honest Elec2 figure is the de-time-leaked +0.074 and this hygiene panel is being re-run on that
representation. INSECTS is unaffected — it is measurable at ess 41% on the full representation.)*

## 6. Feature Engineering Controls Measurability (representation result)

**Reverse demonstration.** A synthetic stream with real concept (gap +0.98, overlap 0.997) becomes
"unmeasurable" (overlap 0.027) by appending **one** time-proxy feature c = t + ε. Same data, same
concept. The dichotomy is a representation functional.

That a high-dimensional feature map *must* eventually destroy common support is a theorem [D'Amour et al.
2021]; what follows is its empirical instance on the time axis, under a measurement rule that **abstains**
where the support is too thin to trust. Concretely, the measurable gate enforces an IW-ESS floor (≥5%) —
the *same* rule the ground-truth validation passes (§4) — so near-disjoint cells abstain rather than emit a
spurious gap (multi-seed CIs, ess-gated; representation summary):
- **sberbank, homecredit → all-≈0 where checkable**: sberbank sparse-MI@{5–50} (ess 46–73%) gap +0.02
  [≈0] at every k; homecredit @5/@10 (ess 100/83%) gap +0.00/+0.006 [≈0], while @20/@50 (ess 0.6/1.1%)
  now **abstain** — the high-k cells that *looked* like a large negative gap (−0.06/−0.11) were thin-overlap
  artifacts and are correctly withheld under the enforced floor.
- **ecom, homesite → no checkable representation** (every rep abstains: ecom cov 1.0/overlap 0; homesite
  sparse ess 0.1–0.6%) — positivity fails throughout; we make no concept claim either way.
- **weather → mixed/unstable**: its sparse reps are measurable (ess 30–80%) but the gap swings −0.023…+0.021
  across k, so we honestly report it as representation-unstable, not as concept or as ≈0.
- **The flagship Elec2 obeys the same law**: on the *full* deployed representation it is **un-checkable**
  (ess 0.6% — two time-proxy features collapse the overlap), and concept is recovered only after
  de-time-leaking (ess 35%, gap **+0.074 [concept]**, > the 0.034 floor). INSECTS is the robust case —
  concept on the full representation (ess 41%, +0.147) and across sparse reps (+0.11…+0.14).

**Re-scoped picture** (replaces the old §13, now ess-gated): *in the deployed representation*, the TabReD
sets and even Elec2 are largely un-checkable (positivity fails); *checked on a representation where overlap
survives*, they are concept ≈ 0 (sberbank/homecredit), irreducibly disjoint (ecom/homesite), or
representation-unstable (weather); only **INSECTS (robustly) and de-time-leaked Elec2 (+0.074)** carry
concept. This **supersedes the full-representation +0.146 Elec2 headline of §5**: the honest Elec2 concept
is the de-time-leaked +0.074, and §5's hygiene is being re-run on that representation (the earlier number
ran through the pre-enforcement gate). The practitioner's situation: your own feature pipeline destroys the
overlap needed to ask whether concept drift is present, and where you reconstruct a checkable
representation, there is little — except INSECTS and, modestly, Elec2.

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

**A motivation–result gap, stated plainly.** Our opening motivation is that *measuring* the kind of shift
should inform method choice on benchmarks like TabReD, where simple models beat deep ones. C1 shows our
measure does **not** predict that ranking — so this specific practical promise is *not* delivered, and we
say so rather than letting the framing imply otherwise. The contribution survives the gap because it is
relocated: the value is the **measurement-and-abstention** result (what is checkable, what is not, and where
the question is ill-posed), not a recipe for the leaderboard. A reader expecting "diagnose the shift →
pick the model" should read this as evidence that, at least via the covariate/concept lens, that pipeline
does not hold on TabReD.

## 8. When Retrieval Structure Helps: the Reoccurring-Drift Niche

**The structure is redundant under *monotonic* drift.** On the measurable-concept datasets, a 5-arm
shared-encoder design isolates structure: the primary contrast **time_tabr_t − tabr_t** (direct time
feature held present in both) is significantly negative (INSECTS incremental −0.0067 [−.012,−.001] p=.006;
incremental_abrupt −0.0205 [−.034,−.008] p<.001; 25 seeds, paired), and the retrieval structure ties a
parametric model (tabr_t ≈ mlp_t). But — and this is the key qualification — **all of these benchmarks
are *monotonic* drift** (the concept moves and stays moved). On monotonic drift, recency/forgetting is the
right inductive bias and retrieval is redundant; that is the honest negative, now *scoped to monotonic drift*.

**The structure WINS under *reoccurring* drift — its identified niche.** When an old concept *returns*
(reoccurring drift), recency fails (it discards the matching old data) and retrieval-by-similarity should
win (it recalls the reoccurred examples). We test this on a panel of synthetic streams with dial-able
drift structure (river: SEA/Agrawal/STAGGER/Sine, A→B→A reoccurring vs A→B monotonic vs no-drift) plus the
real INSECTS-reoccurring stream, comparing static / recency (HGB on the recent window) / retrieval. Two
pre-registered results, both holding with the CI excluding 0:
- *(1) Plain k-NN retrieval beats recency on reoccurring* (mean retrieval−recency **+0.19** [+0.03, +0.36],
  n=12) and **loses on monotonic** (−0.10 [−0.19, −0.00]); no-drift ≈ 0. (`retrieval_vs_recency`)
- *(2) The learned retrieval structure (tabr_t) beats recency by even more* (mean struct−recency **+0.26**
  [+0.15, +0.37], n=9 reoccurring), is redundant on monotonic (struct − parametric ≈ 0, reproducing the
  negative above), and — crucially — **fixes the feature spaces where plain k-NN's fixed metric fails**
  (Agrawal-reoccurring: k-NN −0.10 → learned +0.32). On the **real** INSECTS-reoccurring stream the learned
  structure beats recency by **+0.21** (k-NN gave +0.02 — the learned metric is ~10× stronger). (`learned_retrieval`)

**Reading.** The original "time-indexed retrieval is redundant" negative is real but *drift-structure-specific*:
it holds for monotonic drift, where recency suffices. On **reoccurring** drift the retrieval structure is
the right tool — it recalls the returned concept that recency has thrown away, and the *learned* retrieval
beats both recency and a fixed-metric k-NN. This makes the diagnostic **generative**: the within-overlap
measure says *whether* concept is present, and a drift-structure (reoccurrence) signal says *which*
adaptation pays — monotonic → recency, reoccurring → (learned) retrieval. **Scope (honest):** the strong,
significant win is over *recency*; over a parametric model that already sees all data the structure's edge
is modest (struct−parametric ≈ +0.06–0.07); the panel is synthetic-heavy (one real reoccurring stream, where
the effect is nonetheless large, +0.21); and operationalizing the reoccurrence signal as a deployable
diagnostic is left to future work. This is *not* a claim that retrieval beats everything everywhere — it is
a claim that retrieval has an identified niche (reoccurring drift) that our measure can point to.

**External calibration (anchors).** To rule out that the arm comparison sits below a trivial or a strong
baseline, we ran — on the *same temporal split and features* — k-NN±t, GBDT (LightGBM)±t, and a
no-change/persistence baseline (5 seeds, anchors_summary). Two things follow. *(i) The arms clear the
floors.* On Elec2 the neural arms (mlp_t ≈ 0.905 AUC) sit **above** both the strong GBDT anchor (lgbm
0.887) and the persistence baseline (no_change AUC 0.845), so the well-known Elec2 autocorrelation
critique [Žliobaitė 2013] does not explain our numbers; on INSECTS-incremental the arms (≈ 0.67 acc) are
competitive with the strongest anchor (lgbm_t 0.679) and far above persistence (0.163, multiclass).
*(ii) A non-neural model independently reproduces the regime-dependence of the time signal.* Adding the
time feature to GBDT **helps** under incremental drift (lgbm→lgbm_t **+0.070** on INSECTS-incremental) but
**hurts sharply under abrupt drift** (**−0.192** on INSECTS-incremental-abrupt; ≈0 on Elec2) — the same
in-distribution-help / extrapolation-hurt signature our neural time hook shows. That a tree model exhibits
it too confirms *time-as-feature is an in-distribution device, not an extrapolation one*, corroborating the
redundancy reading rather than resting on it.

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

**Limitations.** (a) ~~ℓ=AUC blind to recalibration drift~~ **resolved**: the verdict is metric-invariant
across Brier/log-loss/KL (§5); (b) ~~median split aliases drift shape~~ **resolved**: the rolling-origin
g(t) trajectory is positive at every cut (§5); (c) the **positive evidence is narrow on real data**: under
the enforced ESS floor Elec2 retires (its gap is autocorrelation + noise, §8) and INSECTS is the only real
robust-concept stream, so the *real-data* concept and the *real-data* reoccurring-niche result each rest on
one stream (INSECTS / INSECTS-reoccurring); the reoccurring method positive is otherwise carried by a
synthetic river panel, and the structure's edge over an all-data parametric model is modest (the strong win
is over recency) — broadening the real reoccurring evidence is the priority; (d) the within-overlap frame is a temporal
adaptation of DISDE, and the overlap-collapse mechanism of §6 is a known theorem [D'Amour et al. 2021] — we
contribute the empirical demonstration on the time/representation axis and the validated abstention, **not a
new identification theorem**; (e) the
A↔B link to TabReD rankings is asserted pending C1; (f) the §9 empirical adjudication is unreproduced; (g)
~~no BH-FDR across the contrast family~~ **resolved**: BH-FDR applied, both reject (§5); (h) the ACS cross-check (§6) used only
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
2. ~~ℓ-robustness (Brier/Bayes-risk/KL) + rolling-origin gap trajectory~~ **done (§5)**.
3. Faithful Cai & Ye reproduction (§9). No-change/GBDT+t/TabPFN anchors (§8).
4. ACSIncome done (§6: toolkit generalizes; no spatial/temporal contrast under our measure). Extend to
   ACSPublicCoverage/ACSMobility — the WhyShift tasks reported as more Y|X-driven.
5. ~~Hygiene: BH-FDR; pre-register Claim-A thresholds / sensitivity grid; seed-CIs on every real gap~~
   **done (§5: BH both reject; verdict invariant 18/18 cells; 15-seed CIs)**.
6. Toolkit packaging: API, datasheet/Croissant, reproducible pipeline.
