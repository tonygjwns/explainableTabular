# Is There Exploitable Concept Drift in Industrial Tabular Data? A Pre-Registered Identifiability Audit

> DRAFT v4.0 (2026-07-05). Target: TMLR (immediate) / NeurIPS D&B (next cycle). Supersedes
> PAPER_DRAFT_V3 (within-overlap lens era). Every number in this draft is traceable to a
> committed artifact: `prereg_results/` (run-meta-stamped JSONs), `audit_artifacts_2026-07-04/`
> (executed kill-tests), `PREREG_DEPLOYMENT_V2.md` §0–10 (rule→prediction→execution→read,
> evidenced by commit timeline). STATUS: Abstract + §1–3 written; §4–8 [PENDING].

---

## Abstract

Temporal shift is routinely invoked to motivate time-aware tabular models, yet which *kind* of
shift — a changing rule P(y|x) versus moving covariates P(x) — is rarely measured, and we show
the verdict depends on the measuring instrument itself. We study a deployment-native probe,
*staleness harm*: the change in future-window performance when old examples are added to an
identical recent training set, which isolates rule change from covariate coverage *by
construction*. With executed adversarial controls we expose three silent failure modes of that
construction: (i) label-noise decay under a provably fixed rule mints a "concept drift" verdict
at industrial magnitude (+0.021, matching a real borderline positive at +0.024); (ii) the overlap
gate that should certify when a null is uninformative saturates to 1.000 under duplicate rows and
entity cohorts with zero covariate shift; (iii) the concept/covariate separation is
hypothesis-class-relative — a kNN probe reads pure covariate shift as concept drift (+0.098)
while tree ensembles read it correctly. We repair the instrument — a cross-fitted *denoised
staleness* arm with a per-window noise gate (validity envelope mapped and enforced by
abstention), a group-aware separability estimate, and learnability-gated injection controls that
make claimed blindness *earned* rather than assumed — and validate it on a 14-cell pre-registered
synthetic battery, including rule change and noise drift co-occurring. Auditing eight industrial
datasets (TabReD) and designed-drift streams under a pre-registered protocol with a confirmatory
fresh-seed replication (10/10 verdicts stable) yields an identifiability *map*, not a detection
table: **no industrial dataset shows exploitable mean-rule drift above its per-dataset detectable
floor (0/8)**; the sole robust concept positive is designed drift (+0.135, denoised +0.152); the
one prior industrial positive is *diagnosed* by the instrument itself as label-noise decay (its
old-window noise proxy is 2.1–2.9× the recent level, and denoising the old labels makes the harm
vanish — indeed reverse sign); the remainder are covariate-dominated, noise-confounded, or
certifiably instrument-blind, each with a detectable-effect bound. Anchor streams give the
instrument a coherent sensitivity profile: it fires on monotone and single-switch rule changes
(9/9), and is correctly silent when old regimes recur — flagged by a negative recency gain, the
fingerprint of returning regimes — including on malware (EMBER), where decay is coverage-driven,
not label rot. We release the instrument, battery, and audit trail, and argue that drift-type
attribution without identifiability certificates — the current default in drift monitoring — is
unsound.

---

## 1. Introduction

Time-based splits on industrial tabular data collapse elaborate time-aware deep methods while
gradient-boosted trees survive (TabReD; Rubachev et al., 2025). The standard motivation for
time-aware architecture is *concept drift*: if the labeling rule P(y|x) changes over time, a
model that indexes, retrieves, or modulates by time should beat one that merely receives time as
a feature. But whether the rule actually changes on these benchmarks — as opposed to the
covariates P(x) moving under a fixed rule — has not been measured, and our central claim is that
measuring it is harder than the field assumes: **the concept/covariate verdict is a property of
the measuring instrument (its hypothesis class, its noise assumptions, its window geometry) as
much as of the data.** We make the measurement itself the object of study, in the setting a
deployed system actually faces: train on the past, predict the future, decide what to do with
old data.

**The probe.** We start from a deployment-native quantity we call *staleness harm*:

> staleness = score(train on recent N) − score(train on recent N ∪ old N), evaluated on a future
> window,

computed rolling-origin over K timestamp-value windows. Because the recent portion is identical
in both arms, covariate *coverage* cancels by construction: the only difference is N extra old
examples. If the rule is fixed, old examples carry correct labels and adding them should not
hurt; if the rule has changed, old labels contradict the current rule and pollute the fit —
staleness > 0. This construction (unlike train-window selection, which uses the same mechanics
for *adaptation*; Klinkenberg & Joachims, 2000) attributes *why* old data helps or hurts, and it
needs no importance weights, no density ratios, and no overlap between window supports to
compute — which is exactly what makes it attractive for industrial feature spaces where
overlap-based decompositions degenerate (D'Amour et al., 2021; Cai et al., 2023).

**Three executed failure modes.** The naive form of this argument — "correct labels can only
help" — is false for finite-capacity empirical risk minimization, and we show it fails silently
in three distinct ways, each demonstrated by a constructed null that the probe misreads:

1. **Label-noise drift (the diagnosis that motivated this paper).** A regression stream with a
   *fixed* conditional mean and label-noise standard deviation shrinking 1.5→0.3 over time mints
   a "concept" verdict at staleness +0.021 [+0.020, +0.023] — statistically indistinguishable
   from a real industrial positive we had previously obtained (+0.024 [+0.018, +0.030] on
   sberbank-housing). Old labels need not be *wrong* to hurt; being *noisier* suffices, because
   unweighted ERM is not efficient under heteroscedasticity. A second channel — x-dependent
   noise whose scale decays — fires the raw probe at +0.025 under an equally fixed rule.
2. **Overlap-gate saturation.** The natural certificate for "a null here is uninformative" is a
   window-separability AUC (can a classifier tell old rows from future rows?). Computed with a
   row-level train/test split, this gate saturates to 0.994–1.000 under exact duplicates, tight
   near-duplicates, and entity cohorts with *zero* covariate shift — the classifier memorizes
   rows, not distributions. Eight of ten real datasets read exactly 1.000 under the naive gate.
3. **Hypothesis-class relativity.** Rerunning the probe's own ground-truth suite with the model
   class swapped shows the concept/covariate *separation* is not a property of the probe design:
   a kNN probe reads pure covariate shift as concept drift (+0.098) and a linear probe reads
   prior shift as concept drift (+0.026), while HGB and random forests pass all controls. This
   is the empirical face of known theory (Shimodaira, 2000; Hinder et al., 2023): under
   misspecification, the ERM optimum depends on the training covariate distribution, so
   "old data hurts" can be manufactured by geometry alone.

**The repaired instrument.** Each failure mode gets a repair that is itself validated by
execution (§3–4): a *denoised staleness* arm that replaces old labels with cross-fitted
within-old-window model predictions — under noise-only drift the pseudo-labels are
approximately correct and the harm vanishes; under a changed rule they still encode the old rule
and the harm persists — plus a per-window *noise gate* with an explicitly mapped validity
envelope beyond which the instrument abstains; a group-aware, size-matched separability estimate
that deflates the memorization channel while leaving honest drift intact; and
*learnability-gated injection controls* that turn "this dataset is unmeasurable" from an
assumption into a demonstration: a known rule rotation is planted in the dataset's own covariate
geometry, verified to be learnable in-window, and only then is a persistent null allowed to
certify blindness. All verdicts are explicitly scoped to the tree-ensemble hypothesis class,
with linear and kNN probes carried as *canaries* whose false fires are reported, not hidden.

**The audit.** We pre-registered the decision cascade, thresholds, seed protocol, and aggregate
reading before the runs (rule → prediction → execution → read, evidenced by commit timestamps),
then audited eight TabReD datasets, Electricity, and the INSECTS streams, with a confirmatory
fresh-seed replication and a model-class panel. The result is not a detection table but an
identifiability map (§5): industrial mean-rule drift above the per-dataset detectable floor is
**0/8**; the designed-drift stream is the sole positive (and its denoised staleness *exceeds* the
raw one — the signature of genuine rule change); the single prior industrial positive is
*diagnosed* — not merely retracted — as label-noise decay, by the same instrument that once
minted it; two datasets earn a verified no-concept certificate via injection recovery; one earns
a blindness certificate; three fail to earn one (their injections are unlearnable — a fact the
naive protocol would have laundered into "earned blindness"). Anchor streams (§6) give the
instrument a coherent sensitivity profile — monotone and single-switch rule changes fire 9/9;
recurring regimes are correctly silent and are flagged by a negative recency gain, which is
impossible under one-way drift; malware (EMBER) reads as coverage-driven decay, not label rot,
consistent with how temporal degradation actually arises there (Pendlebury et al., 2019).

**Contributions.**
1. *An executed anatomy of drift-type attribution failure* (§3–4): three constructed nulls that
   silently mint concept-drift verdicts (noise decay, gate memorization, class relativity), each
   with magnitudes matching real borderline positives.
2. *A repaired, abstaining instrument* (§3): denoised staleness + noise gate with a mapped
   validity envelope; group-aware separability; learnability-gated injection certificates;
   class-scoped verdicts with canaries. Validated on a 14-cell pre-registered battery including
   the adversarial combinations (rule change *and* noise drift co-occurring must still fire —
   it does, at 58% of clean power).
3. *A pre-registered identifiability map of industrial tabular ML* (§5): 0/8 exploitable
   mean-rule drift; one diagnosed false positive with mechanism; per-dataset certificates and
   detectable-effect bounds; 10/10 confirmatory stability and 10/10 cross-class (HGB↔RF)
   agreement, with a real-data demonstration that a linear monitor flips the canonical
   Electricity dataset to "concept drift".
4. *A sensitivity profile with anchors* (§6): what this lens can and cannot see, established on
   designed-drift streams (fires iff old labels contradict the *current* regime), with the
   negative-recency fingerprint for recurring regimes and an honest malware null.

**What we do not claim.** We do not claim concept drift is absent from industrial tabular data —
several cells are certified *blind*, and blindness is not absence. We do not claim the
identifiability theory is new — that overlap failure blocks nonparametric identification is
classical (D'Amour et al., 2021; Ben-David et al., 2010), and the class-relativity principle is
established (Hinder et al., 2023); our contribution is the executed instrument, its failure
anatomy, and the certified map. And we do not claim model-agnosticism: every verdict is relative
to the deployed hypothesis class, which we argue is the only honest way to state drift-type
attribution at all.

## 2. Related Work

**Shift-type maps on tabular data.** WhyShift (Liu et al., 2023) is the closest prior: it
attributes performance drops to Y|X- vs X-shift across tabular datasets and concludes
Y|X-shifts are prevalent — on predominantly spatial/subpopulation axes, under an
overlap-assuming decomposition. Our delta is threefold: the *temporal* axis on deployed
industrial representations (where positivity actually fails and their lens degenerates), an
*abstention arm* with per-dataset certificates (WhyShift assumes the decomposition is
measurable), and the instrument-failure anatomy. DISDE (Cai et al., 2023) supplies the
decomposition formalism we inherit terminology from; its shared-distribution term is identified
only on common support — the boundary our map makes operational. TabReD itself (Rubachev et
al., 2025) frames everything as "gradual temporal shift" and contains no decomposition or
recency experiments; TableShift (Gardner et al., 2023) and Wild-Time (Yao et al., 2022) are
shift benchmarks without drift-type attribution.

**Identifiability and class-relativity.** That P(y|x) is not identified off-support without
assumptions is textbook positivity (D'Amour et al., 2021, for high-dimensional overlap failure;
Ben-David et al., 2010, for the learning-theoretic impossibility). Hinder et al. (2023) prove
constructively that loss-based drift detection is class-relative in both directions — virtual
drift that moves the loss, real drift invisible to it — and their survey states "the used model
class is crucial." Loog et al. (2019) show ERM risk can be non-monotone in sample size even
i.i.d., closing the door on any unconditional "more correct data cannot hurt" lemma. Gower-Winter
et al. (2026) argue drift detection is ill-posed through windowing choices. We treat all of this
as settled theory that the applied drift-monitoring stack has not absorbed, and contribute the
executed demonstration + the repaired protocol; our propositions are scoping lemmas, not theory
contributions.

**Old data harming.** Shimodaira (2000) is the mechanism for the misspecified case; the Data
Addition Dilemma (Shen et al., 2024) documents mixture harm empirically in clinical ML;
Klinkenberg & Joachims (2000) use held-out error over candidate training windows to *adapt*
window size. To our knowledge, no prior work uses the add-old-data contrast as a drift-*type*
identifier with verdict semantics, nor reports the label-noise-decay false-positive channel —
under the field-standard definition of real drift as any change in P(y|x) (Gama et al., 2014;
Webb et al., 2016), noise drift *is* drift, which is precisely why an instrument that claims to
detect *exploitable rule change* must separate the two; ours is, to our knowledge, the first
that does so by construction and validates the separation adversarially.

**Drift detectors and their evaluation.** Classical detectors (surveys: Gama et al., 2014; Lu et
al., 2019) monitor loss or distribution statistics and are evaluated by injecting known drifts
into streams (Poenaru-Olaru et al., 2022); Detectron (Ginsberg et al., 2023) defines harmful
shift model-relatively. Our injection control differs in role: it is an *in-situ, per-dataset
power certificate* (planted in the real covariate geometry, learnability-gated), used to
distinguish earned blindness from instrument vacuity — a distinction we show matters on real
data, where three of four "unidentifiable" cells fail the learnability gate.

**Malware.** TESSERACT (Pendlebury et al., 2019) established temporally honest evaluation in
malware and documents performance decay. Our EMBER read (old data *helps*; staleness −0.012;
detectable floor 0.0014) refines rather than contradicts it: the decay is coverage-driven (new
families appear — covariate expansion), not label rot (a 2017 malware sample is still malware),
which is exactly the distinction a deployment lens should draw.

## 3. The Instrument

### 3.1 Setting and estimand

Rows (x_i, y_i, t_i) arrive over time; the auditor sees the full labeled history (retrospective
audit, not online prediction). Fix K windows by timestamp-value rank (ties never straddle
boundaries; per-seed boundary jitter). For each future window W_j (back half), with old anchor
W_0 (earliest window with ≥200 rows and a trainable label set) and recent window W_{j−1}, draw
size-matched samples (N = min(|recent|, |old|, 6000)) and train three models of the deployed
class: recent-only, old-only, recent∪old. Report per-seed means over future windows of

- **decay** = score(old model, held-out W_0) − score(old model, W_j) — aging, mechanism-blind;
- **recency gain** = score(recent) − score(old) on W_j — adaptation value, conflates coverage
  and rule change; *its sign is itself diagnostic (§6): negative recency is impossible under
  one-way drift and fingerprints recurring regimes*;
- **raw staleness** = score(recent) − score(recent∪old) on W_j — the attribution probe;
- **denoised staleness** = score(recent) − score(recent ∪ (X_old, ĝ_old(X_old))) on W_j, where
  ĝ_old is a 2-fold cross-fitted model *within* W_0 producing strictly out-of-fold hard
  pseudo-labels.

Scores: AUC (binary), accuracy (multiclass), −RMSE on z-scored targets (regression). CIs are
Student-t over 10 seeds; each seed re-draws a 90% row subsample, re-jitters window boundaries,
and re-samples all training sets. The pre-registered estimand is *exploitable mean-rule drift*:
a change in the decision-relevant functional of P(y|x) that makes old labels contradict the
current rule for the deployed hypothesis class. This is deliberately narrower than the
field-standard "any change in P(y|x)" (Webb et al., 2016) — the narrowing is the point, because
the wider definition classifies label-noise drift as concept and thereby licenses retraining
that cannot help.

**Why the denoised arm separates rule change from noise drift.** Under noise-only drift the
conditional mean/Bayes rule is unchanged, so ĝ_old estimates the *same* rule from noisy labels;
its pseudo-labels are approximately correct denoised labels, and adding (X_old, ĝ_old(X_old))
to the recent set cannot systematically hurt — executed: the +0.021 noise-decay false positive
collapses to +0.004 while the gate (below) flags the mechanism. Under genuine rule change,
ĝ_old consistently estimates the *old* rule; its pseudo-labels still contradict the current
rule, and the harm persists — executed: a rotating-rule stream keeps +0.541 of its +0.546 raw
signal, and rule-change-plus-noise-decay still fires at +0.316. The denoiser is not unbiased:
its pseudo-label error grows with old-window noise, and we *mapped the boundary* — at an
old/recent noise ratio ≈5.7 the denoised arm itself crosses the decision floor on a pure null
(+0.026). The instrument therefore carries an envelope constant (abstain above ratio 4.7) rather
than a pretense of a theorem; per Loog et al. (2019), no unconditional soundness lemma exists to
be had.

**The noise gate.** Per window, a fresh model of the deployed class is fit on 70% and its
held-out irreducible-error proxy recorded (regression: MSE on z-scored y; binary: 1−AUC;
multiclass: 1−accuracy). The gate statistic is old-window proxy / median recent-window proxy;
it fires above 1.5 (stable synthetic controls calibrate at 0.75–0.99) and defines the envelope
above 4.7. The gate alone is *not* a sufficient repair — a gate-veto rule would misfile genuine
rule change co-occurring with noise drift (executed: +0.316 must fire despite gate 3.67) — it
supplies the *mechanism label* (NOISE-DRIFT-CONFOUNDED) when the raw arm fires and the denoised
arm does not.

### 3.2 Certificates: separability, injection, learnability

**Window separability D (not "overlap").** D is the median held-out AUC of a classifier
separating W_0 from each back-half window, on the proxy-stripped feature space, with two
repairs over the naive version: size-matching by *random subsample* (a head-slice on
time-sorted rows manufactures separability from window-size imbalance — executed: shuffle-D
0.94 with no static feature, collapsing to 0.50 under the fix) and a *group-aware* train/test
split keyed on near-duplicate clusters (z-scored rows rounded to 0.1): exact duplicates at
multiplicity 5 inflate row-split D to 0.994 and entity cohorts to 1.000 with zero covariate
shift; the group split returns both to ≈0.50 while honest drift keeps D high. We report D as
*separability*, never as support overlap: a single genuinely predictive drifting feature
saturates it, so D ≥ D* = 0.96 routes a staleness null to the injection control rather than
concluding anything by itself.

**Learnability-gated injection.** For any candidate-unidentifiable dataset (and, as a positive
control, for any CONCEPT verdict), a reference concept — a rule rotation of fixed strength on
the two highest-variance features — is planted in the dataset's own (X, t) geometry and the full
staleness pipeline re-run (10 seeds, same power as the main read). Before the null is allowed to
mean anything, the injected rule must be *learnable in-window* (held-out AUC ≥ 0.65 / R² ≥ 0.20
/ accuracy ≥ majority + 0.10): executed on a constructed geometry whose top-variance features
are heavy-tailed junk, the injected rule is unlearnable (in-window AUC 0.506) and the naive
protocol grants "earned blindness" vacuously; the gate converts that to *injection-vacuous* —
certificate refused. Outcomes: recovery (⇒ the real null was informative: verified no-concept),
earned blindness (learnable but unrecoverable ⇒ the geometry genuinely hides this signal
class), or vacuity (no certificate). On the real audit, this distinction is load-bearing: 3 of 4
"unidentifiable" industrial cells fail the learnability gate.

### 3.3 Decision cascade and scope

The pre-registered cascade (PREREG §3; verbatim in the artifact): DEPLOYMENT-CONCEPT requires
the *denoised* arm to fire (CI lower bound > 0 and mean > 0.02) within the noise envelope; a raw
fire with the gate on is NOISE-DRIFT-CONFOUNDED; a raw fire with the gate off is
RAW-ONLY-POSITIVE (unresolved, never concept); nulls route through D to the injection
certificates; a sub-floor denoised CI is reported as a no-evidence band (calibrated: 2/5 no-drift
anchor streams land there at 1/10–1/20 of the floor — CI-significant, decision-irrelevant). A
strict secondary rule (CI lower bound > floor) is computed alongside; any cell whose verdict
differs between the two readings is marked rule-sensitive and barred from headlines. Every
verdict is scoped to the deployed hypothesis class: the ground-truth suite passes under HGB and
random forests and fails under linear/kNN probes (§4), so tree-ensemble verdicts are
decision-grade and linear/kNN run as canaries. All runs emit provenance metadata (commit, argv,
library versions, seeds) into versioned artifacts; the decision rules, thresholds, seed
protocol (exploratory 0–9, confirmatory 100–109), and aggregate reading were committed before
the runs they govern.

[§4 Validation: the 14-cell battery, failure-mode table, envelope mapping, model-class matrix —
PENDING]
[§5 The map — PENDING] [§6 Anchors & sensitivity profile — PENDING] [§7 Discussion — PENDING]
[§8 Reproducibility — PENDING]

## References (partial, verified during the audit)

- Ben-David, Lu, Luu, Pál (2010). Impossibility theorems for domain adaptation. AISTATS.
- Cai, Namkoong, Yadlowsky (2023). Diagnosing model performance under distribution shift
  (DISDE). Operations Research / arXiv:2303.02011.
- D'Amour, Ding, Feller, Lei, Sekhon (2021). Overlap in observational studies with
  high-dimensional covariates. Journal of Econometrics 221(2).
- Gama, Žliobaitė, Bifet, Pechenizkiy, Bouchachia (2014). A survey on concept drift adaptation.
  ACM Computing Surveys.
- Gardner, Popović, Schmidt (2023). Benchmarking distribution shift in tabular data with
  TableShift. NeurIPS D&B.
- Ginsberg, Liang, Krishnan (2023). A learning-based hypothesis test for harmful covariate
  shift (Detectron). ICLR.
- Gower-Winter, Groen, Krempl (2026). The window dilemma: why concept drift detection is
  ill-posed. arXiv:2602.06456.
- Hinder, Vaquet, Brinkrolf, Hammer (2023). On the hardness and necessity of supervised concept
  drift detection. ICPRAM. (+ 2024 survey, Frontiers in AI.)
- Johansson, Sontag, Ranganath (2019). Support and invertibility in domain-invariant
  representations. AISTATS.
- Klinkenberg, Joachims (2000). Detecting concept drift with support vector machines. ICML.
- Liu, Wang, Cui, Namkoong (2023). On the need for a language describing distribution shifts
  (WhyShift). NeurIPS D&B.
- Loog, Viering, Mey (2019). Minimizers of the empirical risk and risk monotonicity. NeurIPS.
- Lu, Liu, Dong, Gu, Gama, Zhang (2019). Learning under concept drift: a review. TKDE.
- Moreno-Torres, Raeder, Alaiz-Rodríguez, Chawla, Herrera (2012). A unifying view on dataset
  shift. Pattern Recognition.
- Pendlebury, Pierazzi, Jordaney, Kinder, Cavallaro (2019). TESSERACT: eliminating experimental
  bias in malware classification. USENIX Security.
- Poenaru-Olaru, Cruz, van Deursen, Rellermeyer (2022). Are concept drift detectors reliable
  alarming systems? IEEE Big Data.
- Rubachev, Kartashev, Gorishniy, Babenko (2025). TabReD: analyzing pitfalls and filling the
  gaps in tabular deep learning benchmarks. ICLR.
- Shen, Raji, Chen (2024). The data addition dilemma. MLHC.
- Shimodaira (2000). Improving predictive inference under covariate shift by weighting the
  log-likelihood function. J. Statistical Planning and Inference.
- Souza, Reis, Maletzke, Batista (2020). Challenges in benchmarking stream learning algorithms
  with real-world data (INSECTS). Data Mining and Knowledge Discovery. [verify exact citation]
- Vela et al. (2022). Temporal quality degradation in AI models. Scientific Reports.
- Webb, Hyde, Cao, Nguyen, Petitjean (2016). Characterizing concept drift. DMKD.
- Yao, Choi, Cao, Lee, Koh, Finn (2022). Wild-Time: a benchmark of in-the-wild distribution
  shift over time. NeurIPS D&B.
