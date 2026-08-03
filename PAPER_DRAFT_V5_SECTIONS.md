# PAPER V5 — §3 and §6 drafts

**Status.** Draft sections for the instrument-first reframe of `PAPER_V5_SKELETON.md`, written
2026-08-03. `PAPER_DRAFT_V4.md` / `_KO.md` / `paper/main.tex` are untouched and remain the live
manuscript; adopting V5 means folding these two sections in and demoting V4 §5 to §7, and
abandoning V5 means deleting this file. **No new claims and no new numbers**: every reading below
is already committed in an artifact and appears in V4, mostly in §4.1, §4.3, §5.2, §5.3 and the
Limitations list. What changes is which of them carry the paper.

Three slots are marked `[E1]`, `[E2]`, `[E3]` and stay empty until the day-4 queue lands. They are
additive: each strengthens a subsection that already stands without it.

---

## 3. The failure channel

### 3.1 Label-noise decay mints concept drift

The question a drift monitor is asked in deployment is not "did the distribution move" but "did
the *rule* move" — because the two answers imply different repairs. Retrain-on-recent is the
remedy for a rule change; it is a mistake when the rule is fixed, since it discards data that is
still valid. Loss-based attribution answers that question with a comparison: a model fit on recent
data versus one fit on recent plus old data, the gap read as evidence that the old rows now
mislead. The comparison has a false-positive channel that, to our knowledge, has not been
reported.

Hold the rule provably fixed and let only the *label noise* decay — early labels noisier than
late ones, the mapping from x to the true label unchanged throughout — and the comparison fires.
On a constructed null of 12,000 rows the raw arm reads **+0.021**, above the 0.02 decision floor
this literature's thresholds live near. That magnitude is not an abstraction: it lands within
0.003 of **+0.024**, the industrial concept positive our own earlier instrument had reported and
we had believed (§3.2). A monitor consuming that reading would retrain on recent data, discarding
correct labels, and would attribute a rule change to a dataset whose rule never moved.

The channel is not a single trick. Adversarial construction against the repair itself found two
more: noise whose *scale depends on x* while decaying (raw **+0.025**), and — from an independent
red-team pass, the denoiser's worst case, since pseudo-label error then concentrates exactly where
the signal lives — noise scale correlated with the rule-carrying feature (raw **+0.026**). All
three fire on the raw arm; all three have a provably fixed rule. The mechanism is elementary once
stated: noisier old labels inflate the loss of any model that fits them, so *removing* old rows
improves recent-window loss for a reason that has nothing to do with the rule. What makes it
dangerous is that the resulting number is indistinguishable, in both sign and magnitude, from the
thing practitioners act on.

Symmetry check: label noise *growing* over time reads **−0.023** — old rows appear to help — so
this is a directional artifact of noise trajectory, not a constant bias.

### 3.2 The instrument diagnoses its own prior positive on real data

Our v2 instrument (raw arm only) reported sberbank-housing, the panel's one regression dataset, as
its sole industrial concept positive: **+0.024 [+0.018, +0.030]**. The constructed null of §3.1
was built at matching magnitude, and the repaired instrument was then pointed back at the real
dataset.

Across window resolutions K ∈ {5, 8, 10, 12, 20} the raw arm fires at K ∈ {8, 10, 12}, at K = 10
reproducing the v2 headline **bit-for-bit** (+0.0239; the raw pipeline's RNG stream is unchanged,
so this is the same signal reinterpreted, not a re-measurement that happened to differ). The old
window's measured noise proxy runs **2.1–2.9×** the recent median at every K. And the denoised arm
— old labels replaced by cross-fitted pseudo-labels — is **significantly negative at every K**
(−0.014 to −0.018, all CIs below zero): with the noise removed, the old rows *help*. The rule did
not change; the early labels are noisier, consistent with 2011–2012 crisis-era Russian housing
prices.

A negative result of this shape invites one objection: that the windows simply lack power. They do
not, and we certify rather than argue it. Under the strict decision rule the cell reaches the
injection control, where a rule planted at reference strength is learnable in this geometry
(in-window R² 0.93) and recovers at **+0.086**; at K = 20 a planted rule recovers **+0.101**
through the same geometry. The same windows that fail to show a rule change do show a rule change
that was put there. The diagnosis is of the minted positive only: a residual rule change *below*
the floor, co-occurring with the noise decay, is not excluded, and we claim nothing sub-floor in
either direction.

We are not aware of a prior case of a drift-attribution instrument diagnosing the *mechanism* of
its own earlier false positive on real data, as opposed to failing to replicate it.

### 3.3 Why a noise gate is not the repair

The obvious fix — measure the noise trajectory, refuse a verdict when old-window noise is high —
is necessary and insufficient, and the battery says so by construction. In the cell where a
rotating rule and decaying noise **co-occur**, the noise gate fires (ratio 3.67) while the rule
genuinely moved; a gate with veto power would suppress a true positive. The instrument therefore
gates and denoises separately: the denoised arm still reads **+0.316** there, 58% of the clean
rotating-rule signal (+0.541), and retention holds at small magnitudes (a 0.8-rad rotation: clean
+0.108, with noise decay +0.062 — attenuated, still firing). The gate flags the confound; only the
denoised arm decides.

Denoising is itself biased — pseudo-label error grows with old-window noise — so its validity
boundary is measured rather than assumed. On pure fixed-rule nulls the denoised arm reads +0.0037
at noise ratio 3.54, +0.0048 at 3.76, +0.0043 at 3.87, +0.0140 at 4.72, and **+0.0259 at 5.71 —
across the decision floor, on a null**. Above ratio 4.7 the instrument abstains. Separation still
exists beyond it (at ratio ≈6 a true rotation reads +0.086 against the null's +0.026, disjoint
CIs), but a threshold there would rest on a single noise family, so we refuse the verdict instead
of calibrating one. Abstention is a measured envelope, not a disclaimer.

`[E2 — detector head-to-head]` *The claim this section still owes: that the channel reaches
type-attributing frames in field use, not only our own probe. The pre-registered test points
DISDE-style reweighting health and the within-overlap decomposition at battery cells whose ground
truth is fixed by construction (`reg_early_noisy`, den +0.0045 at gate 3.53; `reg_xdep_noise`,
+0.0063 at 3.72). Under the field-standard definition (Webb, Gama) noise decay* is *a Y|X change,
so a fire is not by itself an error — the registered question is whether the frames separate the
two mechanisms by magnitude. A local preliminary already weakens the strong version: the frames do
separate by size (rule 0.818 vs noise 0.059, 14×), which refutes "field tools misread this". The
claim that survives is narrower — sign alone does not separate, and a threshold reading calls both
Y|X while the implied repair is right for only one. Fill from `logs/e2_h2h.log`.*

---

## 6. Where the instrument is blind

Sections 4 and 5 establish what the repaired instrument can say. This section establishes what it
cannot, along four axes, each measured rather than asserted. Three are measured with our own
tools; the fourth is settled from outside, against a rule change documented in law.

We put this before the application (§7) deliberately. A verdict of "no rule change" is only worth
its blind spots, and a reader who does not know them cannot price the map that follows.

### 6.1 Class relativity: the same bytes, a different probe, a different verdict

Every component of the instrument — both staleness arms, the noise gate, separability, the
injection control — follows the probe's model class. Swapping that class through the five core
controls separates two properties that are usually conflated. Concept *detection* is class-robust:
all five classes fire on a true rotation (+0.280 to +0.310). Concept/covariate *separation* — the
property a verdict actually rests on — is not. kNN reads pure covariate shift with a fixed rule as
concept (**+0.098**); on fixed-rule prior shift, kNN (+0.048), linear (+0.026) and a two-layer MLP
(denoised +0.042) all false-fire. This is Shimodaira (2000) made empirical: under misspecification
the mixture-ERM optimum moves with P(x), so covariate shift alone manufactures "old data hurts."
Denoising does not repair it — kNN still false-fires at +0.111 on pseudo-labels, because the
misspecification is in the probe, not in the labels.

The neural result deserves its own sentence, because the field's motivating question is whether
deep tabular models have a rule change to exploit: **the first neural probe we tested fails the
separation battery** and is therefore carried as a canary, not as decision grade. We do not
extrapolate that to modern deep architectures at production scale; we claim the direction of the
burden — a probe class earns drift-verdict authority by passing this battery.

On real data the consequence is not hypothetical. **Electricity — the literature's canonical
concept-drift dataset — reads DEPLOYMENT-CONCEPT under two independent canary classes** (linear:
raw +0.023, denoised +0.033, injection recovers +0.190; MLP: raw +0.007, denoised +0.025) and
unidentifiable-with-earned-blindness under the tree-ensemble probe, on identical bytes. Whether a
monitor confirms that dataset's reputation is a property of the monitor. A second canary behaviour
is worth reporting as a warning: on ecom_offers the linear probe fires the *denoised* arm alone
(+0.028) with the raw arm null (−0.002) — a denoiser artifact specific to misspecified probes,
since the linear ĝ_old's systematic error is itself informative to the linear downstream model.
The denoised arm's semantics are class-scoped too.

### 6.2 Power collapses continuously inside the separability gate

The separability threshold D\* = 0.96 routes cells to the injection control, and it reads as a
cliff. It is not one. Just inside it, a planted rule of *fixed* strength recovers at wildly
different magnitudes: cooking_time (D = 0.966) **+0.546**, delivery_eta (D = 0.999) **+0.332**,
Electricity (D = 1.000) **+0.018**, homesite_insurance (D = 1.000) **−0.052**.

Learnability does not explain the ordering — Electricity's planted rule is the *most* learnable in
the panel (in-window AUC 0.976) and still barely recovers. What decays across D is recovery, not
learnability. Over every learnable injection run at decision grade, the rank correlation between D
and recovery is **ρ = −0.47 (p = 0.037, n = 20)**, and the cleanest contrast is within a single
dataset: Electricity's identical planted rule recovers **+0.190 at D = 0.905** under the linear
probe, ten times its recovery at D = 1.000. A certificate therefore certifies that the geometry
had power *at this separability* — not that power is uniform across certified cells.

`[E3 — within-cell D ladder]` *The contrast above still moves D by changing the probe class, which
changes two things at once. The registered test narrows the representation of a single cell
(`--mi-k` ∈ {5, 10, 20, 50}) on the two highest-D certified cells, moving D while holding dataset
and probe fixed. Prediction, committed before the run: D falls monotonically with k (80%);
recovery rises as D falls on delivery_eta (60%). Fill from `logs/e3_k*.log`; if it holds, this
subsection upgrades from a correlation to a controlled ladder.*

### 6.3 Family relativity: the certificate is relative to the class of rule change

An injection certificate says the geometry could carry *a* planted rule. It cannot say the
geometry could carry *any* rule, and the difference is measurable. Sweeping the injection across
signal families on the same windows: on weather's full deployment span, a **low-variance** rotation
recovers **+0.128** and an **interaction-borne** rule recovers **+0.046**, both clearing the strict
rule — while a **subpopulation-local** rule, comfortably learnable in-window at **R² 0.560** (the
gate is 0.20), **fails to recover at −0.050**, reproducing on confirmatory seeds (R² 0.523,
−0.042).

That combination is the point. Learnability and recoverability come apart: the instrument has
power against changes in feature *direction* and none against changes in subpopulation
*membership*, in the same geometry, at the same strength. Every "verified no-concept" in this
paper therefore carries an explicit family denominator, and the two certificates that stand on the
industrial panel (weather, homesite_insurance) stand against the families actually tested — weather
recovering +0.183 / +0.193 on fresh seeds under the low-variance carrier and +0.083 / +0.078 under
the interaction family.

### 6.4 Metric relativity: falsified against an externally documented rule change

The three axes above are the instrument measured with the instrument. This one is settled from
outside, and it is the sharpest thing we can say about our own tool.

The ACA Medicaid expansion is a rule change documented in law: Pennsylvania implemented it on
2015-01-01, Texas never adopted it. On the ACS public-coverage task over 2014–2018 the effect is
visible in the raw data — Pennsylvania's positive rate moves 0.234 → 0.268 → 0.293 → 0.305 →
0.306 with the step at the implementation year, while Texas stays within 0.183–0.185 throughout.

**The probe read both states null.** Pennsylvania: D = 0.534 — fully identifiable, not a gated
abstention — gate 0.91, raw −0.011 [−0.011, −0.010], denoised −0.009 [−0.010, −0.008]. Texas:
D = 0.523, raw −0.015, denoised −0.011. And the Pennsylvania null carries the **tightest
detectable bound anywhere in this paper (δ = 0.00083)**, so this is a confident zero rather than an
underpowered one — which makes the miss sharper, not softer.

The mechanism was registered before the run and is confirmed by it. Our binary score is AUC, which
is rank-based, and an eligibility threshold can move a large mass of people across a decision
boundary **without reordering anyone**. Measured on the same rows with proper scores, the
treatment-versus-control separation widens from 3.1× under AUC to 6.3× under Brier, 6.6× under
log-loss and 15× under a rule-movement divergence. But it does not become large: every reading
stays below the decision floors of both instruments, and the control state also returns
placebo-significant gaps, so only the treatment/control *ratio* is interpretable. The honest
statement is therefore not "the metric explains the miss" but **"the metric compresses the signal
six- to sevenfold, and even uncompressed this policy change sits under our floor."**

Two consequences, and we take both. Every null in §7 is scoped to a rank-based score — the right
scope for a ranking or triage system, the wrong one for the fixed-threshold eligibility systems
this very task represents. And the 0.02 decision floor, calibrated on planted effects and no-drift
anchors, is **larger than a real population-scale policy rule change**. We record that as a
measured criticism of the constant (Appendix C) and do not act on it: changing a pre-registered
threshold after seeing results is precisely what this paper argues against.

The scope statement is fixed and we hold to it (`PREREG_ACS_EXTENSION` §6(b)): the finding is not
"there is no drift in ACS" but **"this instrument does not see a real rule change of this size."**
We do not reinterpret it as a property of the data after the fact.

`[E1 — proper-score probe]` *Whether the probe itself fires on Pennsylvania under Brier and
log-loss. Diagnostic only: changing the score changes the estimand and voids the AUC-calibrated
floor, so no E1 run is a map verdict, and the reading rule fixed before execution discards the
verdict label and reads only arm magnitudes and the PA-vs-TX contrast. Committed predictions: PA
staleness turns positive under a proper score 55%; clears the floor 25%; TX stays ≤ 0 under every
metric 70%. Fill from `logs/e1_*.log`. If PA fires, this subsection gains a repair direction —
blindness localised to the metric and removable by changing it — while §7's nulls keep their
AUC scope. If PA stays null, the blindness is deeper than the score and this subsection says so.*
