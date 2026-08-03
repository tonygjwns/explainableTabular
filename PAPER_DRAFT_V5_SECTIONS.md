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

### 3.4 Does the channel reach the frames practitioners actually use?

The sections above measure our own probe. The obvious objection is that a field tool built on a
different principle — reweighting-based decomposition of the shift into X-side and Y|X-side terms
— would not be fooled. We pointed such a frame at battery cells whose ground truth is fixed by
construction, under conditions favourable to it (covariate overlap intact, cov-AUC 0.500, effective
sample size 71.3% after reweighting).

| cell | truth | Y\|X-side gap |
|---|---|---|
| rotating rule (binary) | rule moved | **+0.4345** |
| rotating rule (regression) | rule moved | **+0.8207** |
| label-noise decay, fixed rule | rule fixed | +0.0576 |
| x-dependent noise decay, fixed rule | rule fixed | +0.0615 |
| stationary (regression) | rule fixed, no noise trend | −0.0208 |

The strong version of our claim is refuted by this, and we say so: the frame does **separate the
two mechanisms by magnitude**, 7.5–14× — a practitioner reading the number, not the label, is not
misled. What survives is narrower and still costly. **Sign does not separate.** Both noise-decay
cells return a *positive* Y|X-side gap where the true stationary cell returns a negative one, so
under the field-standard definition (Webb et al.; Gama et al.) — under which noise decay *is* a
Y|X change — a threshold reading files both as the same event. The two events have different
repairs: retrain-on-recent is right for one and discards valid labels in the other. Attribution by
sign or by threshold is what fails; attribution by calibrated magnitude, against a null of the kind
built in §3.1, is what a monitor would need.

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

We tried to convert that correlation into a controlled ladder and could not, which is worth
reporting because the failure is informative. Narrowing a single cell's representation
(mutual-information top-k, k ∈ {5, 10, 20, 50}) moves D monotonically while holding dataset and
probe fixed — delivery_eta 0.521 → 0.540 → 0.555 → 0.801, homecredit_default 0.721 → 0.796 →
0.833 → 0.912, against 0.999 and 1.000 at full representation. But the narrowed cells fall
*below* the routing threshold, so the injection control never runs and there is no recovery to
read: the handle that moves D also moves the cell out of the region where power is measured. The
within-cell recovery ladder therefore remains unmeasured, and we state the correlation as a
correlation. (Registered predictions, for the record: D was predicted to fall with k and rises;
the recovery prediction was unmeasurable for the reason just given.)

Two things the ladder did show. The denoised reading itself moves with representation on a fixed
cell (delivery_eta −0.002 → −0.013; homecredit_default **+0.013 [+0.010, +0.016]** at k = 5, where
D = 0.721 puts it in the identifiable region, decaying to −0.010 at k = 50) — so representation
choice sets both whether a cell is identifiable *and* the sign of what is read there. And in the
adjacent low-D regime the picture is the opposite of the high-D panel: at D ≈ 0.49 on the river
anchors, all four signal families recover (+0.076 to +0.238, all learnable at 0.916–0.968),
including the subpopulation-local family that fails at high D in §6.3. Representation dependence
is a known hazard for this literature's estimands; here it is a measured one.

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
*membership*, in the same geometry, at the same strength.

The blind spot is a property of family *and* geometry rather than of the family alone, and an
anchor sweep pins that down: on the single-switch river anchors, where separability is low
(D ≈ 0.49), the subpopulation-local family recovers on all three anchors (+0.119, +0.079, +0.076,
learnable at 0.937–0.963), as do the other three (+0.135 to +0.238). It is also, consistently, the
*smallest* recovery of the four — the family that falls off first as separability rises, and the
one that has already fallen off in the high-D industrial geometry. So the scope of a certificate is
a joint statement about which rule families were tested and at what separability, and §7 reports
both. Every "verified no-concept" in this
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

**The blind spot is localised, and the repair direction is measurable.** Re-running the probe
itself under proper scores — diagnostic only, since changing the score changes the estimand and
voids the AUC-calibrated floor, so the reading rule fixed before execution discards the verdict
label and reads arm magnitudes and the PA-versus-TX contrast — flips the sign of the Pennsylvania
reading:

| state | AUC (raw / denoised) | Brier | log-loss |
|---|---|---|---|
| Pennsylvania (expanded) | −0.011 / −0.009 | −0.005 / **+0.002** | −0.018 / **+0.021** |
| Texas (never adopted) | −0.015 / −0.011 | −0.005 / −0.002 | −0.023 / **+0.006** |

The denoised arm turns positive under both proper scores on the treatment state, reaching +0.021 —
the size of the decision floor — under log-loss, against +0.006 on the control, a 3.5× contrast in
the right direction. The raw arm stays negative under every metric. So the mechanism registered
before the run is confirmed on the instrument itself and not only through the external lens: what
the probe could not see, it could not see *because of the score*.

Three things this does not license, all of which we state rather than absorb. The proper-score arm
carries **no certificate** — the injection planted in that run is unlearnable (learnability −0.185),
so this reading has no power guarantee behind it. The **control state also moves positive** under
log-loss (+0.006), so only the ratio is interpretable, never the absolute value; the honest summary
of §6.4 remains that the metric compresses the signal six- to sevenfold and the residue still sits
under the floor. And +0.021 "clearing 0.02" compares against a constant calibrated in AUC units,
which is not a verdict; we record it and leave the constant alone.

The scope sentence therefore gains a clause and loses none of its force: not "there is no drift in
ACS", but **"this instrument does not see a real rule change of this size through a rank-based
score; through log-loss it sees something floor-sized, uncertified, and shared with its control."**
