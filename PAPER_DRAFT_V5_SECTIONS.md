# PAPER V5 — full section pass (§1–§9)

**Status.** Draft sections for the instrument-first reframe of `PAPER_V5_SKELETON.md`, written
2026-08-03. `PAPER_DRAFT_V4.md` / `_KO.md` / `paper/main.tex` are untouched and remain the live
manuscript; abandoning V5 means deleting this file. **No new claims**: every reading below is
committed in an artifact, and all 230 of them pass the artifact cross-check
(`scripts/audit_paper_numbers.py`). What changes is which of them carry the paper.

Four sections are drafted here. **§1** rewrites the title, abstract and introduction around the
noise channel — the subject changes from "industrial data has no drift" to "drift attribution is
fooled by noise", and the map becomes the fourth contribution rather than the first. **§3** promotes
V4 §5.3 and the battery's noise cells into the paper's lead. **§6** collects V4's Limitations 1, 4
and 6 plus the ACS falsification into a standalone map of the instrument's blind spots. **§7**
demotes V4 §5 and §6 — the industrial map — to an application of the certified instrument, with its
numbers unchanged and its qualifiers moved from footnotes into the sentences that carry them.

The three day-4 slots are filled (§3.4 from E2, §6.2 from E3, §6.3 from E4, §6.4 from E1); read-out
and pre-committed predictions are in `PREREG_DEPLOYMENT_V2.md` §19 and
`PREREG_ACS_EXTENSION_2026-07-31.md` §12.

**§2, §4 and §5 are relocation specs, not prose.** Those three are V4 §2, §3 and §4 moving mostly
unchanged, and copying their text into a second file would create two sources of truth for the same
numbers — the exact drift a commit was just spent repairing. Each spec says what transfers verbatim,
what must change, and what must *not* be restated because it has been promoted elsewhere. **§8 and
§9 are drafted in full**, because the discussion genuinely shrinks once the limitations become §6,
and reproducibility gains the re-gate discipline and a second disclosure.

**Still to do**: fold these into a single V5 manuscript (mechanical for §2/§4/§5, per the specs),
then the KO mirror and LaTeX, which should wait until the English is settled. Appendices A–C and
B.4–B.6 carry over from V4 unchanged.

---

# Label-Noise Decay Mints Concept Drift: A False-Positive Channel in Loss-Based Drift Attribution, and What a Certified Instrument Looks Like

## Abstract

A drift monitor's job is not to report that a distribution moved but to say *what* moved, because
the repairs differ: a changed rule P(y|x) calls for retraining on recent data, while moving
covariates under a fixed rule do not. We show that the loss-based comparison this attribution
usually rests on has a false-positive channel that has not, to our knowledge, been reported.
**Hold the rule provably fixed and let only label noise decay over time, and the comparison
returns a concept-drift verdict** — at +0.021 on a constructed null, within 0.003 of a real
industrial positive (+0.024) that our own earlier instrument had reported and we had believed.
Two further channels, found by adversarial construction against our own repair, fire the same way
under equally fixed rules. We repair the instrument — a cross-fitted *denoised staleness* arm, a
per-window noise gate, and an abstention envelope measured to the point where the denoiser itself
crosses the decision floor on a null — and validate it on a 14-cell pre-registered battery in which
rule change and noise decay co-occur and must still fire (it does, at 58% of clean power).
Verdicts are earned rather than assumed, through three certificates: separability, injection, and
learnability. The repaired instrument then **diagnoses the mechanism of its own prior false
positive on real data** — not a failure to replicate, but an identified cause, with the power to
have seen a rule change certified in the same windows. We also measure where the instrument is
blind, along four axes, the last of which is settled from outside: against the ACA Medicaid
expansion, a rule change documented in law, **the probe read null in a fully identifiable regime at
the tightest detectable bound in the paper**, because its score is rank-based and an eligibility
threshold moves mass across a boundary without reordering anyone; under proper scores the same
windows read +0.021 on the treated state against +0.006 on the control. Applied to eight
industrial datasets (TabReD) with confirmatory fresh-seed replication (10/10 stable), the
instrument finds no exploitable mean-rule drift above the per-dataset detectable floor for the
deployed tree-ensemble class (0/8 audited; 0/5 among cells whose identifiability is certified).
A head-to-head against a type-attributing frame shows the channel is not ours alone: the frame
separates the two mechanisms by magnitude (7.5–14×) but **not by sign**, so a threshold reading
files two events with different repairs as the same event. We release the instrument, the battery
and the audit trail, and argue that drift-type attribution without identifiability certificates —
the current default — is unreliable.

---

## 1. Introduction

A team watching a deployed tabular model sees its recent loss rise and has to choose. Retrain on
recent data only, discarding older rows as stale — the right move if the labeling rule has changed.
Or keep everything, because the rule is intact and the older rows still carry correct labels — the
right move if only the covariates moved. Choosing wrongly is expensive in both directions, and
choosing is what drift-type attribution is for.

The comparison that attribution usually rests on is simple enough to be trusted without checking:
fit a model on recent data, fit another on recent plus old data, and read the gap on a future
window. If old rows now mislead, adding them hurts. We call this quantity *staleness harm*, and it
is deployment-native in a way importance-weighted decompositions are not — the recent portion is
identical in both arms, so covariate coverage cancels by construction, and nothing needs density
ratios or overlapping supports, which is exactly what fails in industrial feature spaces.

**The comparison has a false-positive channel.** Its informal justification — correct labels cannot
hurt — is false for finite-capacity empirical risk minimization. Hold the rule provably fixed, let
only the *noise* on the labels decay over time, and the comparison fires: unweighted ERM is not
efficient under heteroscedasticity, so noisier old rows inflate the loss of any model that fits
them, and removing them improves recent-window loss for reasons that have nothing to do with the
rule. On a constructed null this reads **+0.021** — above the 0.02 floor this literature's
thresholds live near, and within 0.003 of **+0.024**, an industrial positive our own v2 instrument
had reported and we had believed. A monitor consuming that number would retrain on recent data and
throw away correct labels, while telling its owners the rule had changed. Two further channels,
constructed adversarially against our own repair, fire the same way: x-dependent noise whose scale
decays (+0.025), and noise correlated with the rule-carrying feature (+0.026), the denoiser's worst
case. Growing noise reads −0.023, so the artifact is directional rather than a constant bias.

**Two more ways the naive instrument lies.** The natural certificate for "a null here is
uninformative" is a window-separability AUC — can a classifier tell old rows from future rows? —
and computed row-wise it saturates to 0.994–1.000 under exact duplicates, near-duplicates and
entity cohorts with *zero* covariate shift: the classifier memorizes rows, not distributions, and
eight of ten real datasets read exactly 1.000 under the naive gate. And the concept/covariate
separation is hypothesis-class-relative: a kNN probe reads pure covariate shift as concept drift
(+0.098), a linear probe reads prior shift as concept drift (+0.026), while tree ensembles pass all
controls. That is Shimodaira's misspecification result made empirical, and it means "old data
hurts" can be manufactured by geometry alone.

**The repair, and the discipline it needs.** Each failure gets a repair validated by execution
(§4–§5): a *denoised staleness* arm that replaces old labels with cross-fitted within-window
predictions — under noise-only drift the pseudo-labels are approximately correct and the harm
vanishes; under a changed rule they still encode the old rule and the harm persists — together
with a per-window noise gate whose validity envelope is *measured* rather than assumed, to the
point where the denoiser itself reads +0.026 on a null and the instrument abstains; a group-aware
separability estimate that deflates memorization while leaving honest drift intact; and
learnability-gated injection controls that turn "this dataset is unmeasurable" from an assumption
into a demonstration. All verdicts are scoped to the tree-ensemble class, with linear and kNN
probes carried as canaries whose false fires are reported rather than hidden.

**The instrument turned on itself.** Pointed back at the dataset that produced our prior positive,
the repaired instrument reproduces the v2 raw reading bit-for-bit, measures old-window label noise
at 2.1–2.9× the recent median, and returns a *significantly negative* denoised arm at every window
resolution: with the noise removed the old rows help. The same windows recover a rule planted at
reference strength (+0.086, learnable at R² 0.93), so the negative is certified rather than
argued. We are not aware of a prior case of a drift-attribution instrument diagnosing the
*mechanism* of its own earlier false positive on real data, as opposed to failing to replicate it.

**And its blind spots, measured.** A null is worth its blind spots, so we map ours (§6) along four
axes: probe class, separability, rule family, and metric. The last is settled from outside. The ACA
Medicaid expansion — implemented by Pennsylvania on 2015-01-01, never adopted by Texas — is a rule
change documented in law and visible in the raw positive rates (0.234 → 0.306 against a flat
0.183–0.185). **The probe read both states null**, in a fully identifiable regime and at the
tightest detectable bound anywhere in this paper, so a confident zero rather than an underpowered
one. The mechanism was registered before the run: our binary score is AUC, which is rank-based, and
an eligibility threshold moves a large mass across a decision boundary without reordering anyone.
Under proper scores the same windows turn positive on the treated state (+0.021 under log-loss)
against +0.006 on the control — the blindness localised, with a repair direction, and with the
three things that reading does not license stated alongside it.

**The application.** With the instrument certified and its blind spots mapped, we spend it on the
question that motivated building it (§7): a pre-registered audit of eight TabReD datasets,
Electricity and the INSECTS streams, with confirmatory fresh-seed replication (10/10 stable). The
result is an identifiability map rather than a detection table — **0/8 audited, 0/5 among the cells
whose identifiability is certified**, the sole robust positive being designed drift, with two cells
refusing a certificate outright rather than being counted as evidence. Anchor streams fix the
sensitivity profile: monotone and single-switch rule changes fire 9/9, and recurring regimes are
correctly silent with a negative-recency fingerprint that is impossible under one-way drift.

**Contributions.**

1. *A false-positive channel in loss-based drift attribution* (§3), executed rather than argued:
   label-noise decay under a provably fixed rule mints concept verdicts at magnitudes matching real
   borderline positives, with two further channels found by adversarial construction, and a
   head-to-head showing a type-attributing frame separates the mechanisms by magnitude but not by
   sign.
2. *A repaired, abstaining instrument* (§4–§5): denoised staleness with a noise gate and a measured
   validity envelope, group-aware separability, learnability-gated injection certificates,
   class-scoped verdicts with canaries — validated on a 14-cell pre-registered battery including
   the adversarial combination in which rule change and noise decay co-occur (fires at 58% of
   clean power).
3. *A measured map of an instrument's blind spots* (§6), including one settled against external
   ground truth: class relativity (a linear monitor flips the canonical Electricity dataset to
   concept drift on identical bytes), continuous power collapse inside the separability gate
   (ρ = −0.47), family relativity (a subpopulation-local rule learnable at R² 0.560 and
   unrecovered at −0.050), and metric relativity (the ACS falsification).
4. *A pre-registered identifiability map of industrial tabular ML* (§7): 0/8 audited and 0/5
   certified, one diagnosed false positive with mechanism, per-dataset certificates and
   detectable-effect bounds, 10/10 confirmatory stability and 10/10 cross-class agreement.

**What we do not claim.** We do not claim concept drift is absent from industrial tabular data:
several cells are certified *blind*, and blindness is not absence. We do not claim the
identifiability theory is new — that overlap failure blocks nonparametric identification is
classical, and class-relativity is established; our contribution is the executed channel, the
repaired instrument, and the certified map. We do not claim model-agnosticism: every verdict is
relative to the deployed hypothesis class, which we argue is the only honest way to state
drift-type attribution at all. We do not claim the noise channel is the only way this comparison
fails, only that it is one nobody had reported and that it is large enough to have fooled us. And
we do not claim our own instrument is trustworthy everywhere — §6 is the list of places where it is
not, and one entry on that list was written by a state legislature rather than by us.

---

## 2. Related work — relocation spec (V4 §2, near-verbatim)

V4 §2 transfers as written, with three edits that the reframe forces. It is not reproduced here:
duplicating prose into a second file is how the number drift we spent a commit fixing gets in.

| paragraph | action |
|---|---|
| Shift-type maps on tabular data | verbatim |
| Identifiability and class-relativity | verbatim |
| **Old data harming** | **promote and sharpen.** The sentence "we are not aware of prior work that … reports the label-noise-decay false-positive channel" is a related-work aside in V4 and the paper's headline claim in V5. State it once here, in the form §1 states it, and make the Gama/Webb definitional point explicit: under the field-standard definition noise drift *is* drift, which is exactly why an instrument claiming to detect *exploitable rule change* must separate the two. |
| Pseudo-labels, denoising, cross-fitting | verbatim |
| **Drift detectors and their evaluation** | **correct a claim that is no longer true.** V4 says "We do not run these detectors head-to-head on the audited cells." We now do run a type-attributing frame head-to-head, on battery cells whose ground truth is fixed by construction (§3.4). The paragraph must say what remains true — a loss-stream detector answers "did anything change?" and is type-blind by construction, so it does not bear on type attribution — and then point at §3.4 for the frame that *does* attribute types and separates the mechanisms by magnitude but not by sign. Leaving V4's sentence in place would be a false statement about our own evidence. |
| Malware | verbatim; the EMBER read now lives in §7.3 |

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

## 4. The repaired instrument — relocation spec (V4 §3)

V4 §3 transfers whole, renumbered, with the cascade figure and the terminology table intact. Two
things change.

- **The reading aid changes its referent.** V4's walk-through uses sberbank as "one real cell that
  exercises the vocabulary". That cell is now the subject of §3.2, three sections earlier, so the
  aid should point back to it rather than introduce it: *the cascade returns NOISE-DRIFT-CONFOUNDED
  on the cell of §3.2 because the raw arm fires, the gate is on, and the denoised arm is negative;
  had all three aligned inside the envelope it would have returned DEPLOYMENT-CONCEPT subject to an
  injection control.*
- **The envelope is already spent.** V4 §4.2's envelope measurement is used in §3.3 to argue that a
  noise gate is not the repair. §4.2 here keeps the *definition* (ratio 4.7, abstain above) and the
  terminology-table row, and points to §3.3 for the measurement rather than repeating the numbers.

Subsections, in order: 4.1 setting and estimand (V4 §3.1, including the DISDE term-ii statement and
the positivity boundary) · 4.2 the denoised arm and the noise gate (V4 §3.1 second half) · 4.3 the
three certificates — separability, injection, learnability (V4 §3.2) · 4.4 the decision cascade and
its scope (V4 §3.3).

---

## 5. Validation: the battery and the gate discipline — relocation spec (V4 §4, reduced)

This section is smaller in V5 than V4 §4 was, and the reduction is the part a merge will get wrong,
so it is spelled out.

**Comes here.** The 14-cell pre-registered battery table (V4 §4.1) in full, with its protocol line
(n = 12,000, d = 10, K = 10, 5 seeds, full cascade) and the "three cells carry the argument"
reading — but the *argument* those three cells carry is now §3's, so the text here states what the
battery is for: an entry gate the instrument must pass before touching real data. The floor
comparability paragraph (V4 §4.1, closing) comes here unchanged, including the per-metric
rescaling sensitivity check that moves no cell.

**Does not come here.** V4 §4.2 (the envelope) is in §3.3. V4 §4.3 (the model-class matrix) is in
§6.1. Neither should be restated; §5 points to both.

**New here, and it belongs to validation rather than to results.** The battery is not run once. It
is the re-gate any change to the instrument must pass, and the discipline has now been exercised
under adversarial conditions: when two opt-in diagnostic flags were added, the battery was re-run
and required to come back **bit-identical** against the reference environment — all fourteen cells,
every field, verdicts included — with the pre-committed rule that a mismatch reverts the change
rather than explaining it. It did (`PREREG_DEPLOYMENT_V2.md` §19.0). That is the operational
content of "pre-registered instrument": the gate has teeth only if it can fail after the code is
already written.

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

---

## 7. Application: auditing an industrial panel

Everything above is about an instrument. This section spends it on the question that motivated
building one: does industrial tabular data contain rule change a deployed model could exploit?
The answer is a map, and the map is worth exactly what §6 says it is worth — every number below is
scoped to a tree-ensemble probe class (§6.1), to the separability at which its certificate was
earned (§6.2), to the rule families that certificate was tested against (§6.3), and to a
rank-based score (§6.4). We state the qualifiers with the result rather than after it.

### 7.1 Protocol and result

Eight TabReD datasets (Rubachev et al., 2025; train segments of the official temporal splits),
Electricity, and INSECTS (incremental-balanced) — HGB probe, K = 10, ten exploratory seeds (0–9),
then a **confirmatory rerun with fresh seeds (100–109)**; any verdict that moves between the two is
reported unstable and barred from claims. The decision cascade, thresholds, seed protocol and
aggregate reading were committed before execution. The prior v2-era industrial positive was
pre-registered as a *retraction candidate with a prediction* (NOISE-DRIFT-CONFOUNDED or
denoised-null) and a survival battery was pre-specified in case the prediction failed; §3.2 reports
what happened. Both the primary rule (CI > 0 and mean > floor) and the strict rule (CI > floor) are
computed, and rule-sensitive cells are flagged. All ten cells are confirmatory-stable.

| dataset | verdict (= confirmatory) | raw | denoised | gate | D | Rec. | certificate |
|---|---|---|---|---|---|---|---|
| insects | **DEPLOYMENT-CONCEPT** | +0.135 / +0.129 | **+0.152 / +0.145** | 1.24 | 0.844 | +0.162 | injection recovers |
| sberbank_housing | **NOISE-DRIFT-CONFOUNDED** | +0.024 / +0.033 (fires) | **−0.015 / −0.011** | **2.11 / 2.21 (fires)** | 1.000 | — | diagnosed (§3.2) |
| cooking_time | INJECTION-RECOVERED | −0.011 | −0.018 | 0.92 | 0.966 | **+0.546** | **verified no-concept** |
| delivery_eta | INJECTION-RECOVERED | −0.012 | −0.014 | 0.89 | 0.999 | **+0.332** | **verified no-concept** |
| maps_routing | NO-STRONG-CONCEPT | −0.008 | −0.010 | 1.01 | 0.578 | n/a | identifiable region |
| elec2 | UNIDENTIFIABLE | +0.001 | +0.001 | 0.66 | 1.000 | +0.018 | **blindness earned** |
| ecom_offers | UNIDENTIFIABLE | −0.001 | −0.007 | 1.25 | 1.000 | — | *vacuous* — injection unlearnable (0/4 families) |
| homecredit_default | UNIDENTIFIABLE | −0.007 | +0.004 | 1.29 | 1.000 | — | *vacuous* (0/4 families; 80 proxy features stripped) |
| weather | INJECTION-RECOVERED † | −0.012 | −0.011 | 0.93 | 0.994 | **+0.183** | **verified no-concept** (2/3 learnable families) |
| homesite_insurance | UNIDENTIFIABLE-INERT | −0.004 | −0.002 | 1.14 | 1.000 | −0.001 | **blindness earned** (2/2 learnable families, stable) |

*D* = the group-aware separability median (D\* = 0.96 routes to the injection control); *Rec.* =
recovered staleness of a rule planted at reference strength; "—" = the injection was unlearnable in
every family tried, so no recovery number is admissible, and *n/a* = the cell never routes to the
control. Recovery moves by less than 0.01 between seed sets everywhere. † weather's reference-family
rule is unlearnable in its geometry; recovery is measured under the low-variance carrier, where it
reproduces on fresh seeds (+0.183 / +0.193), and under the interaction family (+0.083 / +0.078).
The "(m/n families)" counts are the family denominator of §6.3, printed so a certificate is never
read as stronger than the class of rule change it was tested against.

**The aggregate, as pre-registered and as it is worth.** Relative to the tree-ensemble class, the
number of industrial datasets with exploitable mean-rule drift above the per-dataset detectable
floor is **0/8** audited. Over the datasets whose identifiability is *certified* it is **0/5**:
ecom_offers and homecredit_default end with no certificate — their planted probe rule is
unlearnable under every family tried — and so contribute no evidence in either direction. We report
both wherever the aggregate appears; the first is what was pre-registered, the second is what it is
worth.

Three readings deserve their own sentence. The only concept positive is the designed-drift stream,
where the denoised arm *exceeds* the raw arm — pseudo-labels encode the old rule cleanly once label
noise is removed, the signature of genuine rule change and the mirror image of §3.1's channel. Two
of the four blindness claims a naive protocol would have granted are *vacuous*, a distinction
invisible without the learnability gate. And the strongest cells in the map are, counter-
intuitively, the two verified no-concept certificates: geometry demonstrably had power (+0.546 and
+0.332 recovery of a planted rule) and the real staleness is still null. What that power does *not*
mean uniformly is the subject of §6.2.

### 7.2 Across the deployment gap

The map covers the train segments of TabReD's official temporal splits, which is where a
practitioner would fit but not where the model is judged. Re-running all eight cells with train,
validation and test concatenated on the shared normalised timestamp — so the windows cross the
held-out gap the official split withholds — leaves **8/8 verdicts identical between exploratory and
confirmatory seeds and no industrial cell firing CONCEPT**. The aggregate is therefore not
re-scoped to train segments.

Three cells move in ways worth recording rather than burying. sberbank's raw arm no longer clears
the floor over the longer span, so the noise diagnosis of §3.2 is explicitly scoped to the train
segment. cooking_time's separability *falls* below D\* (0.927), against our expectation that a
longer span would raise it, leaving an identifiable null that needs no certificate. And
homecredit_default's denoised arm reads **+0.018** — 89% of the decision floor, the largest
positive-direction number anywhere in this paper — with a certificate that is *earned* here rather
than refused, so it reads as "measured and under the floor" rather than "not measured". That cell
is the closest thing this panel has to a positive, and we flag it as the place a larger-N or
finer-window follow-up should look first.

### 7.3 Anchors and the sensitivity profile

A map of nulls is only as informative as the sensitivity of the instrument that drew it, so we ran
it over 23 synthetic river streams (SEA / Agrawal / STAGGER / Sine / Hyperplane; no-drift, abrupt
single-switch, gradual and reoccurring variants) and all seven INSECTS variants (real sensor data,
lab-controlled temperature drift).

**Monotone and single-switch rule changes fire: 9/9.** River: agrawal_abrupt +0.045,
agrawal_gradual +0.047, stagger_abrupt, sine_abrupt +0.031, hyperplane_incremental +0.112 (with the
gate correctly co-flagging its noise component), sine_reoccur2 +0.047. INSECTS: gradual-balanced
+0.092, gradual-imbalanced +0.069, incremental +0.135 — in every firing cell denoised is at least
raw, and the injection positive control recovers. Weak switches (SEA's threshold nudge) land in the
no-evidence band with consistent sign and are reported with their detectable floors.

**Recurring regimes are correctly silent — and fingerprinted.** INSECTS abrupt variants
(oscillating temperature) read *negative* staleness (−0.070, −0.027) with **negative recency gain**
(−0.058, −0.023): the window adjacent to the test predicts it *worse* than the oldest window does.
Negative recency is impossible under one-way drift; it is the signature of a regime that has
returned, and it replicates across river's reoccurring cells and INSECTS incremental-reoccurring.
This is a scope statement, not a defect: the lens answers the deployment question — does old data
harm a model trained today? — and when old regimes recur, old data genuinely does not harm. A
monitor built on this lens will not *detect* recurring drift; it will correctly say old data is
safe to keep, and the negative-recency flag says why. On the audited panel that flag is silent:
every industrial cell reads recency gain at or above zero (+0.000 on maps_routing to +0.325 on
sberbank-housing). The flag certifies recurrence only at scales that dominate the evaluation
horizon — our own sine_reoccur2 anchor returns to its initial regime in the final ~22% and reads
recency +0.30 — so it rules out horizon-dominating recurrence and says nothing about late, brief or
sub-window-periodic returns.

**Calibration of the no-evidence band.** Two of five no-drift anchor streams produce
"CI-significant" sub-floor denoised positives at 1/10 to 1/20 of the decision floor: seed-level CIs
on overlapping subsamples are anti-conservative at tiny magnitudes. The sub-floor band is therefore
read as *no evidence* everywhere in this paper — a calibration the anchor suite forced and the
pre-registration records.

**Two external cells.** On malware (EMBER, 2018-dense monthly windows) the instrument returns a
certificate-grade DEPLOYMENT-DECAY-COVARIATE: D = 0.834 (identifiable), raw −0.008 [−0.009,
−0.007], denoised −0.004, gate quiet, recency gain +0.031 above the floor, detectable floor 0.0013.
Malware's temporal degradation is real and recency-recoverable, but it is coverage-driven — new
families appear, old labels do not rot — and the lens draws exactly that distinction: keep the old
data, expect decay anyway, retrain for coverage. On ACS income across *years* (California,
2014–2018) the instrument returns the map's best-powered null: NO-STRONG-CONCEPT, raw −0.008,
denoised −0.007 (both CIs negative), gate quiet, D = 0.515, recency near zero, detectable floor
0.0008 — on the same task whose *spatial* axis is reported to exhibit prevalent Y|X-shift. The
axis, not the dataset, determines the shift type. That cell also passes a real-data prior-shift
control: the fixed $50k threshold under inflation produces a monotone positive-rate ramp (0.36 to
0.42) and the instrument does not misread it as concept. The complementary ACS reading — the one
where a documented rule change went unseen — is §6.4, and the two should be read together: the
instrument is well powered on this task and still metric-blind to a particular kind of change on it.

---

## 8. Discussion

**What this paper is.** An instrument, its failure anatomy, a measured map of its blind spots, and
one application. The claim we defend is narrow and, we think, load-bearing: **drift-type
attribution without identifiability certificates is unreliable**, not as an argument from
principle but because we built the uncertified version, believed one of its outputs, and then
identified the mechanism that produced it.

**Limitations.** The four axes along which the instrument is blind are not listed here — they are
§6, promoted out of this section because a limitation you can measure is a result. What remains are
the limits of the *study*.

(1) **Scale.** The probe trains on N ≤ 6,000 rows per arm; industrial models train on orders of
magnitude more. A rule change exploitable only at much larger N is invisible here, so every δ bound
and both aggregates are statements *at probe scale*. A first δ(N) sweep raises the arm cap to the
window-geometry ceiling (N ≈ 14k–24k, up to 4× the headline scale) on the three largest null cells:
every verdict is unchanged and no reading trends toward the floor. Production-scale N beyond that
remains future work, and we flag rather than dismiss the possibility that the map changes there.
Scale enters a second way: because the arms differ in size (N versus 2N), a sample-size gain is
folded into every staleness reading — negligible on binary cells, but 61% of the decision floor on
regression, which effectively raises the detectable floor on the map's five regression cells. We
report those nulls with the displacement stated rather than absorbed.

(2) **Window geometry.** K = 10 rolling windows with a fixed early anchor: drift at time scales far
below the window width is averaged away, and the recurrence fingerprint of §7.3 certifies only
horizon-dominating returns. The injection control partially measures this, since its recovery
varies with K.

(3) **No naturally occurring industrial positive.** The single robust positive on the panel is
lab-designed drift. We found no industrial cell against which to calibrate magnitude — that is
simultaneously the map's finding and its weakness, and the attempt to close it from outside
(§6.4) instead produced a falsification of the instrument. We report that as the outcome it was.

(4) **Benchmark curation.** TabReD is curated to be temporally splittable; external validity to
industrial data at large is an inductive step we flag rather than take.

(5) **Deep architectures.** The panel's neural probe fails the separation battery and is carried as
a canary (§6.1). Modern deep tabular architectures at production scale remain untested. We claim
the direction of the burden rather than the conclusion: a probe class earns drift-verdict authority
by passing the battery of §5, and the first neural probe we tested does not.

**On process.** This project's ledger records nine positive findings that dissolved under scrutiny
before the present result. The ninth dissolution is §3.2, and it is the only one the measuring
instrument itself diagnosed. The methodological claim we stand behind is that the combination that
finally produced a stable result — executed adversarial nulls before real-data claims,
pre-registered cascades with commit-timestamped predictions, confirmatory fresh-seed replication,
certificates instead of assumptions, and canary probes — is cheap relative to the cost of the
dissolutions it prevents. Two of this paper's own results exist only because that discipline was in
place: a prediction registered before execution was falsified by the run (§6.4, and the control
state's behaviour in the proper-score follow-up), and a re-gate caught a battery executed under the
wrong interpreter (§9). We release the audit trail as part of the artifact, dissolutions included.

**What a practitioner should take.** Three things. A loss-based drift verdict should not be acted
on without a noise reading on the old window, because the two are confusable at the magnitudes
people act on. A null should not be reported without a power certificate, because "we saw nothing"
and "we could not have seen anything" are different sentences and only one of them is evidence.
And a verdict should carry its probe class, its separability, its rule family and its metric, since
each of the four can flip it on identical bytes.

**Future work.** Multi-state and post-2019 extensions of the ACS analysis; class-invariance
conditions for the denoised arm (when does a probe family admit *any* sound staleness reading?); a
δ(N) scaling study at production scale; a proper-score arm with its own calibrated floor, which
§6.4 shows is the repair direction for the metric blind spot but which would need its own
pre-registration; and porting the certificate protocol to streaming monitors, where the
negative-recency fingerprint is directly actionable.

---

## 9. Reproducibility

Everything is in one repository, linked in anonymized form for review. The instrument is a single
sklearn-only script; every stochastic step is seeded, and every output artifact embeds its commit
hash, argv, library versions and UTC timestamp.

That stamping is not ceremony, and we can say so from an incident rather than from principle. One
battery run executed under the wrong interpreter, because the shell prompt announced the intended
conda environment while `PATH` still resolved to an unrelated project's virtualenv. The verdicts
passed, so nothing on screen looked wrong; the artifact's recorded library versions did not match
the environment the real-data runs use, and that mismatch is the only thing that caught it. Had the
run been trusted, a gate certified in one environment would have licensed measurements in another.
Every subsequent run invokes the interpreter by absolute path and asserts its version before doing
anything else.

**Compute.** Every run in the paper is CPU-only — the instrument and all probe classes are
scikit-learn models, and no GPU is used anywhere — executed single-node on one shared multicore
Linux server (Python 3.11.15, scikit-learn 1.9.0, NumPy 2.4.6; the environment freeze is committed).
Wall-clock is reconstructable from the committed phase logs: the main pre-registered phases ran
phase-parallel in ≈17 h on one calendar day, the model-class panel adds ≈2 h, and the optional cells
≈13 h across two further days; no single dataset cell exceeds a few CPU-hours.

**Reproducibility of the gate.** The synthetic battery is byte-reproducible: re-running it
regenerates the committed artifact SHA-identical on the same environment, and when the instrument
gained two opt-in diagnostic flags the battery was required to come back bit-identical against the
reference environment before any new run was read (§5). The raw arm's RNG stream is stable across
instrument versions — the v2 headline number reproduces bit-for-bit under v3, which is what makes
§3.2 a reinterpretation rather than a re-roll. Exact cross-version bit-reproducibility of
HistGradientBoosting across scikit-learn releases is *not* claimed; verdict-level stability across
seed sets and across HGB/RF is (10/10 and 10/10).

**Pre-registration.** The pre-registration freezes thresholds, cascade, seed protocol and aggregate
reading, with results appended as read-only sections whose ordering is enforced by the git history;
existing sections are never edited. Predictions for each queued experiment are committed in the
driver script's header before execution, and read against the outcome afterwards including where
they failed. All server runs are reproduced by one resumable driver.

**Two disclosures.** First, one deviation from the registered procedure: the family sweep specified
a human checkpoint after its implementation control, and the batch that ran it computed all four
signal families before that checkpoint was read, so the gate became a read-time filter rather than
an execution-time one. No family was excluded in the event — the control recovered in all four —
but the order differed from what was committed. Second, one decision constant is known to be wrong
in a direction we did not act on: §6.4 shows the 0.02 decision floor is larger than a real
population-scale policy rule change. We record that in the constants appendix and leave the
threshold alone, because changing a pre-registered threshold after seeing results is what this
paper argues against.

**Data access.** Per-source access and license terms are tabulated in the data appendix. TabReD
requires Kaggle authentication and per-competition rule acceptance; Electricity fetches from
OpenML; INSECTS and the river panel install via `river`; EMBER-2018 downloads from its public
archive and is parsed by a dependency-free adapter.
