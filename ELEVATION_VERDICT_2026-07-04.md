# ELEVATION VERDICT — can the surviving assets reach top-tier? (2026-07-04)

> Follow-up to `AUDIT_FINAL_2026-07-04.md`. Question: largest defensible contribution, graded
> against NeurIPS/ICML MAIN-TRACK, every elevation with its own kill-test. Method: 5 deep-dive
> agents (2 theory w/ web adjudication, 1 EXECUTABLE estimator builder, benchmark scoping,
> hostile-review simulation) + 3 red-team killers who re-executed the pivotal numbers.
> All executed artifacts preserved in `audit_artifacts_2026-07-04/` (estimator harness:
> `exp-estimator/harness.py`, battery: `BATTERY_SUMMARY.json`).

## BOTTOM LINE

**Main track is out of reach for this asset base, and the reasons are structural, not
effort-shaped.** The honest ceiling is **NeurIPS D&B (strong, potentially well-cited)** with
**TMLR as the fast honest venue now**. The one genuinely NEW asset produced by this exercise —
and the anchoring result of the D&B package — is the **repaired estimator (denoised staleness +
noise gate)**, which was built, battery-tested, and survived hostile bit-for-bit re-execution
today. Three planned elevations were killed before you could spend months on them.

## Q1 — Maximal defensible thesis

"**Drift-type attribution — the concept-vs-covariate verdict every staleness/loss-based drift
monitor emits — is relative to the probe's hypothesis class and the label-noise process, not a
property of the data.** We demonstrate this with an executed sign-flipping panel (identical data:
kNN probe +0.098 'concept' vs HGB −0.001 on pure covariate shift; label-noise decay alone mints
+0.021 'concept' under a provably fixed rule), repair the estimator (denoised staleness + noise
gate, adversarial battery, mapped validity envelope with abstention), and deliver a
certificate-based audit of industrial tabular ML: no exploitable mean-rule drift above
per-dataset detectable floors, with blindness certificates where measurement fails."

Grading: rigor high (byte-reproducible, red-team-survived); novelty partial (the PRINCIPLE is
published — see Q2; the executed panel, the estimator, and the certificates are new); stakes
moderate (attacks trust in deployed drift monitors, but no demonstrated real-world casualty yet);
generality scoped (loss-based attribution, flexible classes). **Binding constraint: the theorem
territory is occupied, the only concept positive is designed-drift, and the project's
9-dissolution generating process can only be answered socially (external adversary), not
experimentally.** Ceiling: D&B.

## Q2 — F3 as THE contribution: KILLED as theory, survives as the empirical spine

The exact thesis center is already published: **Hinder, Vaquet, Brinkrolf & Hammer (ICPRAM
2023)** constructively prove BOTH directions under finite VC — virtual drift (P(x)-only) that
moves the optimal model's loss, and real drift (P(y|x)) invisible to loss — and their 2024 survey
states "the used model class is crucial in terms of which drift can be detected." Add Ben-David
2010 (HΔH = class-relative distinguishability), Shimodaira 2000 (mechanism), Johansson 2019
(representation-relative overlap), Detectron ICLR 2023 (harmful-shift defined model-relatively).
The red-team weakened even the residual deltas: under the field-standard Webb/Gama definition
(real drift = ANY P(y|x) change) the early-noisy control is a TRUE positive — the "false fire" is
relative to mean-rule semantics, so F1 is a definitional-scoping footnote + classical GLS
inefficiency; and **Loog, Viering & Mey (NeurIPS 2019)** prove ERM risk non-monotonicity with
added iid same-distribution data — E[staleness]>0 is possible with ZERO drift and ZERO
heteroscedasticity, so the soundness-proposition territory outside a vacuous assumption fence is
provably empty. What survives, unclaimed in the literature: (i) the retraining-regret contrast as
a drift-ATTRIBUTION instrument with verdict semantics (Klinkenberg & Joachims 2000 used the
mechanics for window adaptation only); (ii) the first EXECUTED sign-flipping panel with a measured
false-positive band. Lemma-sized + section-sized. Not a theory paper.

## Q3 — Identifiability theorem with teeth: KILLED (drop the line)

The instrument's real setting is retrospective audit (future windows HAVE labels) → the blind
region is the symmetric difference of window supports = **positivity indexed by time**; DISDE's
Proposition 1 (verified by direct fetch of arXiv 2303.02011) is already an identification result
for the shared-support decomposition; "necessary" fails under parametric extrapolation (no
class-free necessity exists — that's F3 again); conditions (ii) rows-per-window and (iii)
noise-separability are finite-sample/instrument conditions, not identifiability conditions.
Retraining regret R is trivially identified (both models trainable, future labeled) — estimand
hygiene, zero theorem content. Keep a 3-proposition FORMAL SPINE as scaffolding (retraining-regret
estimand fixes the circularity; abstention = certified estimability failure; class-relativity as
scope conditions with executed necessity certificates), fully credited to prior work.

## Q4 — The valid estimator: BUILT, BATTERY-PASSED, RED-TEAM-SURVIVED (the new asset)

**Denoised staleness**: replace old labels with 2-fold cross-fitted within-old-window HGB
pseudo-labels before forming the recent∪old arm; **noise gate**: per-window held-out noise proxy,
old/recent ratio, threshold 1.5; **conjunctive rule** with abstention. Executed battery (all in
`audit_artifacts_2026-07-04/exp-estimator/`):

| Cell (truth) | raw staleness | denoised | gate | outcome |
|---|---|---|---|---|
| B1 early-noisy (F1 killer; fixed mean) | **+0.0214 fires** | +0.0037 null | 3.54 fires | NOISE-DRIFT-CONFOUNDED ✓ |
| B2 rotating rule | +0.546 | **+0.541 fires** | 0.91 quiet | CONCEPT ✓ (power retained) |
| B3 concept/covariate/stable/covariate_mc | — | reproduces HGB pattern | quiet | ✓ ×4 |
| B4a concept + noise-decay | +0.32 | **+0.316 fires** | 3.67 fires | CONCEPT ✓ (gate-veto alone would misfile — denoised arm is load-bearing) |
| B4b x-dependent noise drift (fixed mean) | **+0.0254 fires (NEW raw FP channel)** | +0.0048 null | 3.76 | ✓ defused |
| Red-team: rule-feature-correlated noise (fixed mean) | **+0.0262 fires (2nd NEW raw FP channel)** | +0.0055 null | 3.41 | ✓ defused |
| Red-team: small concept (0.8t) + noise decay | +0.115 | +0.062 fires | 3.69 | CONCEPT ✓ (no false negative) |
| B4c old window = 600 rows | — | no FP, no FN | — | ✓ |
| B5 RandomForest | — | B1 null / B2 fires | — | ✓ |
| B5 kNN on covariate | +0.098 | **+0.111 still false-fires** | — | expected: denoising does NOT fix the F3 misspecification channel — guarantee must be class-scoped |

**Mapped hard boundary (self-discovered)**: the denoiser is biased; at old/recent noise-ratio
≈5.7 the denoised arm false-fires (+0.0259) under a pure null → envelope: trust CONCEPT calls up
to gate-ratio ≈4.7, ABSTAIN (NOISE-AMBIGUOUS) beyond. Red-team verdict: **SURVIVES**, with one
correction to adopt: report attenuation as den(concept+noise)/den(concept-clean) = **42%** in
B4a, not "~12%". Honest guarantee: an empirically calibrated envelope on Gaussian noise families
for flexible approximately-well-specified classes (HGB/RF verified; kNN/linear excluded by
executed counterexample) — NOT a theorem, and per Loog 2019 no theorem is available for the
deployed class. Practitioner delta (genuinely new): distinguishes "old labels WRONG → poison,
retrain recent-only" from "old labels NOISY → reusable via self-training pseudo-labels" from
"abstain", which raw staleness could not. Does NOT repair F2 (group-aware D is the separate,
already-validated fix) and does not resurrect sberbank (prediction: it reads
NOISE-DRIFT-CONFOUNDED on the server — that run is the first kill-test below).

## Q5 — Landmark-scale map: scale alone does NOT clear main track

Honest achievable panel: **~18–22 independent real-time-axis datasets (~25–30 tasks)** — TabReD 8
(+ extend into val/test segments), elec2, EMBER-2018 monthly (prep script exists), Avazu, NOAA,
Airlines, gas-sensor, LendingClub matured vintages only (censored recent vintages are literally
the F1 failure mode — a showcase for the noise gate, a landmine without it), folktables 1-year
files 2014–2023 (NOT 5-year; 2020 gap; the fixed $50k threshold is a semi-known drift anchor),
optionally MIMIC via credentialing — plus insects (8 cells, ONE source), river panel (23
ground-truth cells, already in repo), 5-control synth suite → ~55–60-row grand table. Protocol:
pip-installable auditor (X,y,t)→verdict+delta+certificates; HGB+RF panel with linear/kNN
canaries; noise gate; group-aware D; learnability-gated injection; K-sweep column; frozen
thresholds. **The main-track gap is conceptual, not breadth**; the map is a strong D&B at 12–15
well-certified datasets already. Kill-test for emptiness: if ~90% of cells land blind, the map is
a finding only where blindness certificates are EARNED (learnable injections); otherwise it reads
"we cannot measure most things."

## Q6 — Hostile review (verbatim cores)

REJECT: R1 (theory): "…a competent empirical illustration of known results (Shimodaira 2000;
Johansson 2019; D'Amour 2021); an illustration of folklore, however well executed, is not a
theory contribution at this venue." R2 (benchmark): "WhyShift already published the X-vs-Y|X map
genre on tabular… the sole concept positive is a stream whose drift was designed in by its
curators; a well-engineered non-result — 'a better thermometer confirming the patient has no
fever.'" R3 (empiricist, the kill shot): "…the ninth retracted finding in the project's ledger;
every repair was designed after, and validated on controls written by, the same authors; nothing
distinguishes 'now sound' from 'patched until the in-house controls pass.'" — R3 has NO
experimental answer; only an adversary the authors did not design (external replication or a
pre-registered held-out battery) answers it, and that is a months-scale social process. Note: the
history is not removable — the repo's committed record reconstructs it for any reviewer who looks.

ACCEPT hinges: (i) sign-flipping panel — **supported today, synthetic only**; needs the real-data
class panel + a 1-day "first"-claim literature check; (ii) repaired estimator surviving an
adversarial battery its predecessor fails — **EXECUTED TODAY** (this document), "provably" must
be dropped; (iii) 25-dataset certificate map — out of near-term reach (and doesn't answer R2
anyway); (iv) honest certified negative — one server evening with the repaired instrument;
pre-commit that any NEW industrial positive is dissolution-candidate #10, not a headline.

## Q7 — The minimal top-tier package (D&B-landmark version)

Title-shaped thesis: "Your drift monitor's verdict is a property of your probe: class- and
noise-relativity of drift-type attribution, a repaired estimator with a mapped validity envelope,
and a certificate-based audit of industrial tabular ML." Five results, each with its kill-test:

1. **Sberbank repaired-estimator rerun (server, one evening) — RUN THIS FIRST.** Kill-test: if it
   reads CONCEPT with gate quiet and in-envelope, the cell revives and must then survive K-sweep
   + fresh seeds + class panel; predicted outcome NOISE-DRIFT-CONFOUNDED, which validates the
   repair narrative on the exact cell that dissolved.
2. Repaired-instrument full rerun, 10 datasets, after freezing PREREG_DEPLOYMENT_V2.md; fresh-seed
   confirmatory pass. Kill-test: any exploratory→confirmatory verdict move is reported unstable.
3. **Anchor expansion (the collapse risk): EMBER monthly + river 23 cells + insects variants +
   folktables bridge.** Kill-test: literature-established drift anchors MUST fire under the
   repaired instrument; if EMBER/river read null-or-blind with learnable injections, the
   instrument has no real-world positive beyond insects and the package collapses to workshop.
   **This is the single result that, if it fails, collapses the whole thing.**
4. Real-data model-class panel (HGB/RF/linear/kNN × datasets). Kill-test: if all classes agree
   everywhere, the class-relativity attack has no field casualties → demote to instrument
   documentation; if verdicts FLIP on ≥2 real datasets, this becomes the headline and the one
   main-track lottery ticket.
5. Protocol/repro packaging: group-aware D, learnability-gated injection, K-sweep column, pinned
   env, tracked ledger, pip auditor. Kill-test: a third party regenerates the table from the repo.

Rough cost: 1–2 server evenings (1,2), ~1–2 weeks data engineering (3), 1 server day (4), 1 week
(5).

## Q8 — Honest ceiling

**It caps at D&B.** Main track requires one of: a theorem (territory occupied — Hinder 2023,
Shimodaira 2000, Loog 2019 close every proposition), an industrial concept positive (the only
candidate dissolved, and reviving it is a lottery you should not headline), or model-class
invariance (executed counterexamples show the opposite). A credible, well-cited D&B/TMLR
measurement paper is the correct terminal target, and the 9-dissolution ledger becomes an asset
(the method IS the dissolution discipline) at exactly those venues.

## FINAL RECOMMENDATION

- **Target**: TMLR now (after ~1 week of local repairs; honest fit, deadline-pressure-free) AND/OR
  NeurIPS D&B next cycle (conditional on results 1–4 landing). Do not spend a cycle on main track.
- **Anchoring new result**: the repaired estimator (denoised staleness + noise gate + envelope +
  abstention) — already executed and red-team-survived — deployed at scale with certificates; the
  real-data class panel is the potential headline-maker if verdicts flip in the wild.
- **First kill-test before any further investment**: the sberbank rerun under the repaired
  estimator (one server evening). It either validates the repair narrative on the cell that
  dissolved, or — if sberbank survives in-envelope — reopens the one claim worth more than
  everything above, under the full pre-registered battery.
