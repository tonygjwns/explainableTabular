# Phase 1 Findings — time-indexed prototype memory on TabReD

> Evidence chain as of 2026-06-05. For the C→B / direction decision and advisor review.
> Companion to PRE_REGISTRATION.md (criteria) and progress.md (daily log).

## TL;DR
The mechanism **works when concept drift exists** (synthetic: +87%), but **TabReD
has covariate drift, not exploitable concept drift**, so time-indexing the memory
gives **no prediction gain** there — even when the memory is forced to engage.
This is a clean, implementation-verified negative on TabReD, plus a validated
mechanism looking for the right data.

## Evidence chain
1. **Phase 0 — reproduction OK.** TabM reproduced on 8 TabReD datasets within ~1%
   of published MLP/TabReD numbers (sberbank 0.2572, etc.). Pipeline trustworthy.

2. **Test 1 (time-indexed vs fixed memory), isolated (inject off), 10 seeds.**
   Clean **null**: delta≈0, |g|≤0.17, none significant post-FDR (4/4). No gain, no
   significant degradation.

3. **Training diagnostics (the key instrument).** Under the default `concat`
   predictor the memory is **decorative**: `mem_gap≈0` (zeroing the memory readout
   barely changes loss), memory gradients ~100–1000× smaller than backbone/predictor,
   retrieval collapses to ~1 prototype, aggregated ≈ constant. The predictor solves
   the task from `z`; the memory is bypassed (z-shortcut, EXPERIMENT_PLAN open Q5).

4. **Synthetic positive control — implementation is CORRECT.** Pure concept drift
   (x stationary, y = x·w(t), w rotating). time-indexed RMSE **0.13** vs fixed
   **1.03 (=std(y))**, **+87%**, `mem_gap≈+0.93`, drift grads healthy. memory_only
   also 0.13. So the whole pathway (Fourier→drift→P_k(t)→retrieval→predict) exploits
   concept drift when it exists. The TabReD null is **not** a bug.

5. **Drift decomposition (G3, model-light).** TabReD has **strong, pervasive
   covariate drift** (train-vs-test feature AUC≈1.0 on 4/8, stays high after dropping
   the top-5 separating features → not a trivial time-proxy), but **weak label/time
   relationship** (|Spearman(t,y)|≤0.13).

6. **Concept-drift measure (train-early vs train-late on the future).** Recency value
   is small for most; **sberbank +36%**, homecredit +4%. BUT this measure is
   **confounded by the strong covariate shift** (early-trained model extrapolates
   further in x).

7. **Engaged time-vs-fixed on sberbank (the strongest-signal set), 10 seeds.**
   Forcing the memory into the output path (residual / memory_only) **still does not
   help**: residual g=−0.30 (slightly worse), memory_only no signal + unstable
   (codebook collapse to ~0.39 on several seeds). → sberbank's +36% recency was
   covariate extrapolation, **not** concept drift the mechanism can exploit.

8. **Elec2 — time helps, but the MEMORY does not earn its keep (positive path closed).**
   Canonical Electricity concept-drift benchmark, 10 seeds:
   - random split, inject off: time-indexed AUC 0.954 vs fixed(no time) 0.911 (+0.043).
     Looked positive — but this only says *time information* helps, not the memory.
   - random split, **inject on (time as an input FEATURE)**: fixed jumps to **0.9615**
     (a plain time feature > the memory's 0.954), and adding the drift memory on top
     gives **delta +0.0017, p=0.11, g=0.76 (n.s.)**. => the memory STRUCTURE does not
     beat trivially feeding t as a feature (empirically reproduces Cai's input-side
     finding; the time-indexed-retrieval "gap" is empty because it loses to the simpler
     option).
   - **temporal split (realistic future extrapolation), inject off**: time-indexed
     **0.875 vs fixed 0.894, delta −0.019, g=−0.67** — the memory HURTS and is unstable
     (a seed collapsed to 0.77).

## Conclusion
- Implementation verified (4): the mechanism exploits concept drift in a controlled
  synthetic setting (+87%, in-distribution t, memory as the only path).
- **On real data the positive (performance) path is CLOSED both ways**: TabReD has no
  exploitable concept drift (5–7); Elec2 has it, but a plain time FEATURE captures it
  better than the time-indexed memory, the memory adds ~0 over the feature, and it
  DEGRADES under temporal extrapolation. The memory-retrieval structure does not
  justify its complexity over simpler time-conditioning.
- Honest status: a thorough, implementation-verified **negative** for the proposed
  architecture as a *predictor*. (Earlier "+4.3 => A' live" was premature — it lacked
  the time-feature baseline, which dominates.)

## Remaining honest options (narrowed)
- **B (analysis/negative):** when/why time-indexed memory helps (controlled) vs fails
  (covariate-only; feature-subsumed; extrapolation-fragile) + diagnostic toolkit +
  covariate-vs-concept. Needs breadth for a top venue; solid for TMLR/workshop.
- **C (reframe to interpretability):** drop the performance claim; evaluate the
  memory purely as a faithful, inspectable drift-explanation tool (memory_only, no
  z-shortcut), with a real interpretability eval/use-case. Risky; needs new design +
  faithfulness evaluation (current memory is z-shortcut-bypassed).
- A genuinely new mechanism with a reason to beat time-feature conditioning would be
  needed for a positive top-tier method paper. Not currently in hand.

## Open directions (top-tier-shaped)
- **A' (positive):** run the validated mechanism on real concept-drift tabular
  benchmarks (Elec2, Insects, Airlines, sensor/fraud streams). If it beats fixed
  there → method paper. **Decider experiment: Elec2 time vs fixed.**
- **B (analysis):** covariate-vs-concept decomposition + validated mechanism as a
  probe + why temporal tabular methods underdeliver. Needs breadth (methods×benchmarks).
- **A'+B combined** is strongest: method wins where concept drift is real, diagnostic
  explains why it doesn't on TabReD. Hinges on A' succeeding on ≥1 real benchmark.

## Rescue results (2026-06-05) — Q1 PASS, F3 decisive
- **Q1 (functional faithfulness, gate)**: properly-built mechanism (memory_only +
  trend basis + load-balance) recovers true drift on synthetic: recovery 0.991
  (mean, 10/10 seeds) vs ceiling(MLP+t) 0.990, floor(shuffle-t) 0.894 → **PASS**.
  The MECHANISM IS FAITHFUL & CORRECT (necessary condition for the interpretability
  claim holds in principle). Caveat: ≤90° drift makes the floor high (narrow dynamic
  range); harder geometry is a robustness follow-up.
- **F3 (concept measurability)**: measurable only on cooking_time / maps_routing —
  the two LOWEST-covariate datasets (AUC 0.80/0.64), which have ~no concept. Where
  concept might exist (elec2, homecredit, …) strong covariate (AUC≈1.0) destroys
  early/late common support → concept UNMEASURABLE (elec2 ESS=20; 4 datasets overlap=0).
  → **The very covariate dominance that defines TabReD-style drift makes concept
  drift unmeasurable/unexploitable there.**
- **Verdict (pre-registered)**: Q1 PASS (main claim not dead) BUT Q2's premise — a
  clean real benchmark with BOTH measurable AND substantial concept drift — is
  **unsatisfiable on available data** (measurable⇒no-concept; concept-suspected⇒
  unmeasurable). The failure on real data is about the DATA, not the method (Q1 proves
  the method works when concept exists).

### Upgraded §6(다) thesis (now strongly evidenced) — claim PRECISED
"In realistic tabular temporal data, drift is overwhelmingly covariate; strong
covariate shift destroys early/late common support, so concept drift is
**unmeasurable by the standard conditional lens** ((a)(b) — strong & novel); AND on
available benchmarks a **structured time mechanism shows no exploitable concept
beyond a simple time-feature**. (Do NOT claim the universal 'time methods don't help'
— a time FEATURE does help via covariate adaptation on elec2 (test 7); what fails is
the memory/retrieval STRUCTURE beating that feature. 'unmeasurable' (epistemic) +
'structure ≤ feature' is sharper and defensible.) Shown with a mechanism that
provably & faithfully captures concept on synthetic (Q1 PASS) yet cannot help on real.

OPEN (spine, do FIRST): the elec2 F3 row is internally contradictory — overlap
0.438 but ESS 20. 44% in-band = substantial common support; ESS=20 is almost
certainly a GLOBAL-IW heavy-tail artifact (a few out-of-band P≈1 points blow up
Σw²; in-band odds are [0.11, 9], fine). Re-measure concept WITHIN the overlap band
(conditional P(y|x) early-vs-late, no global reweight). If elec2 is within-overlap
measurable AND has concept → the middle case of the dichotomy opens → we hold the
measurable-concept real dataset Q2 needs (run small Q2 factorial there); if a
'negative ON MEASURED concept' results, that is far stronger than 'could not measure'.
Q1 headline use also needs one large-rotation (≥180°, basis-matched) robustness to
widen the narrow recovery dynamic range (floor 0.894→ceiling 0.990).

## Within-overlap concept (spine fix, 2026-06-05) — elec2 has REAL concept
The F3 elec2 ESS=20 was a global-IW heavy-tail artifact. Measuring concept WITHIN
the overlap band (covariate matched, no global reweight):
- **elec2: concept_gap = +0.132 AUC** (transfer gap on a FIXED late-overlap test:
  early-trained 0.716 vs late-trained 0.848; n_overlap 9173/4721). **CONFIRMED with
  out-of-fold p (region selection) and p-strata stability [+0.12,+0.17] across all
  tertiles** → real exploitable concept, NOT difficulty nor residual covariate.
  (in-sample p gave +0.166; OOF +0.132 — still large.)
- cooking/maps: ~0 (measurable, no concept). delivery: −0.03 (small/noisy).
- 5 high-covariate (AUC≈1.0) datasets: n_overlap=0 → truly unmeasurable.
→ Dichotomy REFINED (not broken): high-covariate ⇒ unmeasurable; low-covariate
  cooking/maps ⇒ measurable-but-no-concept; **elec2 (mid covariate) ⇒ common support
  AND real concept** = the measurable-concept benchmark Q2 needs.

### Status: BOTH gates open Q2
Q1 PASS (faithful mechanism) + F3-within-overlap (elec2 measurable + large concept)
→ Q2 is properly motivated. Earlier elec2 "memory ≤ feature / hurts on temporal" used
the BROKEN mechanism (Fourier, learned V_k, collapse) → must re-test with the redesign
(trend basis + load-balance + instance V_k). Q2 = does the STRUCTURE (time-TabR) beat
a basis-matched time-FEATURE on elec2's MEASURED concept? Win → positive method result;
no win → 'negative on MEASURED concept' (far stronger than 'could not measure').
(Do not repeat the +4.3 over-claim: elec2 having concept ≠ our structure winning.)
§6(다) claim must NOT say 'concept unmeasurable everywhere' — elec2 is the counterexample.

## Q2b ANSWERED (2026-06-09) — structure ≤ feature on MEASURED concept (robust negative)
The proper structure test (instance `V_k` = TabR, same-encoder 3-arm, time-conditioning
on/off) on elec2's measured concept. `run_elec2_q2.py` → `results/phase1/elec2_q2/diagnostics.jsonl`.
1. **Bug-vs-drift settled (curves).** train_loss decreases for all arms (no gradient bug).
   On **temporal** split val peaks at epoch 2–4 then declines; on **random** it peaks at
   epoch ~55 with a smooth rise. = textbook concept-drift signature (fit train regime ⇒
   future val degrades), NOT a code/design bug. Confirmed at sub-epoch (step-eval) resolution.
2. **Structure does not beat feature (10 seeds, per-arm oracle lr).**
   - no regularization: mlp_t **0.9054** > time_tabr 0.9003 > tabr 0.8969.
   - **+regularization (dropout 0.1, wd 1e-4) + min_epochs 20** (forces the zero-init
     (t_i→t_q) correction to actually train — best_epoch reaches 11/14/36): mlp_t **0.9027**
     > tabr 0.8955 > **time_tabr 0.8848**. time_tabr−mlp_t = **−0.018** (worse than unreg.),
     time_tabr−tabr = −0.011 (the time hook HURTS retrieval), mlp_t−tabr = +0.007 (time
     itself helps). time_tabr **std 0.05–0.075** (unstable across seeds/lr) → the value-side
     drift-correction term is **ill-conditioned**, adding variance not signal.
3. **Protocol-artifact threat removed.** The advisor's worry — "the negative is just the
   mechanism never engaging (stops at epoch ~1)" — is refuted: with min_epochs+regularization
   the mechanism trained (best_ep up to 36) and the negative **held and strengthened**.
4. val→test Spearman ≈0.07 (val useless as selector under drift). On RANDOM split time_tabr≈
   mlp_t (both high) is the pre-registered **autocorrelation-leakage red flag**, not concept
   exploitation; the decision rests on TEMPORAL, where structure clearly loses.
→ **Verdict: time INFORMATION helps, but a plain time-FEATURE MLP carries it best; the
  time-TabR retrieval STRUCTURE does not beat the feature (−0.018) and is unstable.** This
  is the §6(다) "negative on MEASURED concept" — far stronger than "could not measure",
  and consistent with elec2 being trivially exploitable (autocorrelated stream).
**Caveat / next**: single dataset. Pre-registration needs ≥2 → **Insects (designed drift,
  multiclass)** to test whether the negative generalizes to a non-trivially-exploitable
  concept, or whether structure can win there. (Loader infra being built.)

### Q2b 2nd dataset — INSECTS confirms the negative (≥2 datasets, 2026-06-09)
INSECTS `incremental_balanced` (multiclass, accuracy), temporal, regularized+min_epochs 20,
10 seeds. Unlike elec2, INSECTS is NOT trivially saturated (argmax_val ~15–20 epochs) and
time matters a LOT — so it's a genuine non-trivial concept-drift test.
- oracle mean test: mlp_t **0.6704** > time_tabr 0.6594 > tabr 0.6320.
- **time_tabr − mlp_t = −0.011** (structure again does NOT beat the time-feature; consistent
  with elec2 −0.018). mlp_t − tabr = **+0.038** (time hugely helps — 5× elec2's +0.007).
  time_tabr − tabr = +0.028 (here the time hook HELPS retrieval — opposite of elec2 — but
  still below the plain feature). time_tabr std .014–.033 (still less stable than mlp_t).
- val→test Spearman **+0.87** (vs elec2's 0.07) → on INSECTS val predicts test; the elec2
  val/test breakdown was elec2-specific (autocorrelation/regime), not a general flaw.
→ **Across TWO opposite-character benchmarks (trivially-exploitable elec2 / non-trivial
  designed-drift INSECTS) the time-TabR STRUCTURE never beats a plain time-FEATURE
  (−0.018, −0.011).** Pre-registered "structure win on ≥2 datasets" is UNMET → the §6(다)
  negative/analysis result is locked. (Nuance for the writeup: on INSECTS time itself and
  the time hook both behave sensibly — the mechanism isn't broken; it's just dominated by
  the simpler time-feature. Clean "structure ≤ feature", not "mechanism fails".)
**DECISION (pre-registered)**: ≥2-dataset structure win = NO → direction = §6(다) analysis/
  negative paper (measurable concept exists, structure doesn't beat feature, covariate-vs-
  concept toolkit, Q1-faithful mechanism). A'(method) path closed for time-TabR structure.

### Precision pass (2026-06-10, critique-driven) — restated honestly before alignment
PAIRED per-seed stats (arms share seed/split/init → diff-SE is the honest scale, far below
each arm's marginal std):
- **elec2 = consistent-but-uninformative**: time_tabr−mlp_t paired −0.005, SE .0035, 95% CI
  [−0.013,+0.003] (1.45 SE, sign 2/8) — NOT distinguishable from 0. val→test Spearman 0.07
  ⇒ elec2 is a noisy model-comparison substrate (regime/autocorrelation), even though the
  DATA has concept (+0.132). So elec2 does NOT carry the negative.
- **Insects carries it**: time_tabr−mlp_t paired −0.011, SE .0052, 95% CI [−0.023,+0.001]
  (2.12 SE, sign 3/7) — borderline-significant, val→test 0.87 (clean). The negative is
  Insects-borne (n=1 clean), elec2 is consistent-but-uninformative support.
- **Decompose structure vs substrate** (pre-empt "but time_tabr>tabr ⇒ structure works"):
  mlp_t−tabr=+0.038 (CI [+.029,+.048], 9.6 SE — retrieval substrate ≪ parametric) and
  time_tabr−tabr=+0.028 (CI [+.013,+.043], 4.1 SE — time-indexing DOES add value inside
  retrieval). The time-structure works; it loses because the retrieval SUBSTRATE can't reach
  parametric, so the combined structure can't beat the feature. Consistent with redundancy
  (state as empirical, NOT proof — redundancy is an in-dist capacity argument).
- "oracle test" = best-on-test lr per arm ⇒ strong-form negative ("even given oracle lr,
  structure doesn't beat feature"), but NO deployment claim on elec2 (val→test 0.07).

### Framing decision (venue-dependent — for advisor alignment)
LEAD with **Claim A** (robust, novel: covariate dominance ⇒ concept unmeasurable by the
standard conditional lens, across ~10 datasets, with the F3/within-overlap measurement
frame + a reusable diagnostic toolkit). Present **Claim B** (structure ≤ feature) as
SUPPORTING evidence with the decomposition — do NOT let thin B weaken A.
Honest venue read: current package = workshop-ready negative NOW; **NeurIPS D&B track** is
the best fit with modest scope add (it values benchmark/measurement/analysis over novelty);
top main-track analysis needs scope (15–20 tabular-temporal datasets), **multiple time-aware
methods** through the same lens (turns "our method failed" → "the field's methods
underdeliver for this reason"), toolkit promoted to a validated 1st-class contribution, and
redundancy developed carefully as a hypothesis. (Q1 large-rotation robustness still open for
the "mechanism is faithful" headline argument.)
**Cheap robustification (recommended, not optional)**: add INSECTS abrupt/gradual variants
(same loader, near-free) to take B from n=1-clean → n=3-clean across drift types; deprioritize
elec2-abrupt (inherits elec2's val→test 0.07 noise).

## V2 INVALIDATION of the Q2b negative (2026-06-12, external audit) — re-test pending
The "Q2b ANSWERED" / "INSECTS confirms" sections above are RETRACTED as evidence for
"structure ≤ feature" (the NUMBERS stand; the INTERPRETATION does not):
1. **The tested hook could not express the hypothesis.** The linear value hook
   collapses under aggregation — `Σ_k w_k·Linear(Δτ_k) = Linear(Σ_k w_k·Δτ_k)`,
   independent of y_i — i.e. it is ONE aggregated Δt feature, not a per-neighbor
   stale-label correction. The experiment compared "retrieval-weighted Δt feature"
   vs "direct t feature", not structure vs feature.
2. **Sub-TabR substrate.** No similarity temperature/scaling, no key projection,
   in-batch-255 (train) vs fixed-4096 (eval) candidate mismatch, retrieval knobs
   never tuned (only lr). The decomposition's "substrate deficit" (−0.038) may be
   an artifact of this, not intrinsic.
3. **Confounded arms.** Only mlp_t received the direct time feature; the missing
   arm (retrieval + direct t) means "structure vs feature" was conflated with
   "where time enters". Plus: 10 seeds vs the pre-registered 25; INSECTS CI upper
   bound +0.001.
Counter-signal preserved: time_tabr−tabr=+0.028 (INSECTS) was positive even with
the crippled hook — the V2 re-test can flip the headline.
**Status**: V2 infra implemented (non-degenerate mlp/gate hooks + expressivity
smoke guard, scaled similarity + key projection, aligned train/eval contexts,
tabr_t / time_tabr_t arms, paired Hedges' g, anchors script). Decision rules:
PREREG_V2.md (locked before server runs). Plan: PLAN_V2.md. The pre-V2 runs remain
reproducible via `run_elec2_q2.py --legacy`.

## Assets built (reusable for any direction)
- Verified pipeline (loader/TabM/trainer), Phase-1 model with `predictor_mode`
  {concat, memory_only, residual} and time/inject toggles.
- Training diagnostics (`src/training/diagnostics.py`): mem_gap, retrieval
  concentration, drift liveness, per-module grad norms.
- Analysis: `drift_measure` (covariate AUC + trivial/pervasive split, concept gap,
  label drift), `retrieval_trajectory` (Test 3), `extrapolation` (Test 2).
- Scripts: run_phase1_sanity (Test 1, --predictor-mode), run_test2/3, run_drift,
  run_conceptdrift, run_synth_control (positive control).
