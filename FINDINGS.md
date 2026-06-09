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

## Assets built (reusable for any direction)
- Verified pipeline (loader/TabM/trainer), Phase-1 model with `predictor_mode`
  {concat, memory_only, residual} and time/inject toggles.
- Training diagnostics (`src/training/diagnostics.py`): mem_gap, retrieval
  concentration, drift liveness, per-module grad norms.
- Analysis: `drift_measure` (covariate AUC + trivial/pervasive split, concept gap,
  label drift), `retrieval_trajectory` (Test 3), `extrapolation` (Test 2).
- Scripts: run_phase1_sanity (Test 1, --predictor-mode), run_test2/3, run_drift,
  run_conceptdrift, run_synth_control (positive control).
