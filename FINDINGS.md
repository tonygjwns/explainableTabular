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

8. **Elec2 decider — POSITIVE on REAL concept drift.** On the canonical Electricity
   concept-drift benchmark (random split, inject off so memory is the only time path,
   10 seeds): time-indexed **AUC 0.954 vs fixed 0.911**, delta +0.043, p=0.002,
   g=+7.39. The mechanism exploits real concept drift, not just synthetic.

## Conclusion
- Implementation verified (4). Mechanism is real and capable.
- **Synthetic (+87%) AND real Elec2 (+4.3 AUC) → the mechanism exploits concept
  drift.** On TabReD it doesn't help because TabReD's shift is **covariate, not
  exploitable concept** (5–7). The three points line up: works on concept drift,
  null on covariate-only — a coherent, evidence-backed story.
- **Positive (A') path is LIVE**, pending: Elec2 temporal split (future
  extrapolation), memory-vs-time-as-feature (is the memory structure necessary?),
  strong drift-aware baselines, and more concept-drift benchmarks.

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
