# FINAL COMPREHENSIVE AUDIT — deployment-decay v2 identifiability map (2026-07-04)

> Multi-agent audit (11 auditors + 16 adversarial verifications + 4 executed experiment
> batteries). Unlike all prior reviews, **code was executed**: the synth controls were
> reproduced byte-identically, and new ground-truth controls were run through the unmodified
> `assess()` in `scripts/run_deployment_decay.py`. Experiment artifacts:
> `C:\Users\joon\AppData\Local\Temp\claude\C--Users-joon-Desktop-ExplainableTab\e9542da3-1a26-4cb3-9fd3-ce63bc60e8d9\scratchpad\`
> (exp-reg, exp-dgate, exp-modelclass, exp-inject, exp-synth-repro + verify_* scripts).

## VERDICT (one paragraph)

The map as currently framed **does not survive**: the sberbank DEPLOYMENT-CONCEPT cell is the
9th dissolution (three independent kill mechanisms, one executed today), the D≥0.96 gate is
invalid as a geometry/identifiability claim (executed: saturates to 1.000 under row/entity
memorization with ZERO covariate shift), and the core discriminator "staleness>0 ⇒ P(y|x)
changed" is a property of tree ensembles, not of the instrument (executed: kNN false-fires
CONCEPT at +0.098 and linear at +0.026 under FIXED rules). What survives, and is genuinely
good: the insects positive (robust under every rule reading and all 4 model classes), the
injection-recovered nulls on cooking/delivery (the strongest cells in the map), the honest
delta bounds and flags, byte-reproducible synth validation, a leak-free data path, and an
unusually honest dissolution ledger. The defensible paper is a **measurement/instrument paper
with no industrial concept positive** — workshop-ready now, NeurIPS D&B only after the
repairs listed at the end.

---

## FATAL findings

### F1. sberbank DEPLOYMENT-CONCEPT = the 9th dissolution (EXECUTED kill)
- **Mechanism**: a pure label-noise-decay null — fixed conditional-mean rule
  `y = 3·X0 + ε(t)`, noise std shrinking 1.5→0.3 with t, zero covariate drift, zero mean-rule
  change — run through the unmodified `assess()` produces
  **verdict DEPLOYMENT-CONCEPT, staleness +0.0214 CI [+0.0196, +0.0231]**
  (`exp-reg/reg_early_noisy_result.json`). Sberbank's headline is +0.0239 CI [+0.0178, +0.0300].
  Early-noisy labels are the mundane profile of sberbank (Russian housing spanning the
  2011-12 post-crisis volatility into calmer 2014-15) and of any maturing data pipeline.
  Old labels being *noisier* (not *wrong*) is enough to mint the verdict at exactly the
  observed magnitude. Note the instrument is "right" that P(y|x) changed in variance — but the
  paper's semantics (rule changed / old labels contradict the current rule / exploitable by
  time-indexed modeling) is violated: the mean rule is fixed and old labels are unbiased.
- **Converging evidence** (each independently verified):
  - Knife-edge rule: CI-lower 0.0178 < floor 0.02; code tests `ci[0]>0 AND mean>floor`
    (`run_deployment_decay.py:483-484`) while its own comment says `lower_CI > floor` — under
    the comment's reading sberbank routes to UNIDENTIFIABLE (D=1.0 branch precedes SUBFLOOR).
    Git archaeology: the executed rule IS the pre-committed v1 rule (e680960, 2026-06-28,
    before any real run; the intermediate 0701 run had sberbank *negative* −0.0053, so no
    tuning) — no p-hacking, but the comment is wrong and the cell flips under the stricter
    plausible reading.
  - The verdict fired where the gate declared blindness (D_strip=1.0, `d-gate-invalid`), with
    the map's highest D_shuffle (0.887) — unflagged because both the d-gate-suspect flag and
    the injection control run ONLY in the UNIDENTIFIABLE branch (`:508-517`). The map's only
    industrial positive is its least-controlled cell.
  - Model-class false-positive band brackets it: fixed-rule controls read +0.026 (linear,
    prior shift) and +0.098 (kNN, covariate shift) — see F3.
  - Sign-flip history: −0.0053 CI[−.0085,−.0022] (07-01) → +0.0239 (07-03). The flip coincides
    with the documented tie-split leak fix (predicted direction, so benign per se), but it is
    the fragility profile of the 8 prior dissolutions.
- **Test (server, cheap)**: per-window residual-variance profile of a fixed model on sberbank
  (early-noisy audit); injection + d-gate-suspect controls applied to the CONCEPT branch;
  K∈{5,8,12,20} sweep; rule-sensitivity row (both stale_pos readings).
- **Conclusion changed**: industrial concept positives: 1 → 0. The map's strongest claim
  ("overlap lens was blind; concept is real in TabReD") is unsupported.

### F2. The identifiability gate D does not measure identifiability (EXECUTED)
- **Mechanism**: `_disjointness` (`:173-196`) is a held-out AUC over a ROW-level
  `train_test_split`. Executed on the repo's own functions: exact duplicates (m=5, same
  timestamp) inflate D from 0.4999 → **0.9939**; entity cohorts (iid latent entities observed
  in one time band, ZERO covariate shift) give **D = 1.0000**; near-duplicates with obs-noise
  0.02–0.1 still cross the 0.96 gate. A group-aware split (GroupShuffleSplit) deflates all of
  these to ~0.50 while leaving honest drift intact (`exp-dgate/d1_d4_results.json`,
  `d6_results.json`; independently re-executed by a verifier). The 0.96 threshold was
  calibrated on duplicate-free synth and does not transfer to entity-structured tabular data.
- **Real-data corroboration**: 8/10 datasets read EXACTLY D_strip=1.000, yet at D=1.0 —
  staleness fired (sberbank +0.024), injection recovered (cooking +0.558, delivery +0.322),
  and five datasets show old data significantly HELPING (staleness CI entirely negative).
  All three contradict "old data can't reach the future covariate region" (`:66-68`). Also:
  a single predictive drifting feature suffices to saturate D (proxy-strip keeps predictive
  features by design), so D = window separability, not support disjointness.
  `cov_auc_early_late=1.0` shares the same row-level-split channel (drift_measure.py:40,124-130)
  and is not independent corroboration.
- **Test**: group-aware/deduped split in `_disjointness` (one line, validated); server-side
  duplicate/entity-recurrence audit per dataset; report per-feature max time-AUC alongside D.
- **Conclusion changed**: the five UNIDENTIFIABLE-* cells are unsupported AS GEOMETRY CLAIMS.
  They survive only as "instrument-blind, demonstrated by injection" — see C1 for how much
  that demonstration carries.

### F3. The core discriminator is model-class-dependent (EXECUTED)
- **Mechanism**: the entire verdict matrix re-run with the module's model monkeypatched
  (`exp-modelclass/*_results.json`):

  | control (truth) | HGB | RandomForest | linear | kNN |
  |---|---|---|---|---|
  | concept (rule rotates) | CONCEPT ✓ | CONCEPT ✓ (+0.287) | CONCEPT ✓ (+0.310) | CONCEPT ✓ (+0.295) |
  | covariate (rule FIXED) | not-CONCEPT ✓ (−0.001) | not-CONCEPT ✓ (+0.004) | not-CONCEPT ✓ (−0.002) | **CONCEPT ✗ (+0.098)** |
  | covariate_mc (rule FIXED, prior shift) | not-CONCEPT ✓ (−0.003) | not-CONCEPT ✓ (+0.003) | **CONCEPT ✗ (+0.026)** | **CONCEPT ✗ (+0.048)** |
  | stable | ✓ | ✓ | ✓ | ✓ |
  | nuisance_proxy | CONCEPT ✓ | CONCEPT ✓ | CONCEPT ✓ | CONCEPT ✓ |

  This is the Shimodaira-2000 mechanism made empirical: under misspecification, ERM on a
  covariate-shifted mixture is worse on the target even with fixed P(y|x). Concept DETECTION
  is model-robust; the concept/covariate SEPARATION — the load-bearing property — holds only
  for hypothesis classes that can represent the fixed rule (tree ensembles here).
- **Test (server)**: re-run the real map under RF + LightGBM; any cell that moves gets an
  instrument-dependent flag.
- **Conclusion changed**: "staleness_harm identifies P(y|x) change" is false as stated. Every
  claim must be scoped: "relative to the deployed (tree-ensemble) hypothesis class". The
  false-positive band (+0.026..+0.098 under fixed rules) also contextualizes any borderline
  positive of magnitude ~0.02.

---

## What EXECUTED tests CLEARED (sign-offs — equally load-bearing)

- **Synth ground-truth PASS is real and byte-reproducible**: `--synth` rerun matches the
  committed-era artifact SHA256-identical; PASS invariant holds across generator seeds 0/1/2
  (the covariate_mc EARNED↔RECOVERED sub-label is seed-luck at the 0.02 knife edge, but the
  "never falsely CONCEPT" invariant is robust under HGB).
- **HGB regression path survives its missing controls** (`exp-reg/reg_controls_results.json`):
  reg-stable −0.012 ✓; reg-concept +0.546 CONCEPT ✓; reg-covariate-linear +0.0018 (positive
  but 10× below floor) ✓; **reg-covariate-nonlinear (sberbank twin) −0.005 ✓**;
  late-noisy −0.023 ✓. The only false positive is the early-noisy direction (F1). Under mild
  identifiable covariate drift (ramp 1.5, D≈0.71): NO-STRONG-CONCEPT ✓.
- **Data path is leak-free** (verified in code + live fetches): no t or transform of t enters
  any feature space; elec2 row order proven a true 944-day chronology (day-of-week steps
  +1 mod 7 at 943/943 boundaries) with the corrupt `date` column correctly excluded; insects
  stream = the designed drift trajectory (river metadata matches n=57018/33); no
  time-correlated row filtering; Fourier-tainted quarantine shares zero code with this
  pipeline; seeding is end-to-end (no unseeded RNG path found).
- **insects DEPLOYMENT-CONCEPT survives everything**: stale +0.135, CI-lower 0.123 >> floor
  under every rule reading; D=0.844 (gate never engaged); direction robust across all 4 model
  classes; covariate_mc rules out the prior-shift artifact under HGB/RF. Remaining caveat:
  magnitude entangled with documented achievable-accuracy drift (+0.192) — claim direction,
  not magnitude; and it is a designed-drift benchmark, not industrial.

---

## CAVEAT findings (must fix/disclose; map-shape survives them)

- **C1. Injection control has a demonstrated vacuity mode and no learnability check**
  (executed): junk heavy-tailed top-variance features on stationary, fully-overlapping
  geometry → injected rule unlearnable in-window (held-out AUC 0.506/0.472) → inj_stale −0.004
  → false "unident-earned"; Gaussian control on the same harness: in-window AUC 0.964,
  inj +0.195, recovered. homesite's inj = **−0.054** (a planted rotating rule made old data
  HELP) is the real-data face of this failure; elec2 is "earned" by a 0.0012 margin at 4
  hard-coded seeds (vs 10 for real reads). The earned labels on homesite/weather (and
  possibly ecom/homecredit/elec2) are not uniformly earned. Fix: in-window learnability gate
  (e.g., AUC ≥ 0.65 required), n_seeds=10, strength sweep, log which features were picked.
  Note: cooking/delivery (RECOVERED, +0.56/+0.32) are the *strongest* null cells — geometry
  demonstrably had power and staleness stayed null.
- **C2. Time-shuffle control semantics wrong in both directions** (executed): D_shuffle>0.6
  arises from head-truncation `old[:nn]/fut[:nn]` on time-sorted rows under unequal shuffled
  windows (0.653–0.940 with NO static feature; a genuinely static id gives 0.499), and the
  control MISSES near-dup memorization (D=0.985, D_shuffle=0.512). d-gate-suspect on
  ecom/homecredit does not mean "a static feature"; absence of the flag clears nothing;
  sberbank's 0.887 is silently unflagged (branch scoping). Fix: random subsample instead of
  head-slice; attach the flag on every branch.
- **C3. Pre-registration deficit**: no prose pre-reg of the v2 seven-verdict scheme exists
  anywhere (the in-file "PRE-REGISTERED" docstring and the runtime "PRE-REG:" banner still
  describe the obsolete v1 three-verdict scheme); the ">=6/8 unidentifiable" read appears
  nowhere in repo/commits; the v2 taxonomy was designed 07-02 after the 07-01 v1 map was seen
  (flaw-driven and synth-validated — mitigating, but it must be labeled post-hoc). The
  defensible statement is "code-freeze before the 0703 results", not "pre-registered".
- **C4. The committed record asserts the OPPOSITE headline**: RESULTS_LEDGER.md:64 still says
  "TabReD 8/8 stale≤0" flagged 현행/CLEAN; RESULTS.md ends at §28 "hunt over, broad negative";
  zero committed .md contains any v2 vocabulary; the headline lives in untracked, append-mode
  JSONs mixing two instrument versions with contradictory verdicts and no run metadata.
  PAPER_DRAFT_V3 still headlines the reoccurring +0.26/+0.21 that RESULTS §26 retired 30
  minutes after the draft's last commit — and the exported ReviewPackage contains the
  pre-retirement draft without §26.
- **C5. Estimand gap**: the paper's formal estimand (DISDE term-ii + positivity proposition)
  covers only the abandoned within-overlap lens; staleness_harm's concept link is asserted in
  a docstring and calibrated on synth — now shown class-dependent (F3). The "identifiability
  result" reduces to known positivity/overlap theory (D'Amour et al. JoE 2021 — already
  conceded in the draft; Ben-David et al. AISTATS 2010; Johansson et al. AISTATS 2019 for the
  representation-relativity point) plus concurrent ill-posedness work (Hinder et al. ICPRAM
  2023; Gower-Winter et al., "The Window Dilemma", arXiv 2602.06456, Feb 2026) that must be
  cited.
- **C6. Closest prior is WhyShift, not TabReD**: WhyShift (NeurIPS D&B 2023, the DISDE group)
  already published an X-vs-Y|X map over tabular datasets; TabReD itself (verified by fetch)
  has no decomposition and no recency experiments; the LAMDA ICML 2025 follow-up measures
  training-lag but not shift type. Surviving delta: temporal axis + industrial deployed
  representations + abstention/identifiability verdicts + per-dataset injection certificates.
  The staleness probe as a *shift-type identifier* appears novel (mechanical priors:
  Klinkenberg & Joachims 2000; "Data Addition Dilemma" MLHC 2024; Shimodaira 2000 supplies
  the confound).
- **C7. Statistics**: 10 seeds share ~81% of rows pairwise but are treated iid (borderline
  CIs anti-conservative — sberbank's margin above 0 is +0.018 on a seed-noise CI); no
  multiple-comparison control over 10 datasets × gates; asymmetric bootstrap at pool==N
  attenuates staleness ~26% toward null (executed: +0.0555 → +0.0408); D/D_shuffle are
  seed-0 point estimates with no CI while every other verdict-driver got one.
- **C8. Scope limits**: the map covers only the TabReD TRAIN segment (most-recent val/test
  periods never probed); K=10 was never swept (cyclic/seasonal concept and fast-within-window
  drift structurally invisible — elec2's tiny inj_stale +0.019 is consistent with
  window-averaging of the injected rotation, not blindness); proxy-strip uses marginal MI on
  20k rows (interaction-only predictive drifting features would be stripped; homecredit lost
  80 features); ecom has 45 unique timestamps of unknown semantics.
- **C9. Latent landmines** (didn't fire in the current artifact): single-non-None-seed can
  yield a width-0 CI and mint CONCEPT (`measured` counts Nones); `_load_csv` even-thinning;
  EMBER row sits as stale=None mislabeled DEPLOYMENT-STABLE in an old artifact.

---

## Answers to the meta-questions (Parts 9–10)

- **Falsifiability (9A)**: as framed, every real-data outcome lands in a publishable cell —
  the map itself is descriptive and needs *robustness*, not Popperian falsification. The
  falsifiable content is the instrument-validity claims, and today's audit falsified three of
  them (D-gate semantics F2; model-agnosticism F3; noise-robustness of the CONCEPT verdict
  F1). The v1 pre-registered falsifier ("DEPLOYMENT-STABLE on all ⇒ broad negative") was NOT
  met and is recorded nowhere. The revised claim ("no exploitable-by-tree-ensembles mean-rule
  drift detectable on TabReD train segments above per-dataset delta; blindness certificates
  where shown") is falsifiable: a learnable-injection-validated dataset with lower-CI > floor
  staleness that survives the noise-drift and K-sweep controls would break it.
- **Estimand (9D)**: currently circular in exactly the suspected way (instrument defines
  concept; gate defines trust; injection defines gate-failure) — the external anchors are
  insects (designed drift) and the synth suite; the fix is either the missing lemma
  (E[staleness]≤0 under fixed P(y|x) + well-specification, with the class-dependence
  disclosed) or an explicit retraining-regret estimand, which is what staleness actually
  measures.
- **Contribution (10G)**: honest label today = strong workshop paper / tech report. D&B
  main requires: noise-drift + missingness-drift controls in the suite (the noise one now
  demonstrably FAILS and needs an instrument repair, e.g., robust-loss staleness or a
  variance-drift co-test), group-aware D, injection learnability gate, K-sweep, model-class
  panel on real data (RF/LightGBM), ≥2 external known-drift anchors (EMBER by-value with the
  NO-DATA guard, insects variants, river panel), a frozen PREREG_DEPLOYMENT_V2.md, and one
  confirmatory fresh-seed rerun. WhyShift foregrounded as closest prior.
- **Reproducibility (10H)**: instrument committed & fully seeded (good); everything else
  missing: results untracked (`/results/` gitignored), env unpinned (server sklearn 1.9.0
  known only from a commit message), run configs unrecorded (headline used --n-seeds 10 vs
  default 5), append-mode artifacts mix versions, `river` undeclared, Kaggle manual gates.
  A third party cannot regenerate the Part-7 table from the repo today.

## The single decisive experiment

Already run (locally, this audit): the **early-noisy regression control** — it reproduces the
sberbank cell (+0.021 vs +0.024) with zero mean-rule change. The corresponding server-side
confirmation (cheap, one evening): per-window label-noise profile on sberbank (residual
variance of a fixed HGB under rolling windows) + injection/learnability + K-sweep on the
CONCEPT branch. If early windows are noisier — the expected result for 2011-12 Russian
housing — the cell is formally withdrawn and the paper becomes the (defensible) instrument +
abstention-map story with insects as the designed-drift positive.
