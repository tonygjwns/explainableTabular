# NEXT_TAB — 인계 (이어서 작업할 새 탭용)

> 워크플로우: 로컬(이 repo)서 코드 작성→git push, 서버(`explaintab311` env, py3.11)서 pull→실행.
> 서버엔 Claude 없음 → 로컬서 완성해 push. 최신 커밋 = `git log --oneline -1`.

## ☆☆☆ 현행 = V3 재건 진행 중 (2026-06-17 갱신) — 새 탭은 여기부터
> 읽는 순서: **이 블록 → `PLAN_V3.md`(현행 계획·게이트 판정) → `RED_TEAM.md`(7-에이전트 검토·왜 재건)
> → `PAPER_DRAFT_V3.md`/`_KO.md`(현행 초안) → RESULTS.md**. (PLAN_V2/PREREG_V2/PAPER_DRAFT(v0.1)는 역사.)

**무슨 일**: v0.1 초안을 7-에이전트 적대 검토(RED_TEAM.md)로 자가-red-team → 리드 청구(Claim A 보편형)에
구조적 구멍 발견(home-field 교락·표현 의존성·estimand 부재·퍼즐 미입증) → **재건 결정(PLAN_V3)**. 규율 =
nucleus 죽일 결정적 게이트 *먼저*, 형식 재작성 *나중*.

**V3 진행 상태**:
- **V3.0 게이트 전부 통과/재범위화 (PLAN_V3 상단 판정)**: G1 placebo → elec2 +0.146/insects +0.150이 placebo
  한참 위 = 진짜 concept(home-field 아님). G2 표현 → disjoint TabReD 4/5가 희소표현서 측정가능+concept≈0,
  ecom만 진짜 disjoint, concept 양성은 표현 바꿔도 생존. G3 도구킷 견고(노이즈 +0.034 caveat). G4 웹: 신규성=
  측정불가성+abstention로 좁힘(adversarial-validation 계보 인용 필요, WhyShift가 대조 지지).
- **V3.1 형식 척추 완료**: PAPER_DRAFT_V3(.md/_KO) — estimand(DISDE term-ii 시간축) + positivity 정리(§9
  자기모순 해결) + 표현-인지 §6 + 게이트 통제(§5 placebo 표).
- **V3.2 C1 완료**: cov_AUC가 TabReD per-dataset margin 예측 **못 함**(Spearman +0.22 p=.61; ecom 반례) →
  §7 정직 축소("TabReD 퍼즐 설명 주장 안 함"). `c1_ranking_summary.json`.
- **🔄 V3.2 C5 = 서버 실행 대기 (지금 인계 지점)**:
  ```
  pip install folktables
  python scripts/run_whyshift.py --states CA TX NY FL PA --years 2014 2018 --task income
  # → whyshift_summary.json 로 루트 환류. 예측: 공간 gap>시간 gap, 시간 cov_AUC>공간(X지배).
  ```
- **⬜ 남은 것**: C5 결과 반영(§2·§6) → V3.3 위생(BH-FDR 전 contrast family / Claim A 임계 사전등록·민감도
  그리드 / 모든 실 gap seed-CI / ℓ-robustness Brier·Bayes-risk·KL / rolling-origin gap 궤적) → C2 앵커
  (no-change/GBDT+t/정품 TabR) → C3 §9 faithful Cai&Ye(선택) → 지도교수 정렬.

**V3 코드맵(새로 추가)**: `run_gap_controls.py`(G1 placebo+nulls), `run_representation.py`(G2 표현),
`run_toolkit_adversarial.py`(G3), `run_c1_ranking.py`(C1), `run_whyshift.py`(C5). 진단 코어
`src/analysis/drift_measure.py`에 `concept_within_overlap(permute_time=)` 추가됨(placebo).
**결과 artifact(루트, _synth=합성/없음=실데이터)**: gap_controls_*, representation_*, toolkit_adversarial_,
c1_ranking_, disde_degeneration_, toolkit_validation_, modulation_adj_, summary_fourier, q1_verdict_*.

**다음 탭 첫 행동**: (1) 서버서 C5 돌고 있으면 `whyshift_summary.json` 회수·반영. (2) 안 돌았으면 위 명령 전달.
(3) 병렬로 V3.3 위생 코드 작성(로컬, 정렬 전 필수). 상세·우선순위 = PLAN_V3 §V3.2/§V3.3/우선순위.

---

## (이력) R0~R2 완료 (2026-06-17) — V3 재건 *이전* 상태
- **R1**: V2 재검정 = 구조 음성(유의, CI<0). **R2.1** 문헌(A 미선점). **R2.2** DISDE 퇴화표 10데이터셋 3분법(RESULTS §13).
  **R2.4** 도구킷 ground-truth 검증 4/4 PASS(§14). **R2.5** Q1 큰-회전: 바닥 0.894→0.017, 복원 0.988 10/10 PASS(§15).
- **R2.3 판결**: 정의적(label-free⇒X-side)=확정. 경험적(최소 재현 gain↔cov_AUC)=trend 외삽붕괴(−0.5 무효)→
  fourier도 ~null(+0.231 약함)=충실 재현 아님 → **PREREG §8대로 inconclusive, LAMDA repo gold 재현=future work**(§16).
- **남은 것**: ①지도교수 정렬(R1+R2 들고) ②D&B 범위확장(데이터셋·방법 sweep, 도구킷 패키징, R2.3 faithful 재현).
- 코드/실험 미실행 없음. 결과 artifact: `summary_fourier.json`, `q1_verdict_a6.28_fourier.json`, `disde_/toolkit_ 등`.

## (이력) R1 완료 — V2 재검정 판정 = 구조 음성(유의) (2026-06-14)
25시드 본 실행 완료. **PREREG §4 판정: 구조 우위 = NO**, 교락 없는 *유의* 음성.
주 대비 `time_tabr_t − tabr_t`(temporal, val-fair): incremental −0.0067 [CI −.012,−.001] p=.006,
incremental_abrupt −0.0205 [−.034,−.008] p<.001 — **두 clean 변종 모두 CI<0 유의 음성**. (abrupt ρ=−.43
게이트 탈락; reoccurring ρ=.33 경계지만 trend-기저 외삽 병리 time_tabr_t→0.19, §4 red flag → 크기 비사용; elec2 ρ=−.34 보조.)
- **교락 제거로 pre-V2보다 깨끗·강함**: ①기판 경쟁력(tabr_t−mlp_t≈0~+.011, −.038 적자 소멸) ②시간은 검색 도움(time_tabr_t−tabr=+.042)
  but 피처가 더 나름 ③**★in-dist vs 외삽 뒤집힘**: random서 훅 도움(+.005,+.021)/temporal서 해(−.007,−.021)
  = 시간-인덱싱 훅은 in-dist 장치, 외삽 장치 아님(redundancy 직접 뒷받침) ④trend기저는 비단조 drift서 외삽붕괴(Claim A 먹이).
- 결과 jsonl 환류·커밋. RESULTS §12, FINDINGS "V2 RE-TEST VERDICT" 참조.
- **R2 진행상황 (PLAN_V2 §R2)**:
  - ✅ **R2.1 문헌 검증(웹 원문)**: Claim A 코어 **미선점**(측정프레임만 DISDE와 PARTIAL). REFERENCES §0.
  - ✅ **R2.2 DISDE 퇴화**(`run_disde_degeneration.py`): 10데이터셋 3분법 확정. RESULTS §13. (서버 실행·환류 완료.)
  - 🔄 **R2.3 Cai&Ye 판결 인프라 완료**(아래 커밋). **정의적 절반 = 코드로 증명**(변조는 label-free X-side,
    REFERENCES §0.1). **경험적 절반 = 서버 실행 대기**:
    ```
    python scripts/smoke_test_modulation.py    # 배선/identity-init 검증 (먼저)
    python scripts/run_modulation_adjudication.py --config configs/phase1.yaml --all --elec2 --insects --n-seeds 5
    ```
    기대: gain↔cov_AUC 양의 상관, gain↔concept_gap ~0, **cooking/maps(concept≈0)서도 변조 이득>0 = X-side**.
    결과 `results/phase1/modulation_adj/summary.json` 환류.
  - ✅ **R2.4 도구킷 검증**(`run_toolkit_validation.py`): covariate×concept 4×4, **4/4 PASS**. RESULTS §14. (로컬 실행 완료.)
  - 🔄 **R2.5 Q1 큰-회전 인프라 완료**(`run_q1_faithfulness.py` 확장: `--angle-max`/`--basis`/`--n-harmonics`/`--tag`).
    기존 게이트(π/2+trend)는 기본값으로 불변. **R2.5 robustness = 큰 회전+Fourier 정합**(기저-불일치 교락 회피):
    ```
    python scripts/run_q1_faithfulness.py --angle-max 6.283 --basis fourier --n-harmonics 4   # 2π 전회전
    # (변형) --angle-max 3.1416 --basis fourier  # π 반회전(규칙 완전반전)
    ```
    기대: 큰 회전이면 바닥(shuffle-t)이 ~0으로 내려가 동적범위 넓어짐 → 메커니즘이 그래도 복원(≥PASS선)하면
    "충실"이 헤드라인 논거로 robust. 출력 `results/phase1/q1/q1_verdict_<tag>.json`. ⚠ 출력 파일명이 태그식으로
    바뀜(기존 `q1_verdict.json`→`q1_verdict_a1.57_trend.json`).
  → R1+R2 결과 들고 지도교수 정렬(워크숍 now / **NeurIPS D&B 주타깃**). Claim A 리드, B는 분해와 함께 보조.

## 한 문단 요약 (2026-06-12 대전환 — 위 ★★가 현행 최신)
**외부 감사로 Q2b "구조 ≤ 피처" 음성의 *해석*이 무효화됨**: (i) linear value 훅이 집계 하에
"Δt 피처 1개"로 붕괴(stale-label 보정 표현 불가 — 가설을 검정한 적 없음), (ii) 검색 기판 sub-TabR
(온도·key-proj 없음, train/eval context 불일치, knob 미튜닝), (iii) time_tabr만 직접 시간 피처 미보유.
또한 Claim A의 측정 프레임은 DISDE/WhyShift와 겹쳐 재포지셔닝 필요(인용+적응; 신규성은 "시간축
실증 + support 붕괴로 인한 측정불가" 발견에 있음 — 웹 검증 필요). **계획 전면 갱신 = PLAN_V2.md**:
R0(코드 수정, 완료) → R1(25시드 재검정, 서버) → R2(Claim A 재포지셔닝·Cai&Ye 판결) → R3(정렬·학회).
유효 자산: Phase 0 재현, Q1 PASS, within-overlap concept(+0.132), elec2 val→test 붕괴 발견, 진단 도구.

## V2 인프라 (R0 — 구현 완료, 이 커밋)
- `src/models/tabr.py`: **value_hook {mlp(주)|gate|linear(레거시)}** — mlp/gate는 집계에서 살아남는
  라벨×Δτ 상호작용(zero-init 등가 보존); **learnable τ + √d 스케일링**, **key projection**,
  metric 훅을 **top-k 이전**으로, **arch 5종 {mlp_t, tabr, tabr_t, time_tabr, time_tabr_t}**
  (`_t` = predictor에 τ(t) 직접 concat — 피처 보유 통제).
- `src/training/tabr_trainer.py`: **train ctx = batch+sampled-4096**(배치 제외)/inbatch(레거시),
  **eval ctx = full train**/fixed(레거시, 전용 RNG로 arm 공유), n_classes train∪val, 배치-skip 공통화,
  eval 시 context 1회 인코딩.
- `src/utils/stats.py`: **hedges_g_paired**(d_z·J) 추가 — 시드-페어 비교의 정직한 효과크기.
- `scripts/run_elec2_q2.py`: 5-arm, **val-fair+oracle 병기**, PAIR_PRIORITY(주 대비
  `time_tabr_t−tabr_t`), `--legacy`(pre-V2 정확 재현), `--topk` 등 knob CLI.
- `scripts/run_anchors.py` (신규): lgbm±t / knn±t / **no-change**(persistence) — 외부 보정.
- `scripts/smoke_test_tabr.py`: **선형-붕괴 표현력 가드**(hook이 문서화된 표현력과 다르면 FAIL) +
  init 등가성 테스트. `smoke_test_tabr_trainer.py`: 5-arm + 레거시 경로.

## ★ 다음 행동 — 서버 (순서대로)
```bash
conda activate explaintab311 && cd ~/explainableTabular && git pull
# 0) 배선 검증 (CPU/GPU 무관, 수 분)
python scripts/smoke_test_tabr.py
python scripts/smoke_test_tabr_trainer.py
python scripts/smoke_test_insects.py
# 1) 앵커 (외부 보정; lightgbm 없으면 pip install lightgbm)
python scripts/run_anchors.py --dataset elec2 --split temporal
python scripts/run_anchors.py --dataset insects --insects-variant incremental_balanced
# 2) R1.1 튜닝 단계 (3시드, val로 topk 선택 — PREREG_V2 §2.1; 변종별 1회씩)
for K in 8 32 128; do python scripts/run_elec2_q2.py --dataset insects --report-grid \
  --n-seeds 3 --splits temporal --bases trend --lr-grid 1e-3 5e-4 2e-4 \
  --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk $K; done
# 3) R1.2 본 실행 (25시드; PREREG_V2 §2.2 — topk는 2)에서 val로 고른 값)
python scripts/run_elec2_q2.py --dataset insects --insects-variant incremental_balanced \
  --report-grid --n-seeds 25 --splits temporal --bases trend --lr-grid 1e-3 5e-4 2e-4 \
  --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk <선택값>
# + abrupt_balanced, incremental_abrupt_balanced 동일. elec2는 보조(동일 커맨드, --dataset elec2).
# 결과: results/phase1/<dataset>_q2/diagnostics.jsonl 를 commit해 로컬로 환류.
```
**판정은 PREREG_V2 §4에 기계적으로** (주 대비 time_tabr_t−tabr_t, val-fair, clean ≥2/3).

## 병렬 작업 (로컬, R2)
- **R2.1 문헌 원문 검증 (웹 권한 필요, 최우선)**: DISDE(arXiv:2303.02011)/WhyShift(NeurIPS'23)/
  Webb'16/Cai&Ye ICML·NeurIPS'25/Drift-Resilient TabPFN + "TabReD에 X·Y|X 분해 적용한 2025-26 선행
  유무". → REFERENCES.md 갱신.
- R2.2 DISDE 퇴화 실험(시점분류기 재사용), R2.3 **Cai&Ye 판결**(LAMDA 코드, concept≈0인
  cooking/maps에서 변조 이득 = X-side 증명), R2.4 합성 2×2 + INSECTS 변종으로 도구킷 검증,
  R2.5 Q1 큰-회전. 상세 = PLAN_V2 §R2.

## 잠긴 설계 포인트 (V2)
- **주 대비 = time_tabr_t − tabr_t** (직접 시간 피처를 양쪽에 — 구조×주입위치 교락 제거). PREREG_V2 §4.
- value_hook 주 분석 = **mlp**, gate는 ablation. linear는 레거시 재현 전용(`--legacy`).
- 판정 = **val-fair** (oracle은 강한형 보조). 통계 = paired CI + Wilcoxon + **hedges_g_paired**.
- 헤드라인 문장에 (시드 수, CI, 데이터셋 수) 병기 — "잠금/확정" 단독 표현 금지.
- pre-V2 결과는 은폐하지 않고 "무효화된 선행 시도"로 부록 보고. `--legacy`로 재현 가능.
- Q1 지표·within-overlap concept 정의 등 기존 잠금(8라운드)은 유지.

## R1.1 튜닝 결과 (2026-06-13, 서버 3시드 — 본 실행 전 기록)
서버에서 0단계 smoke 3종 통과 + 앵커 + topk 튜닝(3시드) 완료. 결과는
`elec2_q2_diagnostics.jsonl` / `insects_q2_diagnostics.jsonl`(로컬 환류됨, repo 루트).

**① topk 선택 (PREREG_V2 §7 규칙 = 검색 3-arm best-mean-val 평균 최대 K):**
- incremental_balanced → **32**, incremental_abrupt_balanced → **128**, abrupt_balanced → 8(탈락예정),
  elec2(보조) → 128. ※ topk는 4째 소수점에서 갈리는 무의미 축(노이즈) — 해석 시 민감도 없음 보고.

**② ★abrupt_balanced가 val→test ρ 게이트 탈락 (PREREG_V2 §3):**
- ρ: incremental +0.80~0.95 ✅ / **abrupt −0.43~−0.64 ❌** / incremental_abrupt +0.57~0.70 ✅.
- 원인: abrupt drift + trend 외삽 상호작용(test t가 train 밖 → 틀린 regime으로 외삽 → 시간피처가 해침).
  앵커가 예고: abrupt에서 lgbm_t 0.459(lgbm 0.664 대비 폭락), V2 `*_t` arm도 0.55~0.59로 붕괴(tabr는 0.65 건재).
- → clean에서 제외, `incremental_reoccurring_balanced`로 보충. 부수발견: "trend 시간피처는 abrupt서 실패".

**③ 예비 신호(3시드, 판정 아님):** 주 대비 `time_tabr_t−tabr_t` ≈ 0(clean 변종 −0.001~−0.045),
  보조 `time_tabr_t−tabr`·`tabr_t−tabr`는 일관 +(+0.01~+0.04). **V2 기판 수정 작동**: INSECTS `tabr_t`≈0.685로
  pre-V2 mlp_t(0.670) 상회, elec2 `tabr_t`≈0.90으로 옛 mlp_t 동급(감사의 substrate 적자 −0.038 상당 해소).
  → 잠정 해석 "시간은 기판을 돕지만 *구조적 인덱싱*이 *피처*를 못 넘음"(교락 없는 유효 음성 후보). 25시드가 결정.

**앵커 기록 (1셀 seed0):** elec2 lgbm 0.887/lgbm_t 0.884/knn 0.851/**no_change AUC 0.845**(acc 0.846 — 문헌 비판대로 강함,
보고 의무) → 신경망 arm(mlp_t 0.90)이 그 위. INSECTS no_change 0.16~0.59(위협 아님). lgbm_t가 incremental서 0.679로
pre-V2 최고 arm(0.670) 상회 → 외부 바닥선 필수.

## ★ R1.2 본 실행 (다음, 서버) — 커맨드는 위 "다음 행동"이 아니라 아래
```bash
# (a) 보충 변종 topk 튜닝 3시드
for K in 8 32 128; do python scripts/run_elec2_q2.py --dataset insects \
  --insects-variant incremental_reoccurring_balanced --report-grid --n-seeds 3 --splits temporal \
  --bases trend --archs tabr tabr_t time_tabr_t --lr-grid 1e-3 5e-4 2e-4 \
  --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk $K; done
# (b) 본 실행 25시드 4-arm temporal: incremental(k32) / incremental_abrupt(k128) /
#     incremental_reoccurring(k=위 선택) / elec2 보조(k128) / abrupt(k8, 게이트가 잠긴규칙으로 탈락시키게 형식상 1회)
#     archs = mlp_t tabr tabr_t time_tabr_t. 명령 형식은 incremental 예:
python scripts/run_elec2_q2.py --dataset insects --insects-variant incremental_balanced \
  --report-grid --n-seeds 25 --splits temporal --bases trend --archs mlp_t tabr tabr_t time_tabr_t \
  --lr-grid 1e-3 5e-4 2e-4 --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk 32
# (c) random 대조 10시드(clean 통과 변종만): --splits random, 나머지 동일.
```
판정 = PREREG_V2 §4 (주 대비 time_tabr_t−tabr_t, valfair, clean ≥2/3, CI>0∧p<.05∧g_z≥.5). 결과 2 jsonl 환류.

## 미커밋 상태
없음. 다음 = 서버 R1.2 본 실행(위 ★).
