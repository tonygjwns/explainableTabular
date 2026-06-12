# RESULTS — experiment ledger (검토용)

> 사실(숫자) 중심. 각 결과에 **신뢰등급** [solid / tentative / artifact-risk] + 그 결과가
> 지지하는 한 줄 claim. 해석·계획은 FINDINGS.md / PLAN_RESCUE.md, 사고흐름은 REVIEW.md.
> 모든 실험은 커밋 메시지에 근거 기록(`git log --oneline`). (작성 시점: Q2a 실행 중)

---

## 1. Phase 0 — TabM 재현 (8개, default 시간 split, 5시드)  [solid]
| | 우리(TabM) | TabReD MLP | | 우리 | MLP |
|---|---|---|---|---|---|
| sberbank(rmse) | 0.2572 | 0.2508 | weather(rmse) | 1.5073 | 1.5470 |
| cooking(rmse) | 0.4807 | 0.4820 | ecom(auc) | 0.5906 | 0.6015 |
| delivery(rmse) | 0.5522 | 0.5504 | homesite(auc) | 0.9615 | 0.9500 |
| maps(rmse) | 0.1614 | 0.1622 | homecredit(auc) | 0.8518 | 0.8545 |
→ 공개치 ±1%~노이즈 내. **파이프라인 신뢰 가능.**

## 2. Test 1 — 시간-인덱싱 vs 고정 메모리 (격리: inject off, 10시드)  [solid, 단 as-built 메커니즘]
delta≈0, Hedges' g = +0.05 / −0.17 / −0.15 / −0.05 (sberbank/ecom/homesite/homecredit), 4/4 비유의.
→ 시간-인덱싱 성능 이득 0. **단, 이때 메커니즘은 *부서진 상태*(아래 3) — 해석 주의.**

## 3. 학습 진단 (mem_gap 등)  [solid, as-built]
concat·τ=1: **mem_gap≈0**(메모리 꺼도 손실 불변), 메모리 grad(base 5e-4~1e-3) ≪ backbone/pred(1e-1~7e-1),
검색 PR=620/1000(거의 균일). τ=0.1→PR=1(1개로 붕괴). → **메모리 장식(z-지름길).**
→ Test 1 null은 "메커니즘 미작동"의 산물(가설 기각 아님).

## 4. 합성 positive control (순수 concept drift)  [solid]
`y=x·w(t)` 회전. time-indexed RMSE **0.13** vs fixed **1.03(=std(y))**, **+87%**, mem_gap+0.93.
→ **구현 정확(버그 아님).** 메커니즘은 concept 있으면 착취함.

## 5. 드리프트 분해 (G3, 모델-경량)  [covariate: solid / concept-ρ: blunt]
covariate(시점분류기) AUC: sberbank/homesite/ecom/homecredit/weather **≈1.0**, delivery 0.95, cooking 0.80, maps 0.64.
**drop-top5 후도 높음 → pervasive**(소수 프록시 아님). label ρ(t,y) ≤ 0.13(약함, 단 ρ는 둔한 지표).

## 6. concept gap (early→late, *전역*)  [artifact-risk → #9로 대체]
sberbank +36%, homecredit +4%, 나머지 작음/노이즈. **covariate 외삽에 오염**(공통support 무시) → 신뢰 불가.

## 7. Elec2 (decider, *as-built* 메커니즘 = Fourier·학습V_k·붕괴)  [tentative — 재검 중(Q2a)]
- random, inject off: time 0.954 vs fixed(무시간) 0.911 → *시간*은 유효(메모리 아님).
- random, inject on(시간=피처): **fixed(피처) 0.9615 ≥ memory**; 메모리 추가 +0.0017 (p=.11, g=.76, n.s.) → **메모리 ≤ 시간-피처.**
- temporal(외삽), inject off: time 0.875 vs fixed 0.894 (g=−0.67), 불안정.
→ ⚠ **부서진 메커니즘**으로 얻음. "+4.3"은 시간-피처 baseline 없이 과대해석했던 것. trend 기저+load-balance로 **재검 중(Q2a)**.

## 8. Q1 — 기능적 충실성 게이트 (재설계: memory_only+trend+load-balance)  [PASS, 게이트로 solid]
ceiling(MLP+t) 0.990 / floor(shuffle-t) 0.894 / **모델 recovery 0.991 (mean, 10/10 PASS)**.
metric = per-t `cos(ŵ(t), w(t))`, ŵ(t)=E_x[∂(logit)/∂x] (게이지-고정, Procrustes 없음).
→ **재설계 메커니즘은 충실·정확**(해석 청구 필요조건 성립). ⚠ 단 동적범위 좁음(≤90° drift→floor 0.894) → 헤드라인엔 큰-회전 robustness 1회 필요.

## 9. F3 feasibility + within-overlap concept  [solid] ★핵심
- **F3**(전역): 측정가능 = cooking(overlap .888/ESS4024)/maps(1.0/7513)뿐 — *둘 다 저-covariate·concept~0*.
  고-covariate 5개: overlap 0 → 측정불가. elec2: overlap .438 **but ESS 20**(전역-IW 꼬리 아티팩트).
- **within-overlap**(covariate-매칭, 전역 reweight 없음):
  | | concept_gap | n_ov e/l |
  |---|---|---|
  | **elec2** | **+0.132 AUC** (OOF; early 0.716 / late 0.848; strata [+0.12,+0.17] 안정) | 9173/4721 |
  | cooking | −0.007(rmse) | 16820/17930 |
  | maps | −0.003 | 20000/20000 |
  | delivery | −0.033 | 1290/686 |
  | 고-covariate 5개 | 측정불가(n_ov=0) | — |
  - *정의 명확화*: gap은 **고정된 late-overlap-test 위에서 early-학습 vs late-학습의 전이 격차**
    (AUC(late)−AUC(early), 같은 테스트셋) = **concept(난이도 아님)**. 코드 확인됨.
  - *조이기(재실행 대기)*: 영역선택을 **out-of-fold p**로(in-sample 낙관 제거) + **p-층별 gap 안정성**
    (잔여 covariate면 층마다 들쭉) — `run_concept_overlap` 갱신됨, 재실행해 +0.166 확정 예정.
→ **elec2는 공통support 위에 *크고 측정가능한* concept(+0.166)** 보유. 이분법 정밀화:
  *고-covariate⇒측정불가 / 저-covariate⇒측정가능하나 concept0 / elec2(중간)⇒support+concept 둘 다.*

## 10. Q2b — 인스턴스 time-TabR *구조* vs 시간-*피처* (elec2, 3-arm 요인설계)  [~~solid~~ → **해석 무효화, 재검정 중**] ★결정

> ⚠ **V2 무효화 (2026-06-12, 외부 감사)**: 아래 *숫자*는 유효하나 **"구조 ≤ 피처" 해석은 무효**.
> (i) linear value 훅이 집계 하에 `Σw·Linear(Δτ)=Linear(Σw·Δτ)`로 붕괴 — 실제 검정된 것은
> "검색-가중 Δt 피처 vs 직접 t 피처"이지 구조 vs 피처가 아님. (ii) 검색 기판이 sub-TabR
> (온도/key-proj 없음, train 255 vs eval 4096 context 불일치, lr만 튜닝). (iii) time_tabr만
> 직접 시간 피처 미보유(교락). → **PLAN_V2 §R0–R1 재검정으로 대체 예정, 결정규칙 = PREREG_V2.**
3-arm 공유인코더: `mlp_t`(시간=피처) / `tabr`(검색, 시간없음) / `time_tabr`(검색+value 시간훅, time_mode=value).
elec2 temporal/trend, per-arm lr 선택(oracle 상한), 다중시드. (`run_elec2_q2.py`, `diagnostics.jsonl`)
- **버그 vs drift 판별(①, train_loss+val 곡선)**: train_loss 전부 감소(버그 아님). **temporal은 argmax_val=epoch2~4(조기 peak 후 하락), random은 epoch~55(매끄러운 상승)** → 사전등록 drift 지문 확정. (step 해상도 ③도 동일.)
- **무정규화 10시드 oracle**: mlp_t **0.9054** > time_tabr 0.9003 > tabr 0.8969. (time_tabr−mlp_t = −0.005)
- **정규화+min_epochs 20 (dropout.1/wd1e-4, 메커니즘 engage 보장; best_ep 11·14·36↑) 10시드 oracle**:
  mlp_t **0.9027** > tabr 0.8955 > **time_tabr 0.8848**. **time_tabr−mlp_t = −0.018**(더 악화), time_tabr−tabr = **−0.011**(검색을 해침), mlp_t−tabr = +0.007(시간 자체는 도움).
  **time_tabr std 0.047~0.075** (lr↑서 0.73~0.92 난동) — 정규화로도 불안정 → `(t_i→t_q)` 보정항 **ill-conditioned**.
- **⚠ paired 재진술 (정직, 비평 반영)**: arm들이 시드/split/init 공유 → marginal std(0.05)가 아니라 *차이의* SE가 척도.
  무정규화 paired: time_tabr−mlp_t = **−0.005, SE .0035, 95%CI [−0.013,+0.003], 1.45 SE, sign 2/8 → 0과 구분 불가**.
  즉 elec2의 "구조<피처"는 **consistent-but-uninformative**(noise null). val→test Spearman 0.07 = elec2는 regime/autocorrelation
  때문에 *모델 비교 기판으로는 노이즈*(데이터엔 concept +0.132 있음과 모순 아님 — concept은 있으나 비교엔 부적합).
→ **elec2는 음성을 *나르지 않음*(증거력 없음). 음성의 무게는 Insects(§11, val→test 0.87, 깨끗)가 짊어짐.**
  ("메커니즘 미engage" 위협은 min_epochs로 제거됨 — 별개로 유효.)

## 11. Q2b 둘째 데이터셋 — INSECTS (designed drift, multiclass, accuracy)  [~~solid~~ → **해석 무효화, 재검정 중**] ~~★≥2 잠금~~

> ⚠ **V2 무효화 (2026-06-12)**: §10과 동일 사유로 "≥2 데이터셋 잠금" 철회. 추가로: 시드 10은
> 사전등록 25 미달이고 CI 상단이 +0.001로 0을 스침. 단 **time_tabr−tabr=+0.028(시간-구조가 검색
> 안에서 작동)은 불구 훅으로도 양수**였음 — V2 재검정에서 뒤집힐 가능성이 실재하는 근거. → PREREG_V2.
`incremental_balanced`, temporal, 정규화+min_epochs 20, 10시드 (`run_elec2_q2.py --dataset insects`).
- ①곡선: train_loss 감소(버그 아님), argmax_val **15~20 epoch**(elec2 epoch1~4와 달리 **trivially 쉽지 않음** = 비-trivial concept).
- 결정표 oracle: mlp_t **0.6704** > time_tabr 0.6594 > tabr 0.6320. **PAIRED per-seed(oracle lr, 10시드, 정직한 척도):**
  | 대비 | mean diff | SE | 95% CI | sign +/− | \|m\|/SE |
  |---|---|---|---|---|---|
  | **time_tabr − mlp_t** | **−0.011** | .0052 | **[−0.023, +0.001]** | 3/7 | 2.12 (borderline) |
  | time_tabr − tabr | +0.028 | .0066 | [+0.013, +0.043] | 9/1 | 4.14 (유의+) |
  | mlp_t − tabr | +0.038 | .0040 | [+0.029, +0.048] | 10/0 | 9.58 (강한+) |
- **★구조 vs 기판 분해 (비평 반영, 선제 진술)**: 지는 건 *시간 메커니즘이 아니라 검색 substrate*다.
  `mlp_t−tabr=+0.038`(검색 기판 ≪ parametric MLP) + `time_tabr−tabr=+0.028`(**시간-구조는 검색 안에서 확실히 작동**)
  → 시간-구조가 substrate 적자(−0.038)를 못 메워 결합 구조가 피처를 못 넘음(−0.011). **redundancy 논증과 일관.**
- **val→test Spearman = +0.87** (elec2 0.07) → Insects는 val이 test 예측(깨끗) → elec2 붕괴는 **elec2 고유**.
- time_tabr std 0.014~0.033 (mlp_t .006~.009) — paired SE는 0.0052로 훨씬 tight.
→ **음성은 Insects가 나름**(borderline-significant, CI 상단 +0.001로 0 스침). time_tabr−mlp_t<0이 일관·near-sig =
  "**테스트 가능한 깨끗한 벤치에서 구조가 시간-피처를 못 넘음**"(강한 oracle 형태 음성 — oracle 선택을 줘도 못 넘음).
  단 배포 주장은 불가(elec2 val→test 0.07). **분야 class 음성엔 더 많은 깨끗한 데이터셋·방법 필요(아래 NEXT_TAB).**

---

## 신뢰등급 요약 (무엇을 믿나)
- **Solid**: Phase0 재현 · 구현 정확(합성 +87%) · 강한 pervasive covariate drift · **elec2 within-overlap concept +0.166** · Q1 충실성 PASS(게이트).
- **Tentative/대체됨**: 전역 concept gap(#6, 오염) · elec2 as-built 음성(#7, 부서진 메커니즘 — 재검 중).
- **Open(실행 중)**: **Q2a** — *재설계 **프로토타입** 메모리*가 elec2 concept에서 돕는가. ⚠ **이건 약한 프록시**:
  학습 V_k(§4.2 정보없음 결함)는 trend+load-balance로 안 고쳐짐 → Q2a null을 "구조 실패"로 해석 금지.
  **진짜 구조 테스트 = Q2b 인스턴스 V_k(TabR)** + 같은-모델 시간-조건 on/off ablation(아래).

## 헤드라인 (정밀, 2026-06-10 재프레이밍 — 비평 반영) — ⚠ Claim B 부분은 §10·§11 무효화로 보류(2026-06-12, PLAN_V2/PREREG_V2 재검정 대기. Claim A 부분은 유효.)
**리드 = Claim A (robust·신규)**: 현실 표 시간데이터는 **covariate 지배** → 강한 covariate가 early/late 공통support를
무너뜨려 **concept을 표준 조건부 렌즈로 측정조차 어렵다**(~10 데이터셋; F3/within-overlap 프레임 + 진단 도구킷이 받침).
**보조 = Claim B (구조 ≤ 피처)**: 테스트 가능한 곳에선 시간-인덱싱 *구조*가 시간-*피처*를 못 넘음 — **Insects(깨끗, val→test .87)
paired −0.011, 95%CI[−.023,+.001], borderline**이 나르고, **elec2는 consistent-but-uninformative**(paired −.005, CI가 0 포함; val→test .07).
분해: 지는 건 *시간 메커니즘이 아니라 검색 substrate*(tabr−mlp_t=−.038; time_tabr−tabr=+.028) → redundancy와 일관.
*주의(전칭 금지)*: "시간방법 무용" 아님 — 시간-피처는 도움(Insects mlp_t−tabr +.038). 메커니즘도 안 망가짐(Q1 충실 PASS).
미확정은 "구조 > 피처"가 거짓. **B는 A를 약화시키지 않게 분리** — B는 1개-clean(+1 noisy)이라 class 음성엔 다중 데이터셋·방법 확장 필요.

## 리뷰어용 열린 질문
1. within-overlap concept 측정(in-sample p로 영역 선택)이 elec2 +0.166을 과대평가하지 않나? (held-out p로 재확인 가치?)
2. Q1 동적범위(floor 0.894)가 좁다 — 큰-회전 robustness 전에 "충실성 PASS"를 헤드라인으로 써도 되나?
3. ~~elec2 단일 measurable-concept 벤치 — 일반성?~~ **해결**: Insects(designed-drift, multiclass) 추가 →
   구조 ≤ 피처 음성이 **두 데이터셋서 일관**(§11). 더 강하려면 abrupt variant·random 대조 추가 가능.
4. Q2에서 time-feature baseline의 t-인코딩을 메모리와 같은 기저로 정합하는 것 외에, "구조 기여"를 더 깨끗이 가를 방법?

## 포인터
- 스크립트: run_phase0 / run_phase1_sanity(Test1) / smoke_test_phase1 / run_synth_control(#4) /
  run_drift(#5) / run_conceptdrift(#6) / run_elec2(#7, Q2a) / run_q1_faithfulness(#8) /
  run_f3_feasibility · run_concept_overlap(#9).
- 결과 JSON: `results/phase1/{sanity,q1,f3,concept_overlap,elec2,drift,conceptdrift}/`.
- 문서: FINDINGS.md(증거·계획) · PLAN_RESCUE.md(사전등록 프로토콜·결정규칙) · REVIEW.md(전체 비판).
