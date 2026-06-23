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

## 12. Q2b V2 재검정 — 교락 제거판 (INSECTS designed-drift 4변종 + elec2, 25시드)  [solid] ★R1 판정
PREREG_V2 프로토콜: 5-arm(mlp_t/tabr/tabr_t/time_tabr_t), 비퇴화 mlp 훅, learnable τ+key-proj,
train sampled-4096/eval full-train context, val-fair+oracle 병기, **주 대비 = time_tabr_t − tabr_t**
(직접 시간 피처를 양쪽에 고정 → 구조×주입위치 교락 제거). topk: PREREG §7 규칙(검색 3-arm best-val 평균).
- **val→test ρ 게이트(25시드)로 clean 셋 확정**: incremental +0.89 ✅ / incremental_abrupt +0.77 ✅ /
  reoccurring +0.33 △(경계, 병리) / **abrupt −0.43 ❌** / elec2 보조 −0.34 ❌(기존 0.07과 일관).
- **주 대비 time_tabr_t − tabr_t (temporal, val-fair, 25시드):**
  | clean 변종 | mean diff | 95% CI | p | g_z | 판정 |
  |---|---|---|---|---|---|
  | incremental_balanced | **−0.0067** | [−0.012, −0.001] | .006 | −0.50 | **유의 음성** |
  | incremental_abrupt_balanced | **−0.0205** | [−0.034, −0.008] | <.001 | −0.63 | **유의 음성** |
  | reoccurring(병리) | −0.358 | — | — | −3.83 | time_tabr_t→0.19, std 4.3×mlp_t = §4 red flag, 크기 비사용 |
- **★구조 우위 = NO (PREREG §4).** pre-V2의 "≈0 null"이 아니라 **두 병리없는 clean 변종 모두 CI<0 유의 음성**:
  공정 비교에서 시간을 *구조로 인덱싱*하면 *피처로 넣는 것*보다 손해.
- **분해 (교락 제거 — pre-V2 substrate 변명 소멸):**
  - `tabr_t − mlp_t` = +0.0005 / +0.011 → **검색 기판 ≈ MLP**(pre-V2 −0.038 적자 사라짐; V2 기판 수정 효과).
  - `time_tabr_t − tabr` = +0.042, `mlp_t − tabr` = +0.048 → 시간은 검색 크게 도움, 단 피처가 더 잘 나름.
  - **★in-dist vs 외삽 뒤집힘**: random split 주 대비 = **+0.005 / +0.021**(훅 도움) vs temporal **−0.007 / −0.021**(훅 해)
    → **시간-인덱싱 훅 = in-distribution 장치, 외삽 장치 아님.** "구조가 외삽서 우위" 가설을 정면 반증, redundancy와 일관.
- **부수 발견**: trend 기저는 비단조 drift(abrupt·reoccurring)에서 외삽 붕괴 → temporal 모델선택 무력화(ρ<0.3) → Claim A(측정 신뢰성) 먹이.
→ **R1 완료: 교락 없는 V2-유효 음성(유의). A' method 경로 최종 닫힘. 헤드라인은 Claim A, B는 보조(이 깨끗한 분해와 함께).**

## 13. R2.2 — DISDE 퇴화 vs within-overlap (Claim A 정량 엔진, 10 데이터셋)  [solid] ★Claim A
`run_disde_degeneration.py` (sklearn HGB만, 학습 없음). 데이터셋별 early/late(median t): covariate AUC +
**DISDE식 density-ratio 재가중 건강도**(overlap_mass / ESS% / CV / max-weight) + within-overlap transfer gap.
| 데이터셋 | cov_AUC | drop5 | overlap | ESS% | n_ov | gap | 판정 |
|---|---|---|---|---|---|---|---|
| sberbank_housing | 1.000 | 1.000 | 0.000 | 14 | 0 | — | disjoint → **측정불가** |
| homesite_insurance | 1.000 | 1.000 | 0.000 | 99* | 0 | — | disjoint → **측정불가** |
| ecom_offers | 1.000 | 0.953 | 0.000 | 100* | 0 | — | disjoint → **측정불가** |
| homecredit_default | 1.000 | 0.999 | 0.000 | 100* | 0 | — | disjoint → **측정불가** |
| weather | 1.000 | 0.907 | 0.000 | 100* | 0 | — | disjoint → **측정불가** |
| delivery_eta | 0.997 | 0.994 | 0.061 | 0.21 | 568 | −0.048 | heavy-tail, 프레임 작동(gap 소/음) |
| **elec2** | 0.993 | 0.449 | 0.438 | 0.55 | 4721 | **+0.132** | heavy-tail, **프레임 승**(실concept) |
| cooking_time | 0.753 | 0.627 | 0.880 | 45 | 16960 | −0.005 | DISDE-ok, **concept≈0** |
| maps_routing | 0.566 | 0.568 | 1.000 | 93 | 20000 | −0.003 | DISDE-ok, **concept≈0** |
| **insects_incremental** | 0.707 | 0.707 | 0.973 | 39 | 19000 | **+0.144** | DISDE-ok, **실concept(설계drift)** |
- *ESS%=99~100*은 **완전분리 아티팩트**(early p 전부 clip 바닥→가중치 균일). overlap_mass=0.000이 진짜 신호
  = **disjoint support**(bias 모드). 두 퇴화 모드 분리: **disjoint**(overlap~0: 고-covariate 5개) /
  **heavy-tail**(ESS%~0, overlap>0: delivery·elec2).
- **3분법 확정(10개)**: ①고-covariate 5개 ⇒ support disjoint ⇒ concept 표준렌즈로 **측정불가** /
  ②저-covariate(cooking/maps) ⇒ 측정가능하나 **concept≈0** / ③concept-drift 벤치(elec2/insects) ⇒
  **실concept 큼**(+0.132/+0.144), elec2는 DISDE 재가중 붕괴(ESS 0.55%)지만 within-overlap이 복원.
- **★측정 프레임 검증**: INSECTS는 *설계된* concept drift(ground truth=drift 존재) → 프레임이 **+0.144로 정확 복원**
  (elec2 +0.132와 동급). "측정불가"가 *방법 결함*이 아니라 *support 부재*임을 양방향으로 입증.
→ **Claim A 정량 근거 완성**: "covariate 지배가 concept을 표준 조건부/재가중 렌즈로 측정불가하게 만든다;
  공통 support 있는 곳에선 within-overlap이 복원한다(DISDE의 시간축·model-transfer 적응)." DISDE/WhyShift 위 위치(REFERENCES §0).

## 14. R2.4 — 도구킷 ground-truth 검증 (covariate×concept 통제 그리드)  [solid] ★D&B 요건
`run_toolkit_validation.py` (sklearn만). covariate 강도(mu, 비-rule dim 이동)와 concept 강도
(theta, rule 회전)를 **독립 통제**한 합성 그리드에서 도구킷(cov_AUC+DISDE퇴화+within-overlap)이
ground truth를 복원하는가 + 실패 모드를 아는가. 4×4 그리드, n=4000:
| mu_cov | cov_AUC | overlap | ESS% | measurable | gap @ θ=0/30/60/90 |
|---|---|---|---|---|---|
| 0.00 | 0.508 | 0.995 | 57.4 | True | −0.001 / +0.091 / +0.271 / **+0.460** |
| 0.70 | 0.808 | 0.678 | **2.33** | True | −0.001 / +0.091 / +0.272 / **+0.462** |
| 1.50 | 0.981 | 0.120 | 0.84 | True | −0.002 / +0.072 / +0.228 / +0.393 |
| 3.00 | 1.000 | 0.002 | 0.05 | **False(전부)** | — (concept 심겨 있어도 **거짓신호 안 냄**) |
- **4개 검증 전부 PASS**: ①복원 Spearman(θ,gap)=**+1.000** ②거짓양성 없음(θ=0서 max|gap|=0.002)
  ③퇴화 단조 Spearman(mu,cov_AUC)=+1.0 / (mu,overlap)=−1.0 ④실패모드: mu=3.0서 4/4 unmeasurable(support 소멸→abstain).
- **★핵심(mu=0.70 행)**: **ESS%=2.33 = DISDE 재가중 사실상 사망**인데 within-overlap gap은 covariate 없는
  경우와 **동일하게** 심은 concept 복원 → **elec2 현상(DISDE 붕괴/within-overlap +0.132 복원)의 합성 증명.**
→ 측정 프레임이 (a) ground truth 복원 (b) DISDE 죽는 곳서도 작동 (c) concept 없으면 0 (d) support 없으면 abstain.
  "측정불가"가 *방법 결함 아닌 support 부재*임을 통제 환경서 입증 — **D&B "도구킷 검증" 요건 충족.**

## 15. R2.5 — Q1 충실성 큰-회전 robustness (헤드라인 논거 robust화)  [solid] ★Q1 헤드라인
`run_q1_faithfulness.py --angle-max 6.283 --basis fourier`. 기존 게이트는 π/2(90°)+trend라 바닥 0.894로
동적범위 좁았음. **2π 전회전 + Fourier 정합**(기저-불일치 교락 회피)으로 재측정:
- 천장(MLP+t) 0.972 / **바닥(shuffle-t) 0.017**(0.894→폭락, 전회전이라 고정방향이 못 추종) / PASS선 0.685.
- **메커니즘 복원 mean 0.988, lower95 0.986, 10/10 PASS** (PASS선 0.685 한참 상회).
→ "메커니즘은 충실"이 좁은 동적범위 위 간신히가 아니라 **[0.017,0.972] 넓은 범위에 걸쳐 robust**.
  헤드라인 논거("충실한 메커니즘조차 못 돕는다"의 충실 절반) 확보. (기존 90° 게이트 PASS도 유효·병기.)

## 16. R2.3 — Cai&Ye 변조 판결 (정의적 confirmed / 경험적 inconclusive)  [정의적 solid]
PREREG_V2 §8 규칙대로 처리.
- **정의적 절반(load-bearing, 확정)**: 변조 `γ(t)·YeoJohnson(x,λ(t))+β(t)`는 **label-free**(y 무관, 코드
  `temporal_modulation.py` 확인) → P(y|x) 착취 *구조적 불가* = **X-side**. 코드/수학 사실이라 재현 강약 무관.
- **경험적 절반(inconclusive)**: 최소 재현(MLP, 단일 lr, 5시드, 무튜닝)의 gain↔cov_AUC.
  - trend 기저: Spearman −0.5(외삽 붕괴, weather −0.092) = 교락된 무효 검정.
  - **fourier 기저(교락 제거)**: 폭락 완화(weather −0.092→−0.046)되나 변조가 대부분 ~null/소-음수
    (Spearman +0.231 약함; elec2 −0.029, insects −0.029, cooking/maps ≈0). → **최소 재현이 그들 보고 이득을
    재현 못 함 = 충실 재현 아님** → 경험적 판정 보류.
  - 사용자 결정(2026-06-15): **정의적 논거로 판결, 경험적 절반은 future work(LAMDA repo gold-standard 재현)**.
→ **판결: 그들 'concept drift' 이득은 정의상 X-side(covariate 적응)** — 우리 Claim A에 포섭. 경험적 확증은 미결(과장 금지).

---

## 17. V3.2 C5 — WhyShift 크로스런 (ACSIncome/folktables, `whyshift_summary.json`)  [solid — 대조 *미지지*, 일반화 WIN]
도구킷(cov-AUC·within-overlap gap·permutation placebo)을 TabReD 밖 ACS 계열에 적용. 5주(CA,TX,NY,FL,PA)×2년(2014/2018).
- **SPATIAL**(CA→TX/NY/FL/PA, 2018): 평균 cov-AUC **0.940**, 평균 gap **−0.0023**(≈0, gap≈placebo), **4/4 measurable**.
- **TEMPORAL**(주별 2014→2018): 평균 cov-AUC **0.676**, 평균 gap **−0.0079**(≈0, gap≈placebo), **5/5 measurable**.
- **판정 = 예측한 "공간=Y|X / 시간=X" 대조 *미지지***: (i) 두 축 다 within-overlap concept ≈ 0(공간이 Y|X-지배 *아님*);
  (ii) 공간 cov-AUC(0.94) > 시간(0.68) — 예측(시간이 더 covariate)과 *정반대*. 리뷰 전에 잡은 또 하나의 overclaim retire.
- **건진 것(WIN)**: ① 도구킷이 ACS로(placebo까지) 일반화 = TabReD 밖 작동 입증. ② folktables ~10 raw 피처 → *둘 다 measurable*
  (overlap 생존) = §6 표현 논점의 직접 외부 증거(측정불가는 *축*이 아니라 *261-피처 엔지니어링* 탓). ③ WhyShift의 Y|X는
  다른(성능-변동) 측도라 우리 conditional 프레임과 직접 모순 아님(명시).
- **향후**: WhyShift가 더 Y|X로 본 task(ACSPublicCoverage·ACSMobility)로 재확인 = future work. PAPER_DRAFT_V3(_KO) §2·§6·§11 반영 완료.

---

## 18. V3.3 위생 — within-overlap gap 5축 경화 (`gap_hygiene_summary.json`, elec2+insects, 15시드)  [solid] ★Claim A 경화
사전등록 결정 규칙(CI>placebo ∧ bias-corr>0.034 노이즈바닥 ∧ BH유의 ∧ metric-invariant)을 결과 보기 *전* 박고 검정.
**판정: elec2·insects 둘 다 CONCEPT(4조건 전부 + 민감도 불변).** 5항목 결과:
- **③ seed-CI(15시드)**: elec2 true +0.146 [.141,.151] / placebo −0.035 / bias-corr **+0.181**;
  insects +0.150 [.147,.152] / −0.017 / **+0.167**. 둘 다 true-CI가 placebo-CI 위, 0 제외.
- **④ ℓ-robustness(메트릭 불변)**: AUC/acc뿐 아니라 Brier(+0.43/+0.12)·log-loss=Bayes-risk(+1.41/+0.16) 전부 양성+CI 0제외,
  예측분포 이동 KL(1.66/0.73). → **concept 판정이 측도에 안 달림**(recalibration-drift 맹점 해소, 한계 a 닫힘).
- **⑤ rolling-origin g(t)** cut {.3–.7}: elec2 모든 cut 양성(평균 +0.122 [.109,.135], gradual, trend ρ=+0.5 n.s.);
  insects 모든 cut 양성(+0.157 [.095,.219]) **but cut에 단조감소(ρ=−1.0)** = drift 앞쪽 집중 → median 단일값이 초기창 과소평가.
  (한계 b 닫힘.)
- **② 민감도 그리드** band×min_per_half×clf{hgb,logreg}: 각 데이터셋 **18/18 셀 concept**, verdict_invariant=True.
- **① BH-FDR**: one-sided paired Wilcoxon(true>placebo) 둘 다 BH-p ≈ 3×10⁻⁵ reject(한계 g 닫힘).
→ Claim A(elec2/insects가 배포표현서 genuine concept 보유)가 seed·측도·시간cut·측정선택·다중성 전 축서 robust.
PAPER_DRAFT_V3(_KO) §5 경화 문단 + 한계 a/b/g 해소 + 향후-작업 2·5 완료 반영.

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
