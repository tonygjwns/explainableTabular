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

## 19. V3.2 C2 — 앵커 외부보정 (`anchors_summary.json`, elec2+insects 2변종, 5시드)  [solid] ★§8 외부보정
같은 temporal split·피처로 knn±t / lgbm(GBDT)±t / no-change(persistence). **두 결론:**
- **① arm이 바닥 통과**: elec2 신경 arm(mlp_t≈0.905 AUC; grid_report) > lgbm 0.887 > **no_change AUC 0.845** →
  Elec2 자기상관 비판(Žliobaitė 2013) 해소(persistence 위). insects-incr arm(≈0.67; §11) ≈ lgbm_t 0.679, no_change 0.163 훨씬 위.
- **② GBDT±t가 메커니즘 독립 재현**: 시간피처 추가가 **incremental서 +0.070**(lgbm 0.609→lgbm_t 0.679) / **abrupt서 −0.192**
  (0.708→0.514; elec2 ≈0) = 우리 신경 시간훅의 "in-dist 도움/외삽 해" 서명을 *비신경 트리*가 그대로 → "시간-as-피처=in-dist
  장치, 외삽 장치 아님" 확증(redundancy 해석 뒷받침).
- 부수: elec2서 knn_t<knn(시간이 knn 해침), insects서 knn_t>knn(도움) — 변종별 시간 효능 차이 일관.
→ §8 외부보정 문단으로 PAPER_DRAFT_V3(_KO) 반영. "정품 TabR"=R0-경화 tabr_t 기판이 충족, 외부 rtdl-TabR은 선택/future.

---

## 20. P0 — 외부 적대리뷰 대응: ess-floor 집행 + ground-truth 재검증  [solid] ★기여 #3(검증된 abstention) 복구
외부 LLM 적대리뷰가 두 결함을 정확히 지적(artifact로 검증함): (반론A) §6 gap이 sparse-MI k에 따라 흔들리고
큰 셀은 cherry-pick / (MISSED-1) **`ess_pct_floor=5.0`이 선언만 되고 배포 `measurable()` 게이트엔 미집행** —
배포 코드가 `n_overlap≥200`만 봐서 ess 0.07~0.97% 셀(예: homecredit sparse@50 gap −0.105, delivery_eta "OUR-FRAME-WINS")이
기권 안 하고 숫자를 뱉음. 두 지적은 **같은 뿌리**.
- **수정(`drift_measure._iw_ess_pct` + 게이트)**: `concept_within_overlap`/`_multi`의 measurable 게이트에
  `ess_pct ≥ 5.0` 추가(모든 caller 일관). near-disjoint 셀은 이제 abstain. None-guard도 hardening(_transfer_gap None 크래시).
- **ground-truth 재검증(로컬, ess 집행 하)**: `toolkit_validation` 4×4 그리드 — mu=0(ess 57%) measurable, recovery
  Spearman(θ,gap)=+1.000, no-false-pos 0.001; **mu=0.70(ess 2.33%)·1.50(0.84%)·3.00(0.05%) 전부 abstain**(이전엔
  mu=0.70이 measurable이었음). **PASS 유지** → 이제 *배포 규칙 = 검증된 규칙* = MISSED-1 해소.
- **§6 엄격화(`run_representation`)**: 다중시드 gap-CI + ess 일관 보고 + **사전등록 데이터셋 verdict**(ess-통과 rep
  전부 '~0'면 all-~0 / 하나라도 floor 넘으면 concept-somewhere / 부호·크기 swing이면 mixed-unstable). §5와 동급 엄격성.
- **서버 재실행 대기**: representation(실 TabReD)·gap_hygiene(elec2/insects가 ess≥5로 여전히 measurable인지=Claim A 생존 확인).
→ 반론A·MISSED-1 닫힘. 남은 적대리뷰 항목 = P1(Elec2 noise/자기상관 분해), P2(D'Amour 2021 등 인용·신규성 재범위화), P3(floor 단위). PLAN_V3 §V3.4.

---

## 21. V3.4 P1 — Elec2/INSECTS concept 분해 (`elec2_decompose_summary.json`, de-time-leaked, 10시드)  [decisive] ★Claim A 운명
적대리뷰 반론 B/C: within-overlap gap이 P(y|x) rule 변화인가, 자기상관(Žliobaitė)·노이즈 드리프트인가. 3프로브로 분해.
- **Elec2 ❌ 탈락**: de-time-leaked +0.073이 ① thinning stride1→5서 +0.073→**+0.033(floor 걸침)** ② **lagged-label
  +0.073→+0.013(below-floor)**, y_{t-1}→y AUC **0.849**(자기상관 극강) ③ noise late +0.046 easier. → 자기상관·노이즈가
  대부분, **genuine concept ≈ 0**. Elec2는 깨끗한 concept anchor에서 제외(reviewer 반론 C 확증).
- **INSECTS ⚠️ 부분생존**: thinning stride1 +0.149/stride5 +0.157 **둘 다 concept(>floor)** = *자기상관 아님*(Elec2와
  결정적 차이, river 설계 concept과 일관). 단 **achievable-acc drift +0.192**(early 0.488→late 0.680)가 +0.149와 엉켜
  깨끗이 분리 안 됨. lagged-label은 multiclass라 미수행. → matched-noise-null로 추가 분리 필요(반론 B 잔여).
- **종합**: 완전 분리된 "깨끗한 concept" 증거 얇음(Elec2 死, INSECTS 노이즈 교락). → **§5/§6의 concept 헤드라인 약화**,
  생성 테스트(V3.5 `correct_assumption`)가 "정직 측정 vs 생성적 프레임 vs 피벗"의 진짜 심판. PAPER 반영은 생성 테스트 후.

## 22. V3.5 — 생성 테스트: recency-적응이 concept 측정된 곳서 이기나 (`correct_assumption_summary.json`)  [긍정·underpowered] ★진단=생성적?
사용자 falsification: 진단이 맞으면 올바른-가정 방법(online/recency 적응)이 concept 측정된 곳서만 이겨야.
- **Spearman(concept_gap, recency_gain) = +0.60** (p=0.28, n=5 measurable reps) — 방향 양수, 검정력 부족.
- **패턴은 예측대로(깨끗한 케이스서)**: **INSECTS concept +0.144 → recency_gain +0.054**(static 0.604→recent50 0.658,
  +5.4점) = concept 큰 곳서 적응 크게 이김; cooking(−0.005)→+0.0005, maps(−0.003)→−0.0006 = concept≈0서 적응≈0.
  elec2 de-tl(+0.078)→+0.001(concept이 자기상관이라 적응 무용 — decompose 일관), weather(+0.019)→−0.034(mixed).
- **판정**: 생성 방향 **확증되나 INSECTS 단일에 의존**(또 n=1). 깨끗한 concept을 가진 유일 데이터셋이 유일하게 적응이
  이기는 곳 = 프레임이 예측하는 모양이지만, 통계력 위해 **concept 데이터셋 확장 필수**(river 합성 drift 스트림 10–15개).
- **함의**: 피벗 신호 아님 — "프레임이 작동, 폭을 키워라"는 신호. gap_hygiene2 확정(elec2 abstain/claim_a False,
  INSECTS measurable/claim_a True, floor 0.041). → **다음 = concept 데이터셋 폭 확장 후 measure+생성 테스트 재실행.**

## 23. V3.5 powered — 생성 테스트 18셋 (river 패널 + INSECTS 4 + cooking/maps, `correct_assumption_full_summary.json`)  [NEGATIVE-단순형]
폭 확장(river SEA/Agrawal/STAGGER/Sine/Hyperplane × {no-drift/abrupt/gradual} + INSECTS 4변종)으로 n=5→18.
- **Spearman(concept, recency_gain) = +0.147 (p=0.56, n=18)** — **단순 생성 법칙 미지지**(검정력 갖추니 평탄).
- **구조적 패턴(우연 아님)**: recency가 *단조/지속* drift서 이김(stagger_abrupt +0.37→+0.35, hyperplane_incr
  +0.16→+0.13, insects_incr +0.14→+0.05, sine_abrupt +0.10→+0.055) / *reoccurring·일부 abrupt*서 짐
  (insects_reoccurring +0.32→**−0.03**, insects_abrupt +0.12→**−0.04**) — 순수 recency의 알려진 실패(옛 concept 재발).
  no-drift 대조 6 + cooking/maps 전부 ~0(정상). → recency_gain은 concept *크기*가 아니라 *구조*(단조 vs 재발) 추종.
- **판정**: 단순 "측정이 적응 이득 예측" = **falsified**. 다섯 번째 깨끗한 양성 dissolution(B/C1/C5/Elec2/생성-단순).
  **피벗-급 신호.** 세 갈래: (a) drift-구조 인지(단조=recency / 재발=retrieval) 정교화 — 더 큰 주장·재축소 위험,
  (b) 누적 증거를 피벗 신호로 수용, (c) 재발-drift=retrieval 우위 가설로 원 method 부활 검증(아이러니: reoccurring이
  retrieval이 recency 이겨야 할 곳). 사용자 판단 대기.

## 24. V3.5-C — retrieval vs recency by drift structure (`retrieval_vs_recency_summary.json`)  [부활 신호·underpowered]
사용자 선택 C: reoccurring drift(옛 concept 재발)는 retrieval의 home field인가? static/recency/retrieval(kNN) 비교.
- **구조별 mean(retr−rec)**: monotonic **−0.098** [−.194,−.002] (recency 우위, 유의) / nodrift −0.005 (~0) /
  **reoccurring +0.137** [−.159,+.434] (양수지만 CI 0 포함, underpowered n=5).
- **메커니즘 입증(깨끗한 케이스)**: stagger_reoccur retr−rec **+0.735**(retrieval 압도), sine_reoccur +0.037,
  **실제 insects_incremental_reoccurring +0.018**(recency −0.051 < retrieval −0.032 = 방향 정확). 합성 검증 +0.42/−0.59.
- **약점**: agrawal_reoccur −0.103 — 단 kNN이 Agrawal 피처공간서 원래 약함(monotonic agrawal도 −0.18) = retrieval
  *방법(kNN)* 한계지 가설 반증 아님. sea는 concept +0.015로 너무 작아 무의미.
- **★POWERED 확정 (n=12 reoccurring, 패널 5→11 확장)**: reoccurring **+0.192 [+0.026, +0.358] = CI가 0 제외!** /
  monotonic −0.098 [−.194,−.002] / nodrift −0.005. **사전등록 패턴 통계 확정**(두 그룹 분리·둘 다 유의). 메커니즘 다양:
  stagger_reoccur +0.735/early +0.747/v2 +0.406, sine_reoccur2 +0.330, **실 INSECTS-reoccurring +0.018**(방향 정확).
  agrawal만 −0.103(kNN 피처공간 약점).
- **판정**: 5번 dissolution 끝 **첫 깨끗한 powered·사전등록 양성.** retrieval=재발 drift의 올바른 도구(recency=단조).
  **caveat(정직)**: 합성 위주(실 reoccurring은 INSECTS 1개, 효과 +0.018로 작음); 스트림이 구성상 재발(정당한 실험 조건이나
  측정가능한 재발-진단기 필요); agrawal=kNN 한계. → **다음 = 학습형 TimeTabR을 reoccurring서 kNN·recency와 비교** —
  plain kNN보다 나으면 원 method가 *재발 niche*서 부활 + Claim B가 "구조 redundant"에서 "구조는 재발 drift서 둘 다 이김"으로 조건부 양전.

## 25. V3.5-C 2단계 — 학습형 retrieval(tabr_t) vs recency vs 파라메트릭, drift 구조별 (`learned_retrieval_summary.json`)  [POWERED 양성] ★Claim B 부활
1단계(kNN)에 이어 학습형 retrieval 구조를 reoccurring 패널서. mlp_t(all)·mlp_t(recent)·tabr_t(all)·time_tabr_t(all), n=3시드.
- **struct−recency 평균**: reoccurring **+0.209 [+0.102, +0.315] (n=11, CI 0 제외)** / monotonic −0.061 [−.149,+.028] /
  nodrift +0.002. **struct_gain(tabr_t−mlp_t)**: reoccurring +0.061 / **monotonic −0.001 ≈ 0** / nodrift ~0.
- **사전등록 패턴 확정**: ① 학습형 retrieval >> recency on reoccurring(+0.209 유의) ② **monotonic서 구조 redundant
  (struct_gain≈0) = 원 Claim B 음성이 *단조 한정*임을 입증.**
- **2번째 예측 적중(학습형 > plain kNN)**: agrawal_reoccur kNN struct−rec −0.103 → **학습형 +0.316**; agrawal_reoccur2
  +0.049 → +0.242. 학습 metric이 kNN 피처공간 약점 메움 = method가 단순 kNN보다 나음 직접 입증. stagger_early full +0.649 최고.
- **판정**: **Claim B 조건부 양성 부활** — "구조는 단조 drift선 redundant(원 음성 범위화), *재발 drift*선 recency·kNN 둘 다
  유의하게 이김(옛 concept 회수)." 측정(concept)+drift구조(재발) 진단이 *언제 retrieval이 옳은지* 예측 = 생성적.
- **★실데이터 확정 (caveat 닫힘)**: `--insects-variants` 포함 재실행 — **실 INSECTS-reoccurring struct−recency = +0.211**
  (recency −0.175, 학습 retrieval +0.036; 1단계 kNN +0.018의 ~10배). reoccurring 집계(n=9) struct−recency **+0.260
  [+0.150, +0.370]**. monotonic INSECTS struct_gain ~0(redundant 유지). **합성+실데이터 모두 확정.** struct_gain(vs
  파라메트릭)은 +0.07로 modest — 강한 승리는 vs recency. → §8(Claim B) "재발 niche 구조 우위"로 재작성 완료(영/한).

## 26. V3.5-C 2차 적대검증 — niche 양성 *기각* (artifact로 자가검증)  [NEGATIVE 확정]
fresh LLM 2차 검토 + 내 artifact 재계산으로 §24-25 헤드라인이 무너짐. **세 정량 주장 모두 사실**(learned_retrieval_summary.json):
- **항등식**: struct−recency = retrieval_struct_gain − recency_gain (9/9행). 즉 "+0.26"은 (구조 +0.07)+(recency −0.18).
  recency가 재발서 깊게 음수인 건 *재발 정의상 구성*(옛 concept 복귀, recent train=틀린 B) → 승리의 ~80%가 동어반복.
- **배포될 full method(time_tabr_t−mlp_t)가 실 INSECTS-reoccurring서 −0.006(음수), 재발 9개 중 5개 음수.** 정당 baseline
  (all-data 파라메트릭) 대비 유일 실데이터서 진다. 나는 struct−recency만 헤드라인 썼고 이 열을 표로 안 냈음(over-sell).
- **struct_gain +0.074는 STAGGER 의존**(−0.029/+0.188/+0.455), STAGGER 제외 평균 +0.009(≈0, 실데이터 포함) = 단일 generator.
- **HARKing**: niche는 §23(n=18) 실패 데이터서 "recency가 재발서 짐" 관측 후 사후 carve-out, 다중성 보정 없음, 합성 의사반복.
- **측정 코어**: 실데이터 0개서 발화(TabReD ≈0/disjoint, WhyShift 0/9, INSECTS 단일+§21 achievable-acc +0.192 미해소).
→ **판정: method 부활 헤드라인 죽음.** 리뷰어 = "D&B reject, 강한 workshop/negative-results". §8/§1/abstract over-claim
  정정 필요. **사용자 결정(2026-06-27): 누적 실험 바탕 새 피벗 모색.** = 5번 dissolution + niche 기각의 종합.

## 27. V4.0 pre-test — disjointness는 harmful인가 benign artifact인가 (`positivity_regime_summary.json`)  [benign 확정]
사용자 반론("전처리 artifact 아니냐")을 kill-test로. TabReD 8개+elec2/insects, model-light(cov-AUC·label-relevant
overlap·split-conformal late coverage·perf_drop).
- **Spearman(cov_auc_raw, perf_drop)=−0.106 (p=.77)** — raw disjointness가 예측 harm 예측 못 함. disjoint 7개 중
  harmful 2(elec2 자기상관·非TabReD / ecom 약함).
- **간판 disjoint가 benign**: homecredit cov1.0 → drop 0.009/conf 0.855; homesite cov1.0 → 0.008/0.904; delivery 0.997 → 0.065.
  covLR(예측방향 overlap) homecredit 0.60·weather 0.61·delivery 0.54 = **disjoint는 nuisance 방향뿐, 예측방향 겹침.**
- **판정: 사전등록 kill 충족. 261-dim 안 겹침은 benign 전처리 artifact**(cov-AUC 1.0이어도 예측·conformal 전이 멀쩡).
  사용자 직감 적중. "harmful positivity-failure regime" 척추 사망. → 정직한 측정/datasheet(제안1·3) 또는 도메인 피벗.
- 규율: 7번째 dissolution을 commit 전 1.5일 사전등록 실험으로 차단(2차 리뷰어 HARKing 경고 준수).

## 28. V4-B — 적대적 도메인 probe (fraud×2 + malware): 8도메인 전부 음성  [broad negative 확정]
도구킷(검증됨, 도메인불변)을 공격자가 규칙을 실제 바꾸는 도메인에. 사전등록: ≥1개 REAL-HARMFUL-CONCEPT면 양성.
- **BAF (NeurIPS'22 fraud)**: gap −0.013(음수), drop −0.017(late 더 쉬움) → **concept~0**.
- **IEEE-CIS fraud**: ess 3.07%(<5%, near-disjoint=TabReD 패턴) → **unmeasurable**.
- **EMBER malware**(626 피처 직접파싱, 60만): gap +0.026(floor 0.041 미달), drop +0.030(모델 거의 안 썩음),
  cfU +0.097(약한 conformal harm이나 gap≈0) → **concept~0**. (단 우리 split=balanced median, TESSERACT 배포
  프로토콜 아님 — 충실 temporal-holdout은 미수행; 단 drop 작아 toolkit-blunt보다 "이 split선 decay 작음"에 가까움.)
- **판정: 0/3 적대적 = 0/8 전체 도메인 REAL-HARMFUL-CONCEPT.** 산업+적대적 표 시간데이터 전반에서 측정가능·착취가능·
  유해한 concept drift **부재/측정불가**. 분야의 concept-drift 프레이밍이 실데이터와 어긋남을 8도메인으로 입증.
  → **사냥 종료. 산출물 = 검증된 도구킷 + 광역 negative(measurement/negative-results/D&B-tools).**

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

## 29. deployment-decay v3 — 사전등록 실행 완결, 최종 identifiability map (2026-07-04~05) ★현행 헤드라인

> §28("사냥 종료, 광역 negative")과 §3b의 "TabReD 8/8 stale≤0"을 **대체**한다. 전 과정 사전등록
> (`PREREG_DEPLOYMENT_V2.md` §0~10, 규칙→예측→실행→판독이 커밋 타임라인으로 증빙), 결과 아티팩트
> `prereg_results/`, 계기 = `run_deployment_decay.py` v3 (denoised staleness + noise gate +
> group-aware D + 학습가능성-게이트 주입; 합성 배터리 14/14 PASS 후 실행).

**경위**: 외부 종합 감사(AUDIT_FINAL_2026-07-04.md)가 v2를 3중 기각 — F1(라벨노이즈 감쇠가 고정
규칙에서 +0.021 CONCEPT 오발 = sberbank +0.024와 동크기), F2(D 게이트가 중복/코호트 메모리제이션으로
1.000 포화), F3(concept/covariate 분리가 가설클래스 상대적: kNN +0.098, linear +0.026 오발; HGB/RF는
정상). v3 = 각 킬의 실행-검증된 수리. 원리 교훈: **raw staleness 단독으로는 CONCEPT를 발급할 수 없다**
(denoised arm + noise gate 필수).

**최종 지도 (HGB, 탐색 시드 0–9 ↔ 확증 시드 100–109 **10/10 판정 일치**, unstable 0)**:
- **insects = DEPLOYMENT-CONCEPT** (raw +0.129~+0.135, den +0.145~+0.152 — denoised가 더 강함 = 규칙
  변화의 서명; 주입 회복). 유일 양성.
- **sberbank = NOISE-DRIFT-CONFOUNDED** — v2의 유일 산업 양성(+0.024)이 **계기에 의해 진단됨**: raw는
  비트-동일 재현(+0.023900573591226625), old 윈도우 노이즈 프록시 2.1~2.9×, **denoised 전 K에서 유의
  음수**(옛 라벨의 노이즈를 제거하면 old 데이터가 도움 = 규칙 불변). 9번째 dissolution이자 최초의
  계기-내 진단. K∈{5,8,10,12,20} 전부 비-CONCEPT.
- cooking/delivery = INJECTION-RECOVERED(검증된 무-concept), maps = NO-STRONG-CONCEPT,
  elec2 = UNIDENT(**earned**), ecom/homecredit/weather = UNIDENT(**vacuous** — 주입 학습불능; v2의
  'earned'는 과대주장이었음), homesite = INERT(인증 불안정).
- **집계(§5 고정문): tree-ensemble 기준 산업 mean-rule drift = 0/8.**

**model-class 패널**: HGB↔RF 10/10 작동적 일치(지도는 클래스 내 견고). 카나리아 flip 실증 —
**linear 프로브는 elec2를 CONCEPT로 읽음**(den +0.033, 주입회복 +0.19): "모니터의 판정은 프로브
모델의 성질" 실데이터 예시.

**앵커(계기 타당성)**: river 단조·단발전환 6/6 + insects 단조 3/3 = **CONCEPT 발화**(denoised≥raw
+ 주입회복). 왕복/재발 구조(insects abrupt·reoccurring, river reoccur)는 침묵 — **음의 recency_gain
(−0.058)이 재발 지문**(직전 윈도우가 window0보다 못함 = 옛 레짐 회귀). 프로파일: "old 라벨이 *현재*
레짐과 모순일 때만 발화" = 배포-해악 렌즈의 옳은 의미론, 3중 독립 확인. **EMBER = NO-STRONG-CONCEPT**
(stale −0.012 = old가 도움, delta 0.0014): 말웨어 열화는 라벨 부패가 아니라 커버리지(새 패밀리) —
TESSERACT drift와 모순 아님. nodrift 오탐 체크에서 SUBFLOOR 2/5 → **SUBFLOOR 대역 = 무증거로 재캘리브레이션**(§9).

**논문 골격(집필 개시)**: identifiability map + 수리된 계기(denoised staleness의 유효 envelope 포함)
+ 구조별 감도 프로파일 + 진단된 sberbank + EMBER null. 타깃 TMLR(즉시)/D&B(차기). 상세 = PREREG §7~10.

## 30. 리버탈 런: δ(N) 스윕 + 두-렌즈 head-to-head (2026-07-18, git f6e65b6) — 부록 B로 논문 반영

**성격**: 사전등록 배터리 *이후*의 post-hoc 강건성 점검 (계기·캐스케이드·임계값 불변, 리뷰 라운드
2 P2 큐). raw = `summary_20260718T{040332,052219,055856}_f6e65b6.json` + `representation_summary2.json`
+ `whyshift_summary2.json` (루트, 서버 push 인증 불가로 수동 환류).

**δ(N) (null 셀 3개, cap ∈ {1500, 24000, 96000}, 10시드)**: 판정 전 셀·전 N 불변, floor(0.02)로
접근하는 dose-response 없음. 실현 N은 윈도우 기하가 캡: homecredit ≈17.2k(96k cap도 동일),
weather 24k(cap-bound), maps ≈14.3k.
- homecredit: raw 전 N에서 유의 음수(−0.015→−0.010, old가 도움); denoised는 floor의 ¼(≈+0.005)에서
  평탄, 상단서 축소(+0.0065→+0.0057→+0.0048). 판정 UNIDENT(vacuous, inj 학습불능 재확인) 불변.
- weather: denoised −0.022(1.5k)→−0.011(6k)→−0.002(24k) — 아래에서 0으로 수렴. 불변.
- maps: denoised −0.017→−0.010→−0.008 안정 음수. NO-STRONG-CONCEPT 불변.

**두-렌즈 head-to-head (overlap 성립 셀 2개)**: 일치.
- maps_routing: within-overlap(P0 ess-게이트) 5/5 시드 measurable, gap −0.003 [−.004,−.002] (full
  986피처, ESS 93%) — sparse MI@{5,10,20,50}에서도 −0.005 내외 전부 ~0. 배포 렌즈 NO-STRONG-CONCEPT와 일치.
- ACS CA 2014→2018: cov_AUC 0.680, measurable, gap −0.0093 ≈ placebo −0.0083 → placebo 위 Y|X 이동
  없음. 배포 렌즈(§6 브리지 null)와 일치. **"overlap이 성립하는 곳에선 두 계기가 일치; 본문이 기록한
  불일치는 정확히 overlap이 무너지는 곳에서 발생"** = WhyShift 델타의 실증 보강.

**반영**: PAPER_DRAFT_V4(.md/_KO/§tex) 부록 B 신설 + §6 ACS 문단 포인터 + §7 한계(1b) δ(N) 1차
실행 반영. 잔여 리버탈 카드 = FT-Transformer급 프로브(선택, GPU).

---

## 31. 원고 수치 전수 감사 (2026-08-03, 로컬) — 오탈 2건 수정, 출처 공백 2건 기록

`scripts/audit_paper_numbers.py`를 확장해 실행. 이전 버전은 `rows` 스키마 파일 164개만 읽어
**행 스키마 이전 아티팩트 35개(배터리·모델클래스 행렬·회귀 컨트롤·whyshift·gap hygiene)와
로그 27개를 통째로 건너뛰고 있었다** — 그래서 근거가 있는 수치가 대량으로 unmatched로 나왔다.
확장 3건: ①임의 깊이 재귀 walk(계기 이름을 가진 leaf만) ②`logs/**/*.log`의 `key=value`·CI 수확
③**값 일치만으로 MATCH를 주지 않음** — 아티팩트 라벨(데이터셋·배터리 행 이름)이 그 수치를
인용한 문장에도 나타날 때만 CONFIRMED, 아니면 value-only. (4dp에서 2천 개 값이면 우연 일치가
실재한다.) Windows cp949 stdout·3dp 인용이 tol 경계에 정확히 앉는 문제도 고침.

**결과**: 세 파일(EN/KO/tex) 328·328·303 수치 중 unmatched 25 → **3**, cross-file 불일치 0.

**수정한 오탈 2건** (둘 다 세 파일 동시, 아티팩트가 정본):
- `hyperplane_incremental` **+0.113 → +0.112** (`summary_20260704T172128_5f3217d.json`
  staleness_harm = 0.11249). 같은 문장의 나머지 river 수치 4개는 전부 정확 — 반올림 규약은
  최근접이 맞고 이 항목만 어긋났다.
- 작은 old 윈도우(600행/fold 300) 미탐 없음 **+0.430 → +0.429**
  (`exp-estimator/battery_results.json` `reg_concept__oldcap600` denoised = 0.42946).
  같은 문장의 오탐 없음 +0.004 = `reg_early_noisy__oldcap600` 0.004269 ✓.

**확인된 것 (수정 불요)**: weather_fs subpop@lo −0.050/R² 0.560 = 실측 −0.04947/0.5602이고
확증 시드 −0.04246/0.5233로 재현 ✓ / 같은 윈도우 lowvar +0.128(0.12840)·interaction
+0.046(0.04617) ✓ / **ACS PA 양성률 0.234→0.268→0.293→0.305→0.306, TX 0.183–0.185 =
`logs/acsprep_{PA,TX}.log` 원문과 일치** ✓ (JSON이 아니라 로그가 유일 출처였다).

**출처 공백 2건 (주장은 유효, 재현 경로가 얇다)**:
- envelope 표의 `synth_reg_stable` 절대 수준 −0.1343 → −0.1222: 차이인 size term +0.01212는
  `exp-reg/reg_controls_results.json` staleness_harm −0.01212와 정확히 일치하나, **절대 수준은
  어떤 커밋된 아티팩트에도 없다**(그 런이 per-seed 수준을 저장하지 않음).
- 부록 C 학습가능성 게이트 행의 junk 0.506 / 학습가능 0.964 → 회복 +0.195: 출처가
  `AUDIT_FINAL_2026-07-04.md` §C1(실행된 감사)이고 **JSON 아티팩트가 없다.**
→ 재현 패키지를 만들 때 이 둘은 재실행해서 아티팩트를 남기는 게 맞다. 지금 수치를 바꾸지 않는다.

---

## 32. day-4 큐 판독 (2026-08-02 실행 / 2026-08-03 판독, 커밋 a8f1549) — 사전등록 = PREREG §19, ACS 확장 §12

**게이트**: 배터리 14/14 PASS이고 map-env 참조와 **비트 동일**(전 필드·전 판정). 신규 플래그
2종(`--metric`·`--mi-k`)이 기본 경로를 안 건드렸다 — 되돌림 불필요.

**E1 (proper score)**. ACS: PA denoised가 AUC −0.009 → **brier +0.002 → logloss +0.021**로
부호를 뒤집는다. TX(대조)는 −0.011 → −0.002 → **+0.006**. logloss 대비 3.5×, 방향 정합.
실행 전 등록한 기제(AUC는 순위 기반)가 **다른 렌즈(§11)뿐 아니라 계기 자신에서도** 확인됐다.
단 ①그 팔엔 인증서가 없고(주입 learn −0.185) ②대조도 양수라 비만 읽을 수 있으며 ③0.02는
AUC 단위라 "통과"가 판정이 아니다. 예측 3건 중 ①부분 성립 ②성립 ③**반증**(TX가 0을 넘음).
산업 3셀: 지표를 바꿔도 **vacuity는 안 풀린다** — ecom·homecredit 주입이 네 지표 전부 학습불능.

**E2 (프레임 head-to-head)**. 규칙 셀 +0.4345/+0.8207 vs 고정-규칙 노이즈 셀 +0.0576/+0.0615,
진짜 null −0.0208. → **강한 주장은 반증**(크기로는 7.5–14× 갈린다). 남는 주장: **부호로는 안
갈린다** — 노이즈 감쇠가 진짜 규칙 변화와 같은 부호를 내고, 임계값 판독은 둘을 같은 사건으로
파일링하는데 함의된 조치는 한쪽에서만 옳다. cov_AUC 0.500·ESS 71.3%로 **프레임에 유리한
조건**에서 나온 결과다. 예측 2건 모두 성립.

**E3 (mi-k 내부 D 사다리)**. D는 k에 단조지만 **예측과 부호 반대로 상승**(delivery
0.521→0.801, homecredit 0.721→0.912). 회복은 **측정 불가** — 표현을 좁히면 셀이 D\*=0.96
아래로 내려가 주입 컨트롤이 라우팅되지 않는다. 즉 **D를 움직이는 손잡이가 검정력을 재는
영역 밖으로 셀을 밀어낸다**. §6.2의 ρ=−0.47은 상관으로 남는다. 부수: 같은 셀에서 표현만
바꿔도 denoised가 이동하고(homecredit k=5에서 **+0.013 [+0.010,+0.016]**, D=0.721 = 식별가능
영역), 전 표현에서 인증 불가였던 셀이 여기선 판독된다 → **표현이 식별가능성과 부호를 동시에
정한다**(진단 전용 스코프).

**E4 (앵커×패밀리)**. 12/12 회복·학습가능. **subpop이 저-D 앵커(D≈0.49)에선 회복한다**
(+0.119/+0.079/+0.076) — `weather_fullspan`(고-D)에서 학습가능한데도 −0.050 미회복이던 것과
대비. → **subpop 맹점 = 패밀리 × 기하**이고, subpop은 네 패밀리 중 회복이 일관되게 최소라
**D가 오를 때 가장 먼저 떨어지는 패밀리**다. §17·§18 패밀리-상대성 진술에 "고-분리도 산업
기하에서"라는 한정어가 붙는다.

**원고 반영**: V5 §3.4(E2 신설 소절)·§6.2(E3)·§6.3(E4 한정어)·§6.4(E1 표+수리 방향) —
`PAPER_DRAFT_V5_SECTIONS.md`. 수치 131개 전부 아티팩트 대조 통과.

---

## 33. 고전 탐지기 배터리 (선택 카드 ①) 판독 (2026-08-03) — 사전등록 = PREREG §20

**한 줄**: ADWIN·KSWIN·DDM·PageHinkley를 14-셀 배터리의 prequential 오차 스트림에 통과시킨 결과,
**발화율이 진실 라벨을 전혀 설명하지 못한다.** §2가 논증하던 type-blindness가 실측이 됐다.

- 예측 3건: P1 **성립**(태스크 내에서만 — 예측 문구가 태스크를 안 밝힌 건 설계 결함) /
  P2 **반증**(순수 null인 `reg_stable`이 KSWIN 1.333·PH 0.583으로 규칙 셀과 동급, 고정 규칙
  `reg_late_noisy`의 PH 3.444가 패널 최대) / P3 **측정 불가**(DDM은 binclass 전용인데 노이즈 감쇠
  셀 둘이 모두 회귀 — 이 배터리로는 단측성을 못 시험한다).
- 예측 밖 관측 2건: ①발화율이 진실이 아니라 **태스크**로 갈린다 ②이진 팔에서 **미탐과 오탐이 동시**
  — `concept_noise`(진짜 규칙 이동)가 `stable`과 정확히 같은 발화(0.083/0/0/0.083)이고
  `covariate`(고정 규칙)는 DDM 0.139·PH 0.278로 발화한다.
- **교락**: 회귀 팔은 증분 LinearRegression의 학습 곡선 전이와 분리되지 않는다. 따라서 "진실을
  못 가린다"까지만 쓰고 "노이즈 감쇠에 특이적으로 발화한다"는 쓰지 않는다. 가장 깨끗한 증거는
  학습기가 같은 **이진 팔의 미탐/오탐 쌍**이다.
- ⚠ 아티팩트(`results/phase1/classic_detectors/classic_detectors.json`) **서버 push 대기**. 수치는
  콘솔 판독이며, 대조 전에는 원고에 넣지 않는다.
