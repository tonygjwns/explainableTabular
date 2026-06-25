# PLAN_V3 — 재건 계획 (7-에이전트 red-team 수렴 구제 반영, 2026-06-17)

> **피벗 이력**: V1(방법: 시간-인덱싱 메모리/검색 — 성능 음성) → V2(측정: covariate 지배→concept 측정불가) →
> **V3(재형식화 + 정직-범위화 측정)**. red-team 종합 = RED_TEAM.md. 현행 계획 = 이 문서(PLAN_V2 대체).
> **규율: nucleus를 죽일 수 있는 *결정적 게이트*를 먼저(V3.0). 형식적 재작성(V3.1)은 게이트 생존 후.**

## 0. 살릴 nucleus (red-team 생존분)
> "*배포된 피처 표현*에서, covariate가 공통 support를 무너뜨려 concept이 표준 overlap/reweighting 렌즈로
> 접근 불가 — 즉 **피처 엔지니어링 자체가 concept을 측정불가하게 만든다** — 그리고 우리는 식별 경계를
> 명시하고 그 영역에서 환각 대신 *기권하는*, 적대적으로 검증된 진단을 제공한다."
- 거짓 보편형("tabular temporal data에서 concept 측정불가") **폐기**. 절차/표현 상대성 *전면 인정*.
- 형식 척추 = DISDE term-(ii) 시간축 estimand + positivity 정리(D2). 모든 claim "deployed representation, overlap lens" 한정(D1).

## 0.1 게이트가 다 통과 못 하면 (정직한 사전 분기)
- G1(placebo) 실패 → +0.132/+0.144가 home-field/noise = concept 아님 → "측정 도구는 confounded" 자체가 발견(축소 워크숍).
- G2(E1) 시간-프록시 제거 후 측정가능+concept≈0 → 스토리 = "measurable AND absent"(unmeasurable 아님) — 여전히 논문, 단 다른 헤드라인.
- G3(toolkit) 거짓양성/음성 특성화 → abstention 청구를 그 실패모드로 한정.
→ **어느 결과든 정직히 보고하면 산출 있음. 침묵/과장만 패.**

---

## ★ V3.0 게이트 판정 (2026-06-17, 실데이터 완료) — nucleus 생존(재범위화·강화)
- **G1 ✅ 통과**: elec2 true +0.146[+.140,+.151] vs placebo −0.035 / insects +0.150[+.147,+.152] vs −0.017 →
  양성이 placebo 한참 위 = **진짜 concept(home-field 아님)**. cooking/maps corr≈0(concept 없음 확정). `gap_controls_summary.json`.
- **G2 ✅ 통과(재범위화)**: disjoint TabReD 4/5(sberbank/homecredit/homesite/weather)가 희소표현서 **측정가능+concept≈0**,
  ecom만 진짜 disjoint. elec2/insects concept은 de-time-leak서도 생존(+0.078/+0.108). → 3분법은 표현 함수 →
  "배포된 표현" 한정(강제). **NEW 스토리(더 강함)**: "피처 엔지니어링이 concept을 *점검 불가*하게 만든다; 점검하면
  genuinely ≈0 or disjoint; 진짜 concept(elec2/insects)은 placebo+표현 양쪽 생존(삼중 확인)." `representation_summary.json`.
- **G3 ✅ 견고**(A2 노이즈 +0.034 caveat만), **G4 ✅**(신규성=측정불가성+abstention로 좁힘).
→ **판정: §0.1 분기 불필요. nucleus 생존. V3.1 진행.** §13 표 교체 필요(disjoint→배포표현-점검불가/de-time-leak-concept≈0/ecom-disjoint).

## V3.0 — 결정적 게이트 (model-light, sklearn, 수일; 재작성 *전*)

### G1. within-overlap gap 통제 — concept인가 아티팩트인가 [R1, D3-A2] ★최우선
**합성 부분 완료(2026-06-17, 로컬)**: 양성대조 true +0.985/placebo +0.005(메커니즘 작동), prior null +0.001(무시가능),
**noise_drift null +0.034**(A2 교락 실재하나 red-team 추정 +0.15의 ~1/4 — elec2 +0.132 못 만듦, 단 "+0.132=concept+≤~0.03
노이즈" caveat 필요). `gap_controls_synth_summary.json`. **남은 결정 게이트 = 실데이터 placebo(서버)**: elec2/insects
true gap이 자기 placebo CI 위인가.
신규 `run_gap_controls.py`(기존 `concept_within_overlap` 위):
- **placebo**: overlap 밴드 내 시간라벨 셔플(early=late) → gap **반드시 ≈0**. 안 되면 home-field 확정.
- **prior-shift null**: P(y) 바꾸고 P(y|x) 고정 → gap ≈0.
- **label-noise-drift null**: 경계 고정, early 고노이즈/late 저노이즈 → gap ≈0이어야(A2: 현재 +0.15 거짓양성 위험).
- **N-equalize**: early를 late_tr 크기로 subsample, 다중 resample 평균.
- **permutation null + seed-CI**: 모든 실데이터 gap(elec2/insects/cooking/maps)에 부트스트랩 CI + null p값.
- 게이트: elec2 +0.132 / insects +0.144가 **placebo·3 null 통과 + CI가 0 위** → concept 확정. 아니면 → 0.1 분기.

### G2. 표현 의존성 — disjoint가 데이터인가 피처 엔지니어링인가 [D1] ★최우선
**E4 데모 완료(로컬)**: measurable(cov_AUC .52/overlap .997/gap +0.984)에 시간-프록시 1개 추가 → cov_AUC .995/
overlap .026/**측정불가**. *같은 데이터·concept인데 피처 1개로 뒤집힘* = 3분법은 표현 함수(D1 확증). `representation_synth_summary.json`.
**남은 = 실데이터 sweep(서버)**: 5 disjoint TabReD서 시간-프록시 제거/희소표현 후 overlap 살아나나.
신규 `run_representation.py`:
- 5 disjoint TabReD에서 **시간-프록시 제거**(|corr(feat,t)| 상위 제거) + **희소표현**(MI(feat,y) 상위 k∈{5,10,20,50})로
  cov_AUC·overlap_mass·concept_within_overlap 재계산.
- **양방향 데모(E4)**: measurable(cooking/maps/insects)에 시간-프록시 *추가* → 판정 악화 보이기.
- 게이트: 어느 disjoint든 overlap 살아나고 measurable 되면 → Claim A "deployed representation"로 재범위화(생존).
  안 살아나면 → "시간-프록시 제거 후에도 측정불가"라는 *더 강한* 결과.

### G3. 도구킷 적대적 검증 — "검증된 abstention"을 정직하게 [D3]
**완료(로컬, `toolkit_adversarial_summary.json`)**: 도구킷이 red-team 우려보다 견고 — A1 부분영역(16%) *잡아냄*(gap +0.156),
A2b covariate∥규칙 거짓양성 없음(−0.004), A3 abstain이 착취가능 concept 안 숨김(time-aware 이득 +0.0008), REP 가역회전 *불변*
(+0.355→+0.332). **유일 약점 = A2 노이즈 drift 거짓양성 +0.034**(G1과 일치, modest). ★정밀화: 표현 의존성은 *가역 좌표변환*이
아니라 *시간-프록시 피처 추가*의 문제(REP 불변 + E4 변함) = D1의 더 방어 가능한 버전.
`run_toolkit_validation.py`에 적대적 셀 추가:
- A1: 부분영역 규칙뒤집힘(전역 AUC가 평균낼) → 거짓음성 측정.
- A2: 시간가변 Bayes 노이즈 / prior shift, 경계 고정 → 거짓양성 측정.
- A3: disjoint-but-smooth-trajectory, drift-prior 오라클로 채점 → "측정불가≠착취불가" 시연.
- 비직교 covariate·비선형 규칙·표현 회전(E3) 셀.
- 게이트: 실패모드 *특성화*. abstention 청구를 통과 영역으로 한정.

### G4. 웹 선점/반박 체크 [R9, R10] — 내가 즉시 (웹)
- R9: TabReD류서 early/late 도메인분류기 완전분리 보고한 선행(adversarial validation/classifier-two-sample-test)?
  → 있으면 Claim A 경험 코어 선점. + cov_shift_auc를 그 계보로 인용.
- R10: WhyShift에 temporal config 존재 + Y|X-지배? → "공간=Y|X/시간=X" 대조 반박 위험. + 도구킷을 WhyShift에 돌릴지 결정.

---

## V3.1 — 재형식화 (게이트 생존 시; writing + 경량 실험)

### F1. estimand + 정리 [D2] ★형식 척추
- θ_concept := DISDE term-(ii)를 시간축에 정의(공유분포 S 위 conditional 변화). transfer gap을 "overlap 위 θ의
  추정량(realizability 가정 명시), 밖에선 비식별→abstain"으로 강등.
- **positivity/overlap 명제** 1단락(겹침 밖 identified set=전체 simplex, prior 없으면 비식별; 인과추론 positivity 인용).
- §9 자기모순 수정: "no architecture recovers" → "prior 없이 비식별; 올바른 prior는 자기 가정 안서 식별(TabPFN)" → 설계공간={drift-prior, online}.

### F2. 메트릭·형태 robustness [D2 ②③④]
- AUC 외 **Brier/Bayes-risk/KL 컬럼** 추가, 판정 불변 여부 보고(recalibration drift 맹목 해소).
- median 이분화 → **rolling-origin gap 궤적 g(t)**(다중 cut-point) + 형태통계(gradual/abrupt) + cut-간 CI(=replication, R7).
- 밴드 내 prior 보고/균등화.

### F3. 전면 재범위화 [D1, R5]
- 모든 "unmeasurable/unidentifiable/no architecture" → "deployed representation서 overlap/reweighting 렌즈로 비식별".
- "covariate 지배"를 절차-상대로 명시 + 민감도 그리드(band/classifier/floor)로 *버킷* 불변 보이기(R3, R11).

---

## V3.2 — 연결 + 폭

### C1. A↔B 연결 실험 [R2, D4] ★퍼즐 설명력 입증
- TabReD 방법 랭킹을 우리 렌즈에 통과: 딥-vs-트리 margin이 cov_AUC·overlap과 상관되는가 *보임*.
- **`margin ~ cov_AUC + budget + seed_var` 회귀**: 튜닝예산·시드분산 통제 후 cov_AUC가 유의해야 thesis 증분설명력 입증.
  안 되면 thesis가 퍼즐에 더하는 게 없음을 인정(Claim A를 측정-결과로만, 퍼즐 설명 주장 철회).

### C2. Claim B 재범위화 + 앵커 [R7, R8, D4]  — ✅ 완료 (2026-06-23, `anchors_summary.json`)
- ✅ **재범위화(§8) 완료**: "인스턴스 검색이 시간-피처와 in-dist redundant, 2 concept 데이터서" = *일반 법칙 아님*
  명시됨; "no concept⇒구조 무용" 자기모순 삭제됨(§8 "*not* a general law that time-aware structure cannot help").
  rolling-origin 다중 cut = V3.3 위생 §5서 산출(elec2 gradual / insects 단조감소).
- ✅ **앵커(R8) 완료**(`run_anchors.py` 다중시드+루트 summary): no-change·GBDT(lgbm)±t·knn±t를 elec2 + insects
  2변종(incremental_balanced / incremental_abrupt_balanced)서. **결과 2개**: ① 신경 arm이 floor 통과
  (elec2 mlp_t≈0.905 > lgbm 0.887 > no_change 0.845 → Žliobaitė 자기상관 비판 해소); ② **GBDT±t가 메커니즘 독립
  재현**(시간피처 incremental +0.070 / abrupt −0.192 = 신경 훅의 in-dist도움/외삽해 서명). RESULTS §19, §8 외부보정 반영.
  **"정품 TabR"** = R0-경화 `tabr_t` 기판이 충족 — 외부 rtdl-TabR은 [선택/future].

### C3. §9 Cai&Ye [R3]  — 🔄 faithful 프로토콜 코드 완료, 서버 실행 대기 (2026-06-23)
- ✅ 글쓰기: §9 이미 *용어* 지적으로 강등 + "cannot exploit/subsumed" 삭제 + ρ=−0.50 정직 보고됨.
- 🔄 **faithful 재현 = `run_modulation_adjudication.py --lr-grid` 추가(양 arm 각자 val-튜닝 lr=그들 튜닝
  프로토콜)**. 최소 재현(고정 lr)이 그들 이득 재현 못 한 게 "튜닝 안 해서"라는 반박을 봉쇄: 공정 튜닝 후에도
  modulation이 (a) 안 이기면 adjudication이 strawman 아님 / (b) 이기면 cross-dataset Spearman(gain~cov vs concept)이
  그 이득이 X-side임을 판정. **결정 규칙**: 어느 쪽이든 §9 결론(그들 'concept'=X-side, 정의적 확정) 강화.
  ```
  python scripts/run_modulation_adjudication.py --config configs/phase1.yaml --all --elec2 --insects \
    --mod-basis fourier --lr-grid 2e-3 1e-3 5e-4 2e-4 --n-seeds 10
  # → results/phase1/modulation_adj/summary_fourier_tuned.json 환류. §9에 "튜닝 후에도 …" 문장 반영.
  ```
- (선택, gold) LAMDA repo(github.com/LAMDA-Tabular/Tabular-Temporal-Modulation) 직접 실행은 별도 future.

### C4. Q1 고아 수정 [R6]
- §8을 "오배선 아님"으로 축소. OR time-TabR(실제 쓴 메커니즘)이 실데이터서 알려진 drift 복원하는 진단 추가.

### C5. 폭 [D1, R10, R7]  — ✅ 완료(2026-06-23, ACSIncome), 대조 *미지지*·일반화 WIN
- WhyShift 데이터에 도구킷 크로스런(대조 입증 or 도메인효과 인정). 데이터셋 10→가능한 만큼 확장.
- **결과(`whyshift_summary.json`)**: SPATIAL cov-AUC 0.94/gap≈0(4/4 measurable), TEMPORAL cov-AUC 0.68/gap≈0
  (5/5 measurable). **예측한 공간=Y|X/시간=X 대조 *미지지***(둘 다 concept≈0, 공간이 *더* covariate). → 또 하나
  overclaim retire. **WIN**: 도구킷 ACS 일반화(placebo까지) + folktables ~10피처서 둘 다 measurable = §6 표현논점
  외부 증거(측정불가=피처엔지니어링 탓). RESULTS §17, PAPER_DRAFT_V3(_KO) §2·§6·§11 반영. 향후=pubcov/mobility.

---

## V3.3 — 재작성 + 위생
- PAPER_DRAFT 재작성(nucleus 헤드라인, 기여 6개→정직 재구성, DISDE 전면 인용, estimand/정리 절 신설).
- BH-FDR 전 contrast family 적용(R12). Claim A 임계 사전등록 or 민감도(R11). 모든 실데이터 수치 seed-CI.

### 🔧 위생 5항목 통합 = `run_gap_hygiene.py` — ✅ 완료(2026-06-23, `gap_hygiene_summary.json`)
**판정: elec2·insects 둘 다 CLAIM-A CONCEPT(4조건 전부 + 민감도 18/18 셀 불변).** seed-CI(bias-corr +0.181/+0.167),
ℓ-robustness(Brier·log-loss·KL 전부 양성=메트릭 불변, 한계 a 닫힘), rolling-origin(모든 cut 양성; insects는 cut에
단조감소=drift 앞쪽집중, 한계 b 닫힘), BH-FDR(둘 다 reject, 한계 g 닫힘). RESULTS §18, PAPER_DRAFT_V3(_KO) §5 반영.

한 스크립트가 5항목을 같은 overlap-band 머신·같은 실데이터(elec2/insects)로 한 번에 산출:
③ **seed-CI**(N=15, true·placebo 메트릭별) / ④ **ℓ-robustness**(AUC/acc + **Brier·log-loss(=Bayes-risk)
·KL** — concept *판정*이 메트릭에 안 달림 입증; D2 ②③) / ⑤ **rolling-origin g(t)**(median 단일 cut → 다중
cut 그리드 {.3,.4,.5,.6,.7} 궤적 + 형태통계(trend ρ / abrupt max-jump) + cross-cut CI=replication, R7) /
① **BH-FDR**(데이터셋당 one-sided paired Wilcoxon `true>placebo`를 BH로 묶음, R12) / ② **민감도 그리드**
(band × min_per_half × classifier{hgb,logreg}서 verdict 불변 확인, R11).
- 신규 코어: `drift_measure.concept_within_overlap_multi`(다중-메트릭 + clf 선택; 기존 `concept_within_overlap`
  불변 → gap_controls/representation/whyshift 안 깨짐). `_classif_gaps`/`_transfer_gap_multi`.
- **★사전등록(PRE-REG) 결정 규칙 — 결과 보기 *전* 여기 박음(②의 정직성 핵심)**: 데이터셋이
  "배포된 표현서 genuine concept"이려면 4조건 동시 충족 — (a) true-gap 95%CI 하한 > placebo 95%CI 상한,
  (b) bias-corrected gap(true−placebo) > **floor=0.034**(=G1 noise-drift null → Bayes-noise 교락 위로),
  (c) BH-adjusted one-sided p < 0.05, (d) metric-invariant(auc/acc·brier·logloss 전부 late>early). 미충족 → §0.1 분기.
  민감도 그리드는 (a)+(b)가 band/min_per_half/clf 선택에 불변임을 따로 보고(사전등록 임계의 robustness).
- 서버 명령: `python scripts/run_gap_hygiene.py --elec2 --insects` (+선택 `--tabred cooking_time maps_routing`
  로 concept≈0 대조). 로컬 배선: `--synth-only`(flip 큼·invariant / noflip ~0 확인 — 통과). → `summary.json` 환류.

---

## V3.4 — 외부 적대리뷰 대응 (2026-06-24~, 별도 LLM red-team)
외부 LLM 비판검증(패키지 `Desktop/ExplainableTab_ReviewPackage`) → 핵심 수치 artifact로 검증, 대부분 정확. 우선순위:
- **P0 ✅ 완료 (ess-floor 집행 + ground-truth 재검증)**: (반론A §6 cherry-pick) + (MISSED-1 `ess_pct_floor=5.0`
  선언만·미집행)은 같은 뿌리. `drift_measure`의 measurable 게이트에 `ess_pct≥5.0` 추가(전 caller 일관) +
  None-guard. `toolkit_validation` 재검증 = **PASS 유지하며 mu=0.70(ess2.3%)·1.5·3.0 전부 abstain** → 배포규칙=검증규칙.
  `run_representation` 다중시드 CI+사전등록 verdict로 §5급 엄격화. **서버 재실행 대기**(representation 실TabReD /
  gap_hygiene = elec2·insects가 ess≥5로 생존하는지=Claim A 확인). RESULTS §20.
- **★ P0 서버결과 (2026-06-25, fresh ess-gate 런)**: ① **§6 representation 깨끗해짐** — homecredit 고-k 쓰레기
  (ess 0.6/1.1%) 이제 abstain → verdict `all-~0`; ecom/homesite `no-checkable`; weather `mixed/unstable`; **반론A·MISSED-1
  완전 해소**. ② **반전(중요)**: **elec2 FULL은 ess 0.6%로 abstain**(시간-프록시 2개가 overlap 붕괴) → §6 논지가
  *Elec2 자신에도* 적용; concept은 **de-time-leaked서만 +0.074 [concept]** 생존(G2 +0.078 일치). insects는 full(ess41%,
  +0.147)·sparse 모두 robust concept. ③ **`p0_gap_hygiene`는 stale**(elec2 +0.14560969가 P0前과 비트동일=구코드)
  → **재실행 필요**(새 코드면 elec2-full abstain). **함의: 헤드라인 +0.146(full)은 폐기, 정직한 elec2 concept=de-time-leaked
  +0.074(autocorr 검증 대기). Claim A = insects(강건) + elec2(de-time-leaked only).** ④ **C3 tuned 완료**: 공정 튜닝
  후에도 modulation 이득 ≤+0.005(대부분 ~0/음수), Spearman(gain,cov)=+0.43 → §9 strawman 반박 봉쇄, 결론 유지.
- **P1 🔄 (반론B·C: de-time-leaked Elec2 +0.074 분해) — 코드 수정 완료(de-time-leak+None-guard+ess기록), 서버 대기**: `run_elec2_decompose.py`. 3프로브:
  ① **thinning**(stride 5/25로 단기 자기상관 파괴, 장기 rule-change 보존), ② **lagged-label ablation**(y_{t-1} 추가 시
  gap 줄면 자기상관 기여), ③ **Bayes-noise proxy**(achievable-acc early/late 드리프트). 사전등록: gap이 thinning·lag
  ablation 후에도 CI가 0.034 floor 위면 Elec2 concept 생존, 무너지면 자기상관/노이즈 교락 → Claim A를 INSECTS 단일로
  재범위화(W1). 로컬 합성 검증 통과(진짜 rule-flip은 3프로브 다 생존). 서버: `python scripts/run_elec2_decompose.py --elec2 --insects`.
- **P2 ✅ 완료 (신규성 재범위화·글쓰기)**: D'Amour 2021(고차원 strict-overlap 붕괴 정리)·Budhathoki 2021·Zhang
  ICML 2023 인용 추가(REFERENCES §0.1b). §2에 attribution/overlap 계보 + "분해·overlap-붕괴 = 알려진 것, 우리=시간·표현축
  경험 시연+검증된 기권" 명시. §6은 ess-gate fresh 결과로 재작성(Elec2-full un-checkable·de-time-leak +0.074·homecredit
  garbage abstain). §5에 +0.146 supersede 캐비엇. §11 한계 (c)positive 좁음·(d)D'Amour 선점. 영/한 둘 다. (Pang 2021류
  adversarial-validation temporal-tabular 선점 확인은 잔여.)
- **P3 ✅ 완료 (사소 정직화)**: floor 단위 일관화 — gap_hygiene 기본 floor 0.034→**0.041**(bias-corrected 단위;
  raw 0.041=0.0349−(−0.0065)), representation/decompose는 raw gap이라 0.034 유지(주석 명시). §5에 placebo 음수
  오프셋 해명(raw +0.146을 1차 수치·bias-corrected +0.181은 상한). §7에 C1 negative를 "동기-결과 단절"로 명시. 영/한 둘 다.

## 우선순위 · 의존성
1. **V3.0 G1·G2·G3(서버/로컬 sklearn) + G4(웹, 내가 즉시)** — nucleus 생사. 다른 모든 것의 전제.
2. G1·G2 통과 → V3.1 F1(형식 척추) — 이게 #2~#5 모호함을 "가정 하 robustness"로 격하.
3. V3.2 C1(A↔B) — 퍼즐 설명력의 missing experiment, 헤드라인 등급 결정.
4. C2~C5, V3.3.
- **게이트 실패 시 §0.1 분기로 정직 피벗.** 어느 결과든 산출 있음.

## 타깃
워크숍: nucleus(G1·G2 통과분)로. **NeurIPS D&B**: V3.0~V3.2 완료 시 현실권(수개월). 메인트랙: + 15~20 데이터셋·C1.
