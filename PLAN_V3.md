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

### C2. Claim B 재범위화 + 앵커 [R7, R8, D4]
- "인스턴스 검색이 시간-피처와 in-dist redundant, 2 concept 데이터서" — *일반 법칙 아님*. "no concept⇒구조 무용" 삭제
  (오설정 재보정 반례; §7 자기모순). rolling-origin 다중 cut. **no-change/persistence·GBDT+t·정품 TabR 앵커**(R8).

### C3. §7 Cai&Ye [R3]
- 한 문단 *용어* 지적으로 강등 OR LAMDA repo faithful 재현 완료. **변조 이득↔concept ρ=−0.50(자기 데이터가 정반대)
  정직 보고**. "cannot exploit"/"subsumed" 삭제.

### C4. Q1 고아 수정 [R6]
- §8을 "오배선 아님"으로 축소. OR time-TabR(실제 쓴 메커니즘)이 실데이터서 알려진 drift 복원하는 진단 추가.

### C5. 폭 [D1, R10, R7]
- WhyShift 데이터에 도구킷 크로스런(대조 입증 or 도메인효과 인정). 데이터셋 10→가능한 만큼 확장.

---

## V3.3 — 재작성 + 위생
- PAPER_DRAFT 재작성(nucleus 헤드라인, 기여 6개→정직 재구성, DISDE 전면 인용, estimand/정리 절 신설).
- BH-FDR 전 contrast family 적용(R12). Claim A 임계 사전등록 or 민감도(R11). 모든 실데이터 수치 seed-CI.

---

## 우선순위 · 의존성
1. **V3.0 G1·G2·G3(서버/로컬 sklearn) + G4(웹, 내가 즉시)** — nucleus 생사. 다른 모든 것의 전제.
2. G1·G2 통과 → V3.1 F1(형식 척추) — 이게 #2~#5 모호함을 "가정 하 robustness"로 격하.
3. V3.2 C1(A↔B) — 퍼즐 설명력의 missing experiment, 헤드라인 등급 결정.
4. C2~C5, V3.3.
- **게이트 실패 시 §0.1 분기로 정직 피벗.** 어느 결과든 산출 있음.

## 타깃
워크숍: nucleus(G1·G2 통과분)로. **NeurIPS D&B**: V3.0~V3.2 완료 시 현실권(수개월). 메인트랙: + 15~20 데이터셋·C1.
