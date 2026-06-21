# RED_TEAM — 4-에이전트 적대적 검토 종합 (2026-06-17)

> 4개 독립 적대적 에이전트(논리·측정통계·신규성·Reviewer2)가 각자 "이 논문을 기각시켜라"로
> PAPER_DRAFT/RESULTS/REFERENCES를 공격. **교차검증**(≥2 에이전트 독립으로 같은 구멍 = 최우선) +
> **구조적(설계/실험 변경)** vs **문구수정** 분류. 이건 우리에게 아픈 게 목적 — 리뷰어보다 먼저 맞기 위함.

## 한 줄 결론
v0.1은 **워크숍급**. D&B엔 *특정 통제 실험 + 재프레이밍 + 범위확장*이 필요. **단 방어 가능한 핵심
nucleus("측정불가성 + 검증된 abstention")는 살아남음.** 가장 시급: 리드 청구(Claim A) 측정 도구의
home-field 교락을 placebo로 falsify해야 — 통과 못 하면 +0.132/+0.144가 concept이 아님.

---

## TIER 1 — CRITICAL & 교차검증 (안 고치면 기각)

### R1. within-overlap gap = concept인가, train/test home-field 이점인가? [측정통계 CRITICAL + 논리 M2]
- `_transfer_gap`: late-overlap 반분할, early-학습 A vs **late-학습 B**를 *같은 late 테스트*에. B는 "자기
  분포 학습→자기 분포 테스트" 홈필드 이점 → **P(y|x) 불변이어도 gap>0**(밴드 [0.1,0.9]가 넓어 잔여
  covariate를 B가 적합). + N 비대칭(early 4721 vs late_tr ~2360) 미통제. p-층 안정성은 홈필드를 *못* 잡음
  (모든 층에서 균일 → "안정"으로 오독).
- **위협**: elec2 +0.132 / INSECTS +0.144 = **측정 프레임 전체**가 홈필드를 재고 있을 수 있음. 합성검증(§4)은
  covariate를 규칙과 *직교*로 심어 이 교락이 구조상 안 나타남 → 검증이 교락 없는 경우만 통과.
- **구조적, 단 통제실험으로 해결 가능**. ★결정적 누락 실험:
  1. **placebo**: overlap 밴드 내 시간라벨 셔플(early=late) → gap **반드시 0**.
  2. **순수 prior-shift null**: P(y) 바꾸고 P(y|x) 고정 → gap **0**.
  3. N 균등화(early를 late_tr 크기로 subsample) 후 재계산.
  4. permutation null로 gap의 p값.
  → 통과하면 +0.132가 concept 확정(프레임 견고화). 실패하면 리드 청구 재설계.

### R2. Claim A(TabReD)와 Claim B(elec2/INSECTS)가 다른 데이터셋 — 퍼즐을 둘 다 직접 안 다룸 [Reviewer2 #1 + 논리 C1]
- §13상 TabReD엔 "측정가능+concept" 데이터셋이 0개 → Claim B를 TabReD에서 돌리는 게 *원리적 불가*. A는
  TabReD, B는 INSECTS/elec2, 정작 퍼즐(TabReD 랭킹)은 둘 다 직접 입증 안 함(주장만).
- **숨은 거짓 전제**: "복잡한 방법이 단순을 이기는 유일한 길은 concept 착취" — 거짓(용량·상호작용·정규화로도 이김).
- **구조적**. 해결: (a) **TabReD 방법 랭킹을 우리 렌즈에 통과** — 딥러닝 붕괴/GBDT 마진이 cov_AUC·overlap과
  상관됨을 *보임* (현재 missing experiment), 또는 (b) 청구를 "시간-구조가 concept을 착취 못 함"으로 축소,
  "단순>복잡" 전체 퍼즐 설명 주장 철회.

### R3. §7 Cai&Ye 판결: 과대주장 + *자기 데이터가 정반대를 가리킴* [Reviewer2 #6 + 논리 M3 + 통계 #8 + 신규성 #5 = 4 에이전트]
- **통계 #8(가장 날카로움)**: `modulation_adj_summary.json`이 변조 이득↔**concept** 양의 상관
  (elec2 +0.009, insects +0.028; cov_AUC와 ρ=−0.50). = §7 논지("X-side라 covariate 큰 곳서 도움")의 *정반대*.
  초안은 이걸 "inconclusive"로 묻었지만 데이터는 *반대를 가리킴*.
- **논리적**: label-free *변환* + supervised head는 시간변화 P(y|x,t) 결정면을 바꿀 수 있음 → "concept 착취
  *구조적* 불가"는 거짓(필요조건을 충분조건으로). "subsumed"는 재현도 못 하면서 포섭 주장 = 과대.
- **해결**: §7을 한 문단 *용어* 지적(Related Work)으로 강등 + 기여목록서 제거. OR LAMDA repo faithful 재현
  완료해 이득이 X-side임을 *보임*. "cannot exploit"/"subsumed" 삭제. (반쯤 하는 게 최악 — "경쟁작 재현도
  못 하면서 정의로 explain away" 인상.)

### R4. 측정 프레임 = DISDE + 시간축; "측정불가성"은 DISDE term-3 재진술 [신규성 #1 CRITICAL + Reviewer2 #3 + 통계 #2]
- within-overlap gap = DISDE term(ii). "covariate 지배→측정불가" = DISDE term(iii)(미관측영역)이 지배하는
  *퇴화 극한*. DISDE는 이걸 *이미 항으로 운영화*함. → 측정 *프레임*은 신규 아님.
- 통계 #2: ESS<1% "DISDE 사망"은 strawman — DISDE는 raw self-normalized IW 안 씀(자체 정규화·term-3 분리).
  우리 코드가 naive IW로 죽인 뒤 "DISDE 죽었다" = 부당. overlap_mass<0.05 floor도 HGB가 분리를 *제조*(임의).
- **해결**: DISDE를 *모든* overlap/transfer 언급에 인용. 프레임을 "DISDE의 시간축 적용 + IW 추정량 퇴화 노트"로
  정직 재진술. **신규 kernel = 경험적**("실 산업 시간데이터서 term-3가 corner가 아닌 *지배* 영역, 5/8") +
  **검증된 abstention**(도구가 못 잴 때 거짓 대신 기권을 ground-truth로 입증) — 이 둘만 신규로 주장.

---

## TIER 2 — MAJOR

### R5. "unmeasurable" → "absent/unexploitable" 미끄러짐 + "필요조건을 충분조건처럼" 메타 패턴 [논리 C2 + 메타]
- 반복: label-free(불가의 *필요*조건), gap≈0(없음의 필요조건), 합성충실성(건전의 필요조건), flip(redundancy와
  *일관*이지 증거 아님). 그리고 "우리가 못 잼" → "존재 안 함/착취 불가"(§9 "no architecture recovers"는
  TabPFN 반례와 모순). **문구수정이나 광범위** — 모든 "unidentifiable/no architecture"를 "overlap-기반 측정으론
  비식별"로. "측정불가 by us"와 "원리적 착취불가"를 명시 분리(후자 미입증 인정).

### R6. Q1 충실성 고아 — 엉뚱한 아키텍처 검증 [Reviewer2 #5 + 논리 C3]
- §8은 *프로토타입 메모리* 충실성, 최종 음성(§12)은 *time-TabR*. + 합성·기저정합 충실성 ⇏ "실 음성은 데이터 탓".
  메커니즘이 충실해도 최적화/분산(std 4.3×)/기저불일치로 질 수 있음.
- **해결**: §8을 "오배선 아님"으로 축소. "실 음성=데이터 탓" 추론 삭제. time-TabR이 실데이터서 *잘 최적화됨*을
  보이는 진단 추가(예: 알려진 INSECTS drift 복원).

### R7. n≈2 / 유효 n=1 split [Reviewer2 #2 + 통계 #4]
- Claim B = 한 합성 스트림(INSECTS)의 변종 2개, 각 *결정적* temporal split 1개(25시드는 init/순서만 = pseudo-
  replication). 효과 0.7%, g_z=−0.50 = 사전등록 검정력 *바닥*. → class 법칙 아님.
- **해결**: rolling-origin 다중 cut-point가 진짜 replication. 진짜 다른 데이터셋. "단일벤치 기전 설명"으로 강등.

### R8. 누락 baseline (특히 no-change) + Elec2 자기상관 [Reviewer2 #7 + 논리 m4 + 통계 #5]
- §6 비교에 no-change/persistence(Elec2 ~85%), GBDT+t, 정품 TabR, TabPFN 없음. Elec2 "skill"은 대부분 자기상관 →
  late-학습 B가 자기 테스트서 자기상관 착취 = R1 홈필드의 순수형. **Elec2는 "concept 실재" 닻으로 최악**인데 닻임.
- **해결**: no-change를 arm으로. +0.132가 자기상관 통제(블록/시간갭 분할) 후도 살아남는지 *보임*. 안 그러면 닻 교체.

### R9. [VERIFY] adversarial-validation 선점 위험 [신규성 #2] — 내가 웹으로 확인 가능
- 도메인분류기-as-shift-detector는 10년 됨(Ben-David 2010, Lopez-Paz&Oquab 2017, Rabanser *Failing Loudly*
  NeurIPS'19, Kaggle adversarial validation). **누가 TabReD류서 early/late 완전분리를 이미 보고했으면 Claim A
  경험 코어 선점.** → 웹 검증 필수(신규성 에이전트는 웹 차단됐었음, 나는 가능).
- + `covariate_shift_auc`를 adversarial-validation 계보로 인용.

### R10. [VERIFY] WhyShift에 temporal 설정 존재 [신규성 #3] — 내가 웹 확인 가능
- "공간=Y|X / 시간=X" 대조가, WhyShift(folktables)에 *연도별 temporal* 설정이 있고 그게 Y|X-지배면 *인용한 foil이
  직접 반박*. + 대조가 도메인효과(산업 vs 인구)일 수도. **해결**: 도구킷을 WhyShift 데이터에 직접 돌려 어느 영역에
  떨어지는지 *보임* → 약점이 최고 그림으로 전환.

---

## TIER 3 — 위생 (반드시, 작음)

- **R11. Claim A 임계값 미사전등록 + 숫자 표류** [통계 #6]: band/floor/clip/HGB 모두 분석자 선택, 사전등록 없음.
  elec2 +0.166→+0.132 표류. V2는 pre-V2 null 본 *후* 구축(researcher DoF). → Claim A 사전등록 or 민감도 그리드로
  3분법 불변 입증.
- **R12. BH-FDR 미적용(자체 규칙 위반)** [통계 #7]: CLAUDE.md가 FDR 의무화했는데 헤드라인 표에 0개. ρ-게이트 변종
  선택도 미보정. → contrast family에 BH 적용.
- **R13. 합성검증 너무 관대** [통계 #3]: 직교 covariate+선형규칙+등N → R1 교락이 *구조상 안 보임*. → 적대적 셀
  추가(비직교 covariate, 순수 prior-shift, 비선형 규칙).
- **R14. "concept≈0" 단일시드·무CI** [통계 #5 + 논리 M4]: cooking/maps −0.005/−0.003 = seed0 점추정, CI 없음.
  delivery −0.048 = 프레임이 thin overlap서 쓰레기 출력(기권 안 함). → 시드평균+부트스트랩 CI+null 밴드,
  "탐지가능한 concept 없음(CI⊃0)"으로 재라벨.

---

## 방어 가능한 핵심 nucleus (살아남는 것)
> "현실 산업 시간 표 데이터에서 conditional(concept) shift는 단지 작은 게 아니라, covariate가 공통 support를
> 무너뜨려 표준 조건부/재가중 렌즈로 **형식적으로 측정불가**하며 — 우리는 바로 그 영역에서 concept을 환각하지
> 않고 **기권하는, ground-truth로 검증된 진단 도구**를 제공한다."
신규 동사 = **"측정불가" + "검증된 기권(abstention)"**. ("covariate 지배"는 예측됨, "concept 작음"은 application.)

## 우선순위 실행 (결정적 → 부수)
1. ★**R1 placebo + prior-shift null** — 리드 청구 측정의 생사. 통과 못 하면 그 위 전부 무너짐.
2. **R9/R10 웹 [VERIFY]** — 선점/반박 위험(내가 즉시 가능).
3. **R2 TabReD 랭킹을 렌즈에 통과** — 퍼즐↔측정 연결의 missing experiment.
4. **R8 no-change arm + Elec2 자기상관 통제**.
5. **재프레이밍**(R3·R4·R5): nucleus로 수렴, §7 강등, DISDE 전면 인용, "unmeasurable≠absent" 분리.
6. **R7/R10 범위확장**(WhyShift 크로스런, rolling-origin, 더 많은 데이터셋).
7. **R11–R14 위생**(사전등록·FDR·합성 적대셀·시드CI).

## 메타
honesty·prereg·paired 통계는 진짜 강점(에이전트들도 인정). 문제는 *그 엄밀함을 엉뚱한 데 씀* + *기여표면 과장*.
red-team이 리드 청구의 구조적 구멍(R1)을 잡은 것 = 이 검토의 가장 큰 가치. 지금 아픈 게 리뷰어에게 아픈 것보다 낫다.

---

# 심층 패스 (foundations-level, 2026-06-17) — 3 에이전트

> 1차(위)는 "리뷰어 수준"(증거가 주장 뒷받침하나). 심층은 "구성물이 잘 정의됐나". 더 근본적.

## D1. ★표현(representation) 의존성 — Claim A 보편형은 자기 도구킷에 의해 자기반박 [FATAL→재범위화로 생존]
- `P(x)P(y|x)` 분해는 불변이나 "covariate 지배"·"concept" *라벨*은 좌표 의존. **worked counterexample**: 순수 concept drift
  데이터에 시간-프록시 피처 1개(`c=t+ε`, lag/expanding-window/calendar — 산업 파이프라인 기본) 추가 → cov_AUC→1,
  overlap→0 → "측정불가". *같은 데이터·같은 concept인데 판정 뒤집힘.* "측정불가" = 단조(피처 풍부함 × 시간상관).
- TabReD 261피처가 covariate 지배를 *제조*. drop-top-5(of 261)는 공선성으로 무력. HGB 용량도 분리 제조(논문은 ESS=100%를
  "분리 아티팩트"라 인정하면서 *같은 분류기*의 overlap=0은 "진짜 신호" 취급 — 양다리 불가).
- §4 합성검증은 covariate를 규칙과 직교·*고정 좌표*에 심어 이 모드를 구조상 못 봄.
- **생존 형태**: "*배포된 표현*에서 표준 렌즈로 concept 접근불가 = 피처 엔지니어링이 concept을 측정불가하게 만든다"
  (거짓 보편형보다 *더 신규*). ★결정 실험 **E1**: 시간-프록시 제거/희소표현서 3분법 재계산(양방향 win, 침묵만 패).

## D2. ★형식적 estimand 없음 — "측정"이 아니라 ad-hoc 통계량 [FATAL→DISDE-term 명명으로 생존]
- transfer gap = 두 HGB 추정량의 유한표본 성능차, 수렴하는 모집단 θ 없음. "자기 자신"엔 consistent, *분포 불변량*엔 아님.
  부수: ①concept은 스칼라 아님(Δ(x) 함수) — AUC-gap은 **rank-보존 recalibration drift서 gap=0**(cooking/maps가 순수
  recalibration일 수도 → 그 클래스에 맹목) ②median 이분화가 gradual≡abrupt aliasing, drift 형태 버림 ③AUC 선택 무근거
  (Brier/Bayes-risk/KL이면 다른 데이터셋이 concept 보유; 밴드 내 prior leak).
- **생존 형태**: θ := DISDE term-(ii)를 시간축에 정의 + **positivity/overlap 정리** 명시(겹침 밖 = identified set 전체
  simplex = prior 없으면 비식별) + gap을 "overlap 위 θ 추정량, 밖에선 abstain"으로 강등. → §9 자기모순("no architecture
  recovers" ↔ TabPFN) *해결*: "prior 없이 비식별, 올바른 prior는 자기 가정 안서 식별" → 설계공간={prior, online}.

## D3. ★도구킷이 자신있게 틀리는 DGP 3개 + 합성검증은 순환 [FATAL: validation 무효]
- **A1 거짓음성**: 부분영역 규칙뒤집힘(16% mass) → 전역 AUC가 평균내어 gap≈0. cooking/maps −0.005가 부분영역 drift를
  숨길 수 있음. p-층 안정성 체크가 *틀린 답을 확증*(covariate 동일이라 모든 층 균일).
- **A2 거짓양성**: 경계 *고정*인데 시간가변 Bayes 노이즈(H(y|x)↓) 또는 prior shift → **gap +0.15(elec2 +0.132와 동급)**,
  concept 0. "difficulty 통제"는 covariate-난이도만, 라벨노이즈/조건부엔트로피 drift 미통제. elec2 +0.132가 이것과 *구분 안 됨*.
- **A3 측정불가≠착취불가**: disjoint support지만 *매끄러운 궤적* 규칙drift → 도구킷 abstain, but drift-prior 방법(논문이
  인정한 TabPFN)이 *외삽으로 착취*. §9 "no architecture recovers" 깨는 구체 구성.
- **합성검증 4/4 PASS는 순환**: `make()`는 concept을 *전역 선형회전·대칭노이즈·균형prior·전support*로만 심음 = 추정량이
  설계된 *유일* 클래스. 위 3 모드를 구조상 못 만듦. → 적대적 셀(부분영역·라벨노이즈·prior·궤적) 추가 전엔 "검증된 abstention" 주장 불가.

## D4. 퍼즐이 애초에 drift 얘기가 아닐 수 있음 [Claim A의 동기 자체 위협]
- "단순>복잡 on TabReD"의 비-thesis 대안 5개(논문 미배제): ①튜닝예산 비대칭(최강) ②딥 아키텍처 최적화 불안정(논문 자기
  time-TabR이 std 4.3×) ③전처리가 트리 유리 ④용량/정규화 mismatch ⑤프로토콜 lag(Cai&Ye). 전부 covariate/concept 없이
  퍼즐 예측. **thesis의 퍼즐에 대한 *증분* 설명력 ≈ 0** until `margin ~ cov_AUC + budget + seed_var` 회귀로 분리.
- **핵심 전제 "no concept ⇒ structure 못 도움" 거짓**: 오설정 하 covariate shift서 시간-피처가 regime 지표로 재보정 도움
  (Shimodaira 2000). 논문 §7(Cai&Ye 변조가 X-side인데 *이김*)이 *자기모순으로* 이를 입증. → Claim B 해석은 숫자 맞아도
  틀림; "인스턴스 검색이 시간-피처와 in-dist redundant"로 한정해야지 "시간-구조 일반"이 아님.

---

# ★ 최종 종합 판정 (7 에이전트)

## 수렴 진단 (한 문장)
논문은 **절차-상대·추정량-특정·표현-의존적 진단을 *세계의 사실*로 제시**하고, "필요조건→충분조건"·"못 잼→없음/착취불가"로
반복 격상한다 — 이게 모든 구멍의 공통 뿌리.

## ★ 수렴 *구제* (3+ 에이전트 독립 → 가장 강한 신호 = 살릴 수 있다)
1. **형식적 estimand**: θ = DISDE term-(ii) 시간축 (D2).
2. **positivity/overlap 정리** 명시 → §9 자기모순 해결, abstention 정당화 (D2, §6 nucleus).
3. **모든 보편형 → "배포된 표현에서, overlap/reweighting 렌즈로"** 재범위화 (D1).
4. **transfer gap 강등** + **abstention을 정직 검증**(적대적 셀 A1/A2/A3 추가) (D3).
5. **Claim B 한정**: "인스턴스 검색 ≤ 시간-피처, 2 concept 데이터서" — 일반법칙 아님 (D4).
6. **A↔B 연결 실험**: `margin ~ cov_AUC + budget + seed_var` on TabReD (D4, R2).

## 판정: **구제 가능, 단 대대적 재작업. v0.1 프레이밍은 방어 불가.**
- 살아남는 nucleus(=더 흥미로움): "피처 엔지니어링이 concept을 측정불가하게 만든다 + 식별경계를 명시한 검증된-abstention 진단".
- 현재 헤드라인(보편 측정불가성, 3분법, §7 subsumed, Claim B 일반화)은 도구가 지탱 못 함.
- 워크숍: nucleus로 지금 가능. D&B: 위 6개(estimand·정리·E1·정직검증·A↔B·재범위화) = 수개월 재작업.

## 메타 (깊이는 수확체감, 단 여기까진 판정을 바꿈)
표면→심층으로 갈수록 *더 근본적* 발견(estimand 부재, 표현 의존성)이 나왔고 이는 판정을 "fixable"→"재작업 필요"로 바꿈.
**단 심층 3 에이전트가 *같은 구제*에 수렴 = bedrock 도달 신호.** 추가 패스는 nit만 늘 뿐 판정·구제경로 불변 → red-team 종료,
"재건할 것인가"의 결정 시점.
