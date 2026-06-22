# Concept Drift는 *네가 배포하는 표현*에서 측정 불가능하다: 표 시간 분포변화를 위한 positivity-경계 진단

> 초안 v3.0 (2026-06-17). 7-에이전트 적대적 검토(RED_TEAM.md) + V3.0 결정적 게이트 실험(PLAN_V3.md G1–G4,
> 전부 통과/재범위화) 이후 재형식화. PAPER_DRAFT.md(v0.1, red-team이 무너뜨린 판)를 대체. 영문판
> PAPER_DRAFT_V3.md의 한글 대응본. 숫자: RESULTS §1–16 + gap_controls/representation/toolkit_adversarial 등.
> 정직 범위: 모든 보편 청구를 *배포된 표현* 하 *overlap/reweighting 렌즈*로 한정. 리드 기여 = 형식적 식별
> 경계 + ground-truth *및* 적대적으로 검증된 기권(abstaining) 진단. [미완]/[향후]는 아직 안 된 항목.

---

## 초록

표 시간 벤치마크에서 단순 모델이 화려한 시간-인지 딥러닝을 맞먹거나 이긴다. 우리는 측정이론적 설명을 주되
*그 설명이 주장할 수 있는 범위*를 못박는다. **(1) 형식적 대상.** DISDE[Cai·Namkoong·Yadlowsky 2025]를 시간축에
적응해 concept-drift estimand θ를 early/late *overlap* 위 P(y|x) 변화로 정의하고, 표준 조건부/재가중 렌즈가
*overlap이 사라질 때 정확히 퇴화*(positivity 실패)함을 보인다. **(2) 세계가 아니라 표현의 성질.** overlap
positivity 성립 여부는 피처 맵의 함수다: TabReD에서 시간-프록시 피처 *하나*를 더하면 측정가능 concept(+0.98)이
"측정불가"로 뒤집히고; 고-covariate TabReD 5개 중 4개는 시간-누수 피처를 벗기면 다시 측정가능해지며 — 그때
concept은 ≈0이다. 따라서 "표 시간데이터에서 concept 측정불가"는 보편 명제로는 거짓이고, 참은 **"*배포된*
261-피처 표현에서 concept은 overlap 렌즈로 점검 불가하며, 점검 가능한 표현(희소 예측 표현)에서 보면 genuinely
≈0이거나 support가 환원 불가하게 disjoint"**다. **(3) 검증된 기권 진단.** 도구킷(adversarial-validation
covariate AUC, density-ratio 퇴화, within-overlap transfer gap)은 심은 concept을 복원(Spearman 1.0)하고
positivity 실패 시 *기권*하며, 우리는 두 실제 실패모드를 *적대적으로* 특성화한다(~0.03 조건부엔트로피 교락;
*가역 좌표변환*이 아니라 *피처 추가*에 대한 표현 민감성). 진짜 concept 벤치(Elec2 +0.146, INSECTS +0.150)는
permutation placebo를 통과(gap은 train/test home-field 아티팩트 아님)하고 *표현 변화도* 통과한다. **(4)** concept이
진짜 있는 곳에서도, 시간-인덱싱 검색 *구조*가 시간 *피처*를 못 넘으며(paired, 25시드), in-distribution 장치다.
진단 도구를 식별 경계와 함께 공개한다.

---

## 1. 서론

산업 표 데이터의 시간-기반 분할(TabReD[Rubachev et al. 2025])은 검색·사전학습 딥러닝을 무너뜨리고 GBDT가
살아남는다. *왜*는 논쟁적이나(프로토콜 아티팩트[Cai&Ye 2025a]; 피처 변조[Cai&Ye 2025b]) **shift의 성질** —
covariate P(x) vs concept P(y|x) — 은 측정된 적 없다. 시간-인지 구조는 규칙 P(y|x)가 변할 때만 값을 한다;
아니면 시간 피처로 충분하다. 우리는 이 벤치마크에 착취 가능 concept이 존재하는가를 묻고 — 배포된 표현에서 그
질문 자체가 ill-posed임을 발견해 — *측정가능성*을 연구 대상으로 삼는다.

**기여(정직, 재건 후).**
1. **형식적 estimand + 식별 경계**(§3): concept drift = DISDE의 overlap Y|X 항(시간축), positivity 명제로
   *겹침 밖 concept은 prior 없이 비식별* — 이것이 기권을 정당화하고 설계공간을 {drift-prior, online}으로 확정.
2. **표현 상대성, 시연**(§6): 측정가능성 판정은 피처 맵의 함수; 피처 엔지니어링이 "측정불가"를 제조함을 보이고
   경험적 그림을 배포 vs 희소 표현으로 재범위화.
3. **ground-truth *및* 적대적으로 검증된 기권 진단**(§4–5): 심은 concept 복원, positivity 실패 시 기권, 실패모드 공개.
4. **양성이 아티팩트 아닌 concept임을 통제**(§5): permutation placebo가 home-field 교락 반증(Elec2/INSECTS 생존).
5. **교락-통제 음성**(§8): concept 있는 곳서 시간-인덱싱 검색 구조 ≤ 시간 피처, in-dist(외삽 아님) 기전.
6. **공개 도구킷**(식별 경계 명시).

## 2. 관련 연구

**Shift 분해.** DISDE[*Oper. Res.* 2025]는 성능하락을 (i)본 어려움 (ii)overlap 내 Y|X (iii)미관측영역 X로,
공유 overlap 측도 S+density-ratio로 분해. **우리 concept estimand = 그들 term(ii) 시간축; 우리 "측정불가" =
term(iii) mass→1 영역.** DISDE를 프레임 출처로 인용하고, 시간축·model-transfer 운영화 + *판정의 표현-상대성*
+ 검증된 기권을 기여. **adversarial validation**(classifier-two-sample-test; Lopez-Paz&Oquab 2017; Rabanser
*Failing Loudly* 2019; Kaggle; 신용평가 Pang 2021)이 covariate-AUC의 계보 — 기법 신규성 없음. **WhyShift**[Liu
2023]는 *공간* 표서 Y|X 지배, 그리고 *자기 temporal 셋(ACS Time)서 X-shift 지배를 이미 언급* — 우리 공간/시간
대조를 **지지**하되 "temporal=covariate"를 부분 선점; 우리 신규성은 covariate 지배가 아닌 측정불가성/positivity
결과. TabReD는 X vs Y|X 분해 안 함; Cai&Ye는 프로토콜 수정/피처 변조. **Drift-Resilient TabPFN**[Helli 2024]은
SCM drift prior로 성공 — 우리 §3 corollary가 예측하는 존재증명. **스트림**: covariate-vs-concept는 분야 창립
분류[Gama 2014; Webb 2016]; 시간-인지 인스턴스 검색 존재[Žliobaitė 2011; Losing 2016] → "빈 교차점"은 현대
미분가능 딥 표 검색으로 한정. Elec2 자기상관 비판[Žliobaitė 2013]은 통제(§5 no-change).

## 3. 세팅, Estimand, 식별 경계

**Estimand.** overlap O = supp(P_early) ∩ supp(P_late) 위 기준측도 S 고정. concept drift를
  θ := E_{x∼S}[ ℓ(P(y|x, late)) − ℓ(P(y|x, early)) ]
로 정의(ℓ=skill 함수). θ를 **within-overlap model-transfer gap**으로 추정: OOF 시점분류기 P(late|x)∈[0.1,0.9]로
O 선택; early/late-overlap에 f_early, f_late 학습; 고정 late-overlap 테스트서 gap = ℓ(f_late) − ℓ(f_early).
가정 (A1)overlap positivity(O 위 P_early, P_late 상호절대연속) (A2)O 위 P(y|x) realizability 하에서 gap은 O 위
θ에 consistent. ℓ=AUC(검색 예측기에 discrimination-관련) 사용, ℓ∈{Brier, Bayes-risk, KL} robustness는 [미완];
rank-보존 recalibration kernel은 인정된 맹점.

**명제(positivity / 겹침-밖 비식별).** R = supp(P_late) \ supp(P_early). 조건부에 제약 없으면, R 위 P(y|x)는
{P_early, P_late}로부터 비모수 비식별: supp(P_early)서 일치하고 R서 다른 두 조건부가 동일 관측분포를 유도(R에
early 데이터 없음). 따라서 O 밖 concept 변화는 extrapolation prior 없이 전체 simplex로 set-identified(무정보).
*Corollary.* 올바른 prior(smoothness/SCM, Drift-Resilient TabPFN)는 자기 가정 안서 식별; 겹침-밖 concept의
설계공간은 정확히 {drift-prior, online}, **"어떤 아키텍처도 못 함"이 아님.** 이것이 positivity 실패 시 **기권**을
정당화하고 순진한 불가능성 프레이밍을 교정.

**표현-상대성(선두 명시).** O, 따라서 (A1) 성립 여부는 피처 맵 φ의 함수. 피처 추가(특히 엔지니어링 시간-프록시)는
O를 줄임. 그래서 모든 진단을 *표현 상대적*으로 보고: 배포 피처셋, 그리고 희소 예측셋(§6). "측정불가"는 항상
"이 표현에서, overlap 렌즈로".

## 4. 진단 도구킷과 검증된 기권

**구성.** covariate_shift_auc(adversarial validation + drop-top-k pervasiveness), disde_iw_degeneration(ESS·
overlap_mass — heavy-tail vs disjoint 모드), within-overlap transfer gap. Model-light(부스팅 트리). [URL] 공개.

**Ground-truth 검증**(covariate×concept 합성 그리드, RESULTS §14): 심은 concept 복원(Spearman 1.0), concept
없으면 ~0(max|gap|=0.002), 퇴화는 covariate 단조(ρ=±1.0), disjoint서 **기권**(4/4). ESS%=2.33(재가중 사망)서도
within-overlap gap이 심은 concept 복원 — Elec2의 통제된 유사물.

**적대적 검증**(정직한 부분, toolkit_adversarial): 그리드가 안 심는 DGP를 돌림. 도구킷은 16%-mass 부분영역
규칙뒤집힘을 잡고(gap +0.156); covariate가 규칙축에 얽혀도 거짓발화 안 함(−0.004); disjoint-but-smooth 궤적에
대한 "기권"이 착취가능 concept을 숨기지 않으며(time-aware 이득 +0.001); *가역 좌표변환에 불변*(gap +0.355 vs
회전 +0.332). 두 실제 실패모드를 보고: (a) **조건부엔트로피(라벨노이즈) 교락 ~+0.034**(경계 고정+early노이즈/
late클린이 작은 거짓 concept으로 — 헤드라인 +0.15보다 한참 아래지만 명시 caveat); (b) **피처 *추가*에 대한
민감성**(좌표 선택 아님) = §6 표현 결과.

## 5. Gap은 home-field 아티팩트가 아니라 concept을 잰다

transfer gap은 late-학습 모델(자기 분포서 테스트)을 early-학습과 비교 — train/test "home-field" 이점 가능.
**permutation placebo**로 검정: overlap 밴드 *내*에서 early/late 라벨 셔플(같은 영역, 진짜 구조 없음); 진짜
concept gap은 ~0이어야. 결과(gap_controls, 15시드, 95% CI):

| 데이터셋 | true gap [CI] | placebo [CI] | concept (true − placebo) |
|---|---|---|---|
| **Elec2** | +0.146 [.140,.151] | −0.035 [−.038,−.033] | **+0.181** (생존) |
| **INSECTS** | +0.150 [.147,.152] | −0.017 [−.019,−.015] | **+0.167** (생존) |
| cooking | −0.009 | −0.012 | +0.003 (≈0) |
| maps | −0.003 | −0.003 | +0.000 (≈0) |

두 concept 벤치의 true-gap CI가 placebo CI보다 한참 위: gap은 home-field 아님(placebo 바닥은 작고 약간 음수).
합성 양성대조 true +0.985 / placebo +0.005; prior·noise null은 +0.001, +0.034(§4 caveat). 순: Elec2/INSECTS는
home-field 바닥 + 노이즈 caveat 둘 다 빼도 ~+0.11–0.15 concept 보유.

## 6. 피처 엔지니어링이 측정가능성을 통제한다 (표현 결과)

**역방향 시연.** 진짜 concept 합성(gap +0.98, overlap 0.997)이 시간-프록시 c = t + ε *하나* 추가로 "측정불가"
(overlap 0.027). 같은 데이터·concept. 3분법은 표현 함수.

**TabReD**(representation summary): 5개 "disjoint" 데이터셋을 de-time-leak / sparse-MI(@5–50) 표현서 재계산:
- **4/5가 측정가능 + concept ≈ 0**: sberbank(sparse: cov 0.63, overlap 0.996, gap +0.02), homecredit(cov 0.50,
  overlap 1.0, gap +0.00), homesite(cov 0.91, gap −0.01), weather(de-time-leak: cov 0.92, overlap 0.72, gap +0.02).
- **ecom만 진짜 disjoint** 모든 표현서(cov 1.0, overlap 0) — 환원 불가 케이스.
- concept 벤치는 **de-time-leak서도 생존**: Elec2 +0.132→+0.078, INSECTS +0.144→+0.108(concept은 표현-robust, 피처 아티팩트 아님).

**재범위화 3분법**(기존 §13 대체): *배포된 표현*서 TabReD 5/8 점검 불가(positivity 실패); *희소 표현*서 점검하면
concept≈0(4) 또는 환원불가 disjoint(1); cooking/maps는 점검 가능+concept≈0; Elec2/INSECTS는 placebo(§5)·표현
변화 생존하는 concept. 즉 실무자 상황: *자기 피처 파이프라인*이 concept 점검에 필요한 overlap을 파괴하고, 점검
가능 표현을 복원하면 (거의) 아무것도 없다.

## 7. 왜 "단순 ≥ 복잡"을 설명하나 (그리고 무엇은 설명 안 하나)

퍼즐은 *비교* 결과(단순 ≥ 복잡)이고 우리 설명이 이를 다룸: concept이 genuinely ≈0(대부분 TabReD, 점검됨)이거나
점검 불가인 곳에선 concept-표적 구조가 더 착취할 게 없고, 시간 *피처*가 가용 covariate 적응을 포착. 단 이것이
시간 피처의 *효능* 자체를 설명하진 **않음** — 그건 모델 오설정 하 covariate 재보정[Shimodaira 2000], X-side 기전
(§9). 그리고 TabReD 랭킹의 대안 설명(튜닝예산 비대칭, 딥 최적화 불안정, 전처리)은 우리가 배제 못 함; thesis
증분설명력 분리는 `margin ~ cov_AUC + budget + seed_var` 회귀 필요[향후, C1].

## 8. concept 있는 곳서 시간-구조 vs 시간-피처 (재범위화)

measurable-concept 데이터서 5-arm 공유인코더가 구조 격리: 주 대비 **time_tabr_t − tabr_t**(직접 시간피처 양쪽
보유)가 유의 음성(INSECTS incremental −0.0067 [−.012,−.001] p=.006; incremental_abrupt −0.0205 [−.034,−.008]
p<.001; 25시드 paired). 기판 경쟁력(tabr_t ≈ mlp_t), 훅은 **in-dist 도움(random +0.005,+0.021)/temporal 외삽
해(−0.007,−0.021)** — in-dist 장치, redundancy와 일관. **범위(정직):** "시간-인덱싱 인스턴스 검색이 시간 피처와
in-dist redundant, 2 concept 벤치서", *시간-구조 일반 무용 법칙 아님* — covariate 재보정은 도움(§9), 우리 §9가
covariate shift 하 시간-변조 승리를 인정.

## 9. 최근 양성 결과의 조정 (반박 아닌 remark)

Cai&Ye[2025b]는 피처 통계를 시간 변조해 TabReD 능가, "concept drift" 처리라 칭함. 변조는 label-free(변환에 y
없음)라 구조상 covariate(X-side) 정규화 — 그들이 "concept drift"라 부른 건 피처-분포 drift. 우리는 이를 **용어**
지적으로 표시(반박 아님): 우리 최소 재현은 그들 이득을 재현 못 하고, 사실 변조 이득이 *측정된 concept과* 상관
(covariate와 ρ=−0.50)하므로, 그들 이득이 X-side라고 *경험적으로* 주장 **안 함** — 변환이 라벨에 *직접* 조건화
못 한다는 것만. 이득을 국소화하는 faithful 재현은 [향후].

## 10. 메커니즘은 오배선 아님 (범위 한정)

시간-인덱싱 메커니즘 저성능의 이유가 구현 버그인지 배제 위해, 정합 기저 합성 concept서 기능적 충실성 검증:
복원 0.991(90° 회전), 0.988(전 2π 회전; random 바닥 0.017, 천장 0.972). **범위:** 정합-기저 세팅서 오배선
아님을 보일 뿐, 실데이터 최적화 품질을 보증하진 않음(별개 문제); §8 음성은 in-dist redundancy이지 버그 아님.

## 11. 논의

식별 경계(§3)가 설계공간을 구체화: 겹침-밖 concept은 **drift prior**(concept 있고 prior 맞을 때 — TabPFN) 또는
**online 적응**(지표=적응 속도, 정적 외삽 아님)으로만 도달. 연구 벤치서 binding fact는 아키텍처 *상류*: 피처
엔지니어링이 positivity를 파괴하고, 점검 가능 현실은 ~0 concept.

**한계.** (a)ℓ=AUC가 recalibration drift에 맹목[robustness 미완]; (b)median early/late 분할이 drift 형태 aliasing
— rolling-origin 궤적 [향후]; (c)Claim B는 2 concept 벤치; (d)within-overlap은 DISDE 시간축 적응이지 새 식별전략
아님; (e)A↔B의 TabReD 랭킹 연결은 C1 대기 주장; (f)§9 경험 조정 미재현; (g)contrast family BH-FDR 아직[위생 미완].

## 12. 결론

표 시간 벤치마크에서 concept drift가 *측정 가능한가*는 데이터의 사실이 아니라 *배포된 피처 표현*의 사실이다:
산업 피처 엔지니어링이 조건부 측정에 필요한 early/late overlap을 파괴하고, 점검 가능 표현을 복원하면 concept은
genuinely 작다. 우리는 estimand와 그 positivity 경계를 형식화하고, 기권 진단을 ground-truth·적대적 양면으로
검증하며, 헤드라인 양성을 home-field 교락에 대해 통제하고, concept이 있는 곳에선 시간-인덱싱 검색 구조가 시간
피처와 in-dist redundant함을 보인다. 도구킷을 식별 경계와 함께 공개해, 방법 선택 *전에* — 벤치마크가 *어떤*
shift를 담는지뿐 아니라 *그 표현이 그 질문을 허용하는지*를 — 물을 수 있게 한다.

---

## 부록 / 포인터 (내부)
- 숫자: RESULTS §1–16; gap_controls/representation/toolkit_adversarial/disde/toolkit_validation summaries.
- Red-team: RED_TEAM.md(7 에이전트). 재건+게이트 판정: PLAN_V3.md. 문헌: REFERENCES §0.
- 코드: run_gap_controls.py(§5), run_representation.py(§6), run_toolkit_adversarial.py(§4),
  run_disde_degeneration.py / run_toolkit_validation.py(§4), run_elec2_q2.py(§8), run_q1_faithfulness.py(§10).

## [향후] 제출 전 (우선순위)
1. C1: TabReD 리더보드서 `margin ~ cov_AUC + budget + seed_var`(§7 퍼즐 연결 입증).
2. ℓ-robustness(Brier/Bayes-risk/KL) + rolling-origin gap 궤적(§3, 한계 a,b).
3. Faithful Cai&Ye 재현(§9). no-change/GBDT+t/TabPFN 앵커(§8).
4. WhyShift 데이터를 도구킷에(공간/시간 대조 입증).
5. 위생: BH-FDR; Claim-A 임계 사전등록/민감도 그리드; 모든 실 gap에 seed-CI.
6. 도구킷 패키징: API, datasheet/Croissant, 재현 파이프라인.
