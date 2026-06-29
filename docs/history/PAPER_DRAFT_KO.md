# Concept drift는 애초에 *측정 가능한가*? — Covariate 지배, 시간-인지 표 모델의 한계, 그리고 검증된 진단 도구킷

> 초안 v0.1 (2026-06-17). 타깃: NeurIPS Datasets & Benchmarks (워크숍 즉시 가능).
> 영문판(PAPER_DRAFT.md)의 한글 대응본. 숫자는 RESULTS.md §1–16에서 검증. [미완]/[향후]는
> 아직 안 된 항목. 정직한 범위: Claim A 리드 / Claim B 보조 / Cai&Ye 판결은 정의적(경험 재현은 향후).

---

## 초록 (Abstract)

표 시간 벤치마크에서 단순 모델(GBDT, MLP+시간피처)이 화려한 시간-인지 딥러닝을 자주 맞먹거나 이긴다 —
잘 알려졌으나 설명되지 않은 퍼즐. 우리는 이를 **측정(measurement)** 관점에서 기전적으로 설명한다.
첫째, 현실 표 시간데이터의 분포 변화는 **압도적으로 covariate(P(x))**다: TabReD 8개 중 5개에서
과거-미래 분류기 AUC ≈ 1.0이고 early/late 입력 support가 **분리(disjoint)**되어, concept drift(P(y|x)
변화)는 표준 조건부·중요도재가중 렌즈로 **식별 불가능**하다 — DISDE도 그 density-ratio 추정량이 퇴화한다
(유효표본수 ESS가 <1%로 붕괴). 둘째, concept이 식별 가능한 공통 support 위에서만 측정하는
**within-overlap model-transfer** 프레임을 도입하고, 통제된 covariate×concept 합성 그리드에서 검증한다:
심은 concept을 복원(ground truth와 Spearman 1.0)하고, 없으면 ~0을 내며, support가 사라지면 **기권한다**.
실데이터에서 깨끗한 3분법: 고-covariate는 *측정 불가*, 저-covariate는 *측정 가능하나 concept 부재*,
오직 Elec2(+0.132 AUC)와 INSECTS(+0.144 acc)만 *측정 가능한 실질 concept* 보유. 셋째, concept이 측정
가능한 곳에서 시간-인덱싱 검색 **구조**가 시간을 **피처**로 넣는 것을 이기는지 교락-통제 프로토콜
(25시드, paired)로 검정한다: **못 넘는다**(−0.007, −0.021, 둘 다 95% CI < 0). 그리고 이 메커니즘은
*in-distribution* 장치다(무작위 분할에선 돕고, 시간 외삽에선 해친다). 끝으로 최근의 양성 결과를 조정한다:
최첨단 시간 변조는 **구조상 label-free**라 covariate(X-side) 적응이며 concept을 착취할 수 없다 — 그 이득은
우리 설명과 모순되지 않는다. 진단 도구킷을 공개한다.

---

## 1. 서론

표 시간 분포 변화는 배포 ML 대부분의 무대다(신용·수요·가격). TabReD[Rubachev et al., ICLR 2025]는 시간-기반
분할에서 검색·사전학습 딥러닝이 무너지고 GBDT·MLP가 살아남음을 보였다. *왜* 단순 방법이 이기는지는 프로토콜
아티팩트·주기성 부재[Cai & Ye, ICML 2025]로 설명되거나 피처-통계 변조[Cai & Ye, NeurIPS 2025]로 다뤄졌으나,
**변화의 성질 자체** — covariate(P(x)) vs concept(P(y|x)) — 는 이 벤치마크에서 측정된 적이 없다.

우리는 이 퍼즐이 근본적으로 **측정** 문제라고 본다. 시간-인지 구조는 *규칙* P(y|x)가 시간에 따라 변할 때만
값을 한다(concept drift); *입력*만 변하면(covariate) 피처 적응으로 충분하고 시간 피처가 이미 그것을 잡는다.
질문: **표 시간 벤치마크에 착취 가능한 concept drift가 존재하며, 측정조차 가능한가?**

**기여.**
1. **측정 프레임**(within-overlap model-transfer gap): early/late 공통 support에서 concept을 식별 +
   *표준 재가중(DISDE)이 언제 퇴화하는지* 진단.
2. 통제된 covariate×concept 합성 그리드에서 도구킷의 **ground-truth 검증**(disjoint support 시 기권 포함).
3. 10개 데이터셋 **실증**: 고-covariate TabReD는 측정 불가, 측정 가능한 곳은 concept ~0(cooking, maps),
   Elec2(+0.132)·INSECTS(+0.144)만 실질 concept.
4. **교락-통제 음성**: 측정 가능한 곳에서 시간-인덱싱 *구조*가 시간 *피처*를 못 넘음(paired, 25시드);
   in-distribution 장치(외삽 장치 아님).
5. **판결**: 최근 양성(피처 변조)은 label-free → 구조상 X-side → 우리 설명에 포섭.
6. **공개 도구킷**(covariate AUC, DISDE-퇴화, within-overlap transfer gap).

## 2. 관련 연구

**Shift 분해.** DISDE[Cai·Namkoong·Yadlowsky, *Operations Research* 2025]는 성능 하락을 (i) 본 영역 내
어려움, (ii) overlap 내 Y|X 변화, (iii) 미관측 영역 X-shift로, 공유분포+density-ratio로 분해. 우리 within-
overlap gap은 (ii)의 **시간축·model-transfer 적응**이며, 추가로 *DISDE 추정량이 퇴화하는 지점*과 우리 프레임이
작동하는 영역을 특정한다. WhyShift[Liu et al., NeurIPS 2023]는 5개 *공간* 표 데이터에서 분해해 Y|X-shift 지배를
보임 — 우리 시간 결과와 **반대 축**(모순 아닌 상보 대조). 스트림 문헌은 covariate vs 조건부 drift를 오래전에
형식화[Gama et al. 2014; Webb et al. 2016].

**표 시간 방법.** TabReD는 shift를 holistic하게(앙상블-분산 프록시)만 특성화하고 X vs Y|X **분해를 안 함**.
Cai & Ye[ICML 2025]는 학습 lag/validation bias로 진단하고 Fourier 시간 임베딩 추가; Cai & Ye[NeurIPS 2025]는
피처 통계를 시간 변조해 TabReD서 능가. Drift-Resilient TabPFN[Helli et al., NeurIPS 2024]은 SCM mechanism-shift
prior로 *성공* — "설계는 concept이 있는 곳에서 가정을 넣어야만 돕는다"는 우리 논의와 일관.

**시간-인지 검색.** TabR[Gorishniy et al., ICLR 2024]은 시간 좌표 없는 인스턴스 검색. 시간-인지 인스턴스 선택은
스트리밍에 존재(FISH[Žliobaitė 2011], SAM-kNN[Losing et al. 2016]); 우리는 "빈 교차점"을 *현대 미분가능 딥 표*
검색으로 한정. Elec2 자기상관 비판[Žliobaitė 2013](no-change ≈85%)을 결과와 함께 보고.

## 3. 측정 프레임

**Within-overlap model-transfer gap.** covariate 외삽 오염 없이 P(y|x)가 early→late 변했는지 재기 위해:
(i) out-of-fold 시점분류기 P(late|x)를 적합, overlap 밴드 P(late|x)∈[0.1,0.9]로 한정; (ii) *고정된 late-
overlap 테스트셋* 위에서 early-overlap 학습 모델(AUC_early) vs late-overlap 학습 모델(AUC_late) 비교,
gap = AUC_late − AUC_early. 같은 테스트·같은 입력영역 ⇒ 난이도·입력분포 통제 ⇒ gap은 concept. (iii)
P(late|x) 3분위 안정성으로 잔여 covariate 검사.

**DISDE-퇴화 진단.** DISDE는 density-ratio w(x)=P(late|x)/P(early|x)로 source를 재가중. 퇴화 정량화:
**ESS** = (Σw)²/Σw²(heavy-tail/분산 모드), **overlap_mass** = P(late|x)∈[0.1,0.9] 비율(disjoint-support/
편향 모드; 완전분리 시 가중치가 clip 바닥으로 붕괴해 ESS가 호도되므로 overlap_mass가 진짜 신호).

**도구킷.** covariate_shift_auc(drop-top-k pervasiveness 포함), disde_iw_degeneration, concept_within_overlap.
학습 없이 부스팅 트리만 사용. [URL]에 공개.

## 4. Ground-truth 검증

두 손잡이를 독립 통제한 합성: mu_cov(*규칙-무관* 차원의 covariate 이동), theta(규칙 회전 = concept). 4×4
그리드에서 4개 검증 모두 통과: (1) **복원** 저-covariate서 Spearman(theta,gap)=+1.0; (2) **거짓양성 없음**
theta=0 셀 max|gap|=0.002; (3) **퇴화 단조** Spearman(mu,cov_AUC)=+1.0, Spearman(mu,overlap_mass)=−1.0;
(4) **실패 모드** mu=3.0(overlap 0.002)서 모든 theta에 대해 *측정불가* 보고 — 거짓 concept 대신 기권. 특히
mu=0.70서 DISDE 재가중은 사실상 사망(ESS=2.33%)인데도 within-overlap gap이 covariate 없는 행과 동일하게 심은
concept을 복원 — 실데이터 Elec2 현상의 통제된 증명. (RESULTS §14.)

## 5. 실데이터의 3분법

TabReD(8) + Elec2 + INSECTS (median train t로 early/late):

| 데이터셋 | cov_AUC | overlap | ESS% | n_overlap | concept_gap | 영역 |
|---|---|---|---|---|---|---|
| sberbank / homesite / ecom / homecredit / weather | 1.00 | 0.000 | — | 0 | — | **측정불가(disjoint)** |
| delivery | 0.997 | 0.061 | 0.21 | 568 | −0.048 | heavy-tail 퇴화 |
| cooking | 0.753 | 0.880 | 44.9 | 16960 | −0.005 | 측정가능, **concept ≈ 0** |
| maps | 0.566 | 1.000 | 93.1 | 20000 | −0.003 | 측정가능, **concept ≈ 0** |
| **Elec2** | 0.993 | 0.438 | 0.55 | 4721 | **+0.132** | 측정가능 concept (DISDE 퇴화) |
| **INSECTS**(incremental) | 0.707 | 0.973 | 39.3 | 19000 | **+0.144** | 측정가능 concept |

세 영역: (i) 고-covariate ⇒ disjoint support ⇒ concept **측정불가**; (ii) 저-covariate ⇒ 측정가능하나
**concept ≈ 0**; (iii) 실질 concept(Elec2, INSECTS), Elec2는 DISDE 재가중이 붕괴(ESS 0.55%)해 within-overlap
프레임이 필요. INSECTS는 *설계된* concept-drift 스트림(ground truth = drift 존재)이고 프레임이 크게 복원
(+0.144) — §4를 반영하는 실데이터 검증. (RESULTS §13.)

## 6. 구조가 concept 있는 곳에서 도움이 되나?

measurable-concept 데이터에서 시간-인덱싱 검색 **구조**(time-TabR)가 시간 **피처**(MLP+τ(t))를 넘는지 검정.
교락 제거 위해 한 인코더 공유 5-arm — mlp_t, tabr, **tabr_t**(검색+직접 시간피처), **time_tabr_t**(검색+시간
훅+직접피처) — 으로, **주 대비 time_tabr_t − tabr_t**가 시간 피처를 양쪽에 고정한 채 *구조*만 격리한다.
비퇴화 value 훅, 스케일된 유사도+key projection, full-train eval context, val-fair 선택, 25시드, paired.

**결과(temporal split, paired 95% CI):**
- INSECTS incremental: −0.0067 [−0.012, −0.001], p=.006
- INSECTS incremental_abrupt: −0.0205 [−0.034, −0.008], p<.001

둘 다 **유의하게 음수**: 구조가 피처를 못 넘고, 오히려 약간 손해. 분해: 검색 기판이 이제 경쟁력 있음
(tabr_t − mlp_t ≈ 0; 이전 −0.038 적자 소멸), 시간은 검색을 도움(time_tabr_t − tabr > 0)이나 *피처*가 시간을
적어도 그만큼 잘 나름. **in-distribution vs 외삽 뒤집힘**: 시간 훅은 무작위 분할서 *도움*(+0.005, +0.021),
시간 분할서 *해*(−0.007, −0.021) — 외삽 장치가 아닌 in-distribution 장치, 표현 redundancy 설명과 직접 일관
(시간-입력 신경망은 이미 in-distribution P(y|x,t)를 표현). (RESULTS §12.)

## 7. 최근 양성 결과의 조정 (판결)

Cai & Ye[NeurIPS 2025]는 피처 통계를 시간 변조해 TabReD 능가, "concept drift" 처리라 주장. 그 변조는
x ↦ γ(t)·YeoJohnson(x, λ(t)) + β(t)이며 γ, β, λ가 시간 임베딩의 선형 — **label-free**(어떤 항도 y에 의존
않음). 구조상 시간-인덱싱 *covariate* 정규화라 P(y|x) 변화를 **착취할 수 없다**; 그들이 "concept drift"라 부른
건 피처-분포(X-side) drift다. 따라서 그들의 양성 결과는 우리 설명에 **포섭**된다: 시간-인지 방법이 이길 수
있으나 covariate 적응에 의해서지 concept 착취가 아니다(concept은 이득이 가장 큰 곳에서 측정불가). **[미완]**
그들의 튜닝된 파이프라인으로 이득을 X-shift 영역에 국소화하는 경험적 재현은 향후 과제; 우리 최소 재현은 그들
이득을 재현 못 해 경험적 절반은 inconclusive(정의적 논거로 충분·load-bearing). (RESULTS §16.)

## 8. 메커니즘은 충실하다 (음성은 고장난 방법이 아님)

"구조가 진 건 메커니즘이 부서져서"를 배제하기 위해, 진짜 drift 방향 w(t)를 아는 합성 concept에서 기능적
충실성을 검증: recovery = mean_t cos(ŵ(t), w(t)), ŵ(t)=E_x[∂score/∂x](게이지 고정, Procrustes 없음).
메커니즘은 90° 회전서 0.991(10/10), **전 2π 회전 + 정합 Fourier 기저서 0.988(10/10)** 복원, 이때 무작위 바닥은
0.017로 붕괴(천장 0.972) — 넓은 동적범위에 걸쳐 충실. 따라서 음성은 *데이터*와 구조의 *in-distribution 성질*
때문이지 결함 메커니즘 때문이 아니다. (RESULTS §8, §15.)

## 9. 논의: 설계는 언제 도울 수 있나?

장벽은 모델 용량이 아니다. (i) disjoint support 하에선 미래 입력 위 규칙 변화가 **식별 불가** — 함수의 변화를
짝 관측이 없는 정의역에서 복원하는 아키텍처는 없다. (ii) concept ≈ 0인 곳엔 신호가 없다. (iii) concept이
실재하는 곳에서 시간 shift 하 concept 처리는 **외삽** 문제이고, in-distribution 구조를 더해도 외삽 우위는
생기지 않는다(뒤집힘; redundancy). 설계는 **(a) drift prior 주입**(예: Drift-Resilient TabPFN의 SCM-shift
prior — 가정-기반, concept 있는 곳에서만) 또는 **(b) online 프로토콜로 이동**(정적 외삽이 아니라 적응 속도·
샘플 효율이 지표)으로만 도울 수 있다. 이로써 설계 공간을 구획 — 음성의 건설적 따름정리.

**한계.** Claim B는 clean 2개(+Elec2는 노이즈 기판)에 기댐; 측정 프레임이 DISDE와 겹침(발명 아닌 시간축
적응+퇴화 확장으로 포지셔닝); Cai&Ye 경험적 판결은 faithful 재현 대기; redundancy는 in-distribution 논증이지
증명 아님.

## 10. 결론

표 시간 벤치마크에서 covariate 지배가 concept drift를 표준 조건부/재가중 렌즈로 측정불가하게 만든다;
within-overlap model-transfer 프레임은 공통 support가 있는 곳에서 그것을 복원하며, "단순이 복잡을 이긴다"는
퍼즐이 상당 부분 *착취 가능 concept의 부재(또는 측정불가)* 이고 — concept이 있는 곳에서도 시간-구조 검색이
시간 피처에 대해 in-distribution redundant 함을 보인다. 어떤 벤치마크가 실제로 *어떤* shift를 담는지 방법
선택 전에 특성화할 수 있도록 검증된 진단 도구킷을 공개한다.

---

## 부록 / 포인터 (내부)
- 숫자: RESULTS.md §1–16. 증거: FINDINGS.md. 문헌(검증): REFERENCES.md §0.
- 코드: run_disde_degeneration.py(§3,§5), run_toolkit_validation.py(§4), run_elec2_q2.py(§6),
  run_modulation_adjudication.py(§7), run_q1_faithfulness.py(§8). 결정규칙: PREREG_V2.md.

## [향후] 제출 전
- 방법 sweep 확대(GBDT+t, kNN+t, no-change, 정품 TabR, Drift-Resilient TabPFN) 앵커로.
- Faithful Cai&Ye 재현(LAMDA repo)으로 §7 경험적 절반 완성.
- (메인트랙) 표 시간 15–20개로 확장; 여러 시간-인지 방법을 같은 렌즈로.
- 도구킷 패키징: API, datasheet/Croissant, 재현 파이프라인.
- Claim B 폭 위해 INSECTS abrupt/gradual(단 abrupt는 val→test 게이트 탈락 — 발견으로 보고).
