# OVERVIEW_V2 — 현행 상태 반영판 (간이 논문, 2026-06-14)

> 기존 OVERVIEW.md는 V2 이전(Q2b 음성 무효화·재포지셔닝 이전) 서술이라 갱신 필요.
> 이 문서가 R1 완료 + R2.1–2.4 완료 시점의 현행 요약. 숫자 근거 = RESULTS.md §12–14,
> 증거사슬 = FINDINGS.md, 문헌 = REFERENCES.md §0, 계획 = PLAN_V2.md, 결정규칙 = PREREG_V2.md.

---

## 0. 요약 (Abstract)

표 데이터는 시간이 지나며 분포가 변한다(temporal distribution shift). 우리는 "시간에 따라
진화하는 프로토타입 메모리 + 검색"으로 이 변화를 해석 가능하게 다루려 했으나, 통제된 실험 끝에
**기여의 장르가 방법(method) 논문에서 측정·분석 논문으로 이동**했다. 핵심 발견 셋:

**(A, 리드)** 현실 표 시간데이터의 분포 변화는 **압도적으로 covariate(P(x))**이고, 강한 covariate가
과거/미래 공통 support를 무너뜨려 **concept(P(y|x)) 변화를 표준 조건부/재가중 렌즈로 측정조차
불가능**하게 만든다. within-overlap model-transfer 프레임으로 support 존재하는 곳의 concept을
복원한다(10 데이터셋, ground-truth 검증 완료).

**(B, 보조)** 측정 가능한 곳에서도, 시간을 *구조적으로 인덱싱한 검색*은 시간을 *입력 피처로 넣는 것*을
못 넘는다 — 교락 제거 공정 비교에서 **유의하게 못 넘는다**(25시드, paired CI<0). 지는 이유는
"외삽 장치가 아니라 in-distribution 장치"이기 때문.

**(판결)** 시간-인지 방법이 TabReD서 이긴다는 최근 결과(Cai & Ye)의 이득은, 그들 변조가 라벨에
의존 않는 피처-분포 변환임을 코드로 확인함으로써 **covariate 적응(X-side)**임을 보인다 — 그들이
"concept drift"라 부르는 것은 X-side이며 Claim A에 포섭된다.

---

## 1. 무슨 일을 한 것인가 (문제와 여정)

**원래 베팅**: TabM 백본 위에 `P_k(t) = P_k^base + drift_k(Fourier(t))` 시간-인덱싱 프로토타입
메모리를 얹고, 입력이 "자기 시점의 어느 프로토타입에 가까운지" 검색해 예측. 동기 = "Cai의 변조 ≠
검색 / TabR의 검색엔 시간 좌표 없음 → 시간-인덱싱 검색의 빈 교차점."

방향이 두 번 바뀐 검증 순서:
1. **토대**: TabM을 TabReD 8개서 공개치 ±1% 재현.
2. **메커니즘 작동 확인**: 순수 concept 합성서 시간-인덱싱 0.13 vs 시간무시 1.03 RMSE(+87%). 이후 음성 = 버그 아님 보장.
3. **실데이터 null + 진단**: TabReD 이득 0, 메모리 장식. 드리프트 해부 → 압도적 covariate(AUC≈1.0 pervasive), 시간↔정답 약함.
4. **측정 문제 발견(전환점 1)**: "concept 없음" ≠ "측정 못 함". covariate 심하면 과거/미래 입력영역 안 겹쳐 질문 자체 불가 → within-overlap 프레임. elec2 concept +0.132.
5. **충실성 게이트(Q1)**: 메커니즘이 합성 drift 복원 0.991(10/10) PASS → 음성은 "거짓말해서"가 아님.
6. **구조 vs 피처(Q2b)**: time-TabR + 시간 훅이 시간-피처 넘나? 초기 음성 →
7. **외부 감사로 음성 무효화(전환점 2)**: value 훅 선형 붕괴(`Σw·Lin(Δτ)=Lin(Σw·Δτ)`)=라벨무관 Δt피처 1개 → 가설 미검정. 기판 sub-TabR, 교락.
8. **V2 재검정**: 비퇴화 훅+정품 기판+피처 양쪽 고정+25시드 → 주 대비 두 clean 변종 CI<0 유의 음성. pre-V2보다 깨끗·강함.
9. **재포지셔닝(R2)**: 문헌으로 Claim A 미선점 확인, DISDE-퇴화 표로 3분법 확정, Cai&Ye 판결 코드 증명, 도구킷 ground-truth 검증.

요컨대 **"내 방법이 왜 안 되나"를 끝까지 추적하다 그 추적 자체가 측정 방법론·분야 현상 발견이 됨.**

---

## 2. 사전 연구 (R2.1 웹 원문 검증)

| 선행 | 무엇 | 우리와의 거리 |
|---|---|---|
| **DISDE** (Cai·Namkoong·Yadlowsky, OR 2025, arXiv:2303.02011) | 성능하락 3항 분해(seen내 어려움 / **Y\|X 변화** / 미관측영역), 공유분포 S+density-ratio | 측정 프레임 직접 선조. **ACS·위성만, TabReD/시간축 없음** → 인용+시간축·model-transfer 적응 차별화 |
| **WhyShift** (Liu et al., NeurIPS 2023) | 5 표 데이터(주로 공간) X vs Y\|X, **Y\|X 지배** | 반대 축(공간=Y\|X ↔ 시간=X). 우군화 |
| **TabReD** (Rubachev et al., ICLR 2025) | 산업 표+시간분할, 검색·DL 붕괴/GBDT 생존 | **X vs Y\|X 분해 안 함**(앙상블-std만) → 분해는 우리 고유 |
| **Cai & Ye ICML 2025** (arXiv:2502.20260) | TabReD 실패=프로토콜 결함+Fourier 임베딩 | 시간-피처 baseline 정당화 |
| **Cai & Ye NeurIPS 2025** (arXiv:2512.03678) | 피처 통계 시간 변조로 TabReD 능가, "concept" 처리 주장 | 최대 위협+기회 → R2.3 판결 |
| **Drift-Resilient TabPFN** (NeurIPS 2024) | 시간-인지 도움(SCM-shift prior) | 반례 → Claim B를 검색 *구조*로 한정 |
| FISH(~2011)·SAM-kNN(2016)·Žliobaitė Elec2 비판(2013) | 시간-인지 인스턴스 검색·Elec2 자기상관 | "빈 교차점"을 "현대 딥 표 검색 내"로 한정 |

**신규성**: TabReD에 X/Y\|X 분해 / covariate 지배→측정불가 선행 **미발견** → Claim A 미선점(프레임축만 DISDE와 PARTIAL).

---

## 3. 우리가 한 것 (검증된 기여)

### 3.1 Claim A — 측정불가 + 복원 (RESULTS §13–14)
10 데이터셋 도구킷(cov AUC + DISDE식 재가중 퇴화 + within-overlap gap) → 3분법:
- 고-covariate 5(sberbank/homesite/ecom/homecredit/weather): cov_AUC 1.0, overlap 0.000 → **측정불가**(disjoint)
- 저-covariate 2(cooking/maps): 측정가능, **concept≈0**(−0.005/−0.003)
- concept 벤치 2(elec2/insects): **실concept**(+0.132/+0.144), elec2는 DISDE 붕괴(ESS 0.55%)나 within-overlap 복원

DISDE 두 퇴화 모드(disjoint vs heavy-tail) 명시 + **도구킷 ground-truth 검증**(covariate×concept 4×4, 4/4 PASS:
복원 ρ=+1.0, concept 없으면 0, support 없으면 abstain). **ESS%=2.33(DISDE 사망)서도 within-overlap 동일 복원** = elec2 합성 증명.

### 3.2 Claim B — 구조 ≤ 피처, 교락 없는 유의 음성 (RESULTS §12)
V2 재검정(25시드, val-fair, 주 대비 `time_tabr_t − tabr_t`):
- incremental_balanced: −0.0067 [−0.012,−0.001] p=.006
- incremental_abrupt: −0.0205 [−0.034,−0.008] p<.001
둘 다 CI<0. 분해: ①기판 경쟁력 회복(−0.038 적자 소멸) ②시간 도움이나 피처가 더 나름
③**in-dist(random) 훅 도움 vs 외삽(temporal) 훅 해 = 외삽 장치 아님**(redundancy 뒷받침)
④trend기저 비단조 drift서 외삽붕괴.

### 3.3 판결 — Cai & Ye 이득은 X-side (REFERENCES §0.1)
`temporal_modulation.py` = `γ·YeoJohnson(x,λ)+β`, **라벨 y 미의존** → 시간-인덱싱 covariate 정규화,
P(y\|x) 착취 원리적 불가. 정의적 절반 완결, 경험적 절반(concept≈0 cooking/maps서도 이득>0) 서버 대기.

### 3.4 방법론 산출물
충실성 게이트(Q1), 사전등록(PREREG_V2), paired 통계, 합성 양성대조, 재사용 측정 도구킷(+ground-truth 검증).

---

## 4. 정직한 한계
- Claim B 범위 좁음(clean 2 + elec2 보조). class 음성엔 다중 벤치·방법 필요.
- 측정 프레임이 DISDE와 겹침 → "적응+확장"으로 정밀 포지셔닝.
- R2.3 경험·R2.5(Q1 큰-회전) 미실행.
- redundancy는 가설(외삽 뒤집힘 경험적 일관일 뿐, 증명 아님).

---

## 5. 현재 위치와 다음
기여 등급: "방법 실패"(논문 아님) → **"분야에서 시간-인지가 왜 안 통하는지를 측정 가능하게"**(논문).
워크숍 지금 충분, **NeurIPS D&B 주 타깃**. 남은 것: R2.3 경험(서버), R2.5 Q1 큰-회전, 지도교수 정렬.

---

## 부록 — 문서 지도
- 현행 계획 `PLAN_V2.md` / 결정규칙 `PREREG_V2.md` / 숫자 ledger `RESULTS.md`(§12–14가 V2) /
  증거 `FINDINGS.md`("V2 RE-TEST VERDICT") / 문헌 `REFERENCES.md`(§0 R2.1 검증) / 인계 `NEXT_TAB.md`.
- 역사 문서(V2 이전): `OVERVIEW.md`, `PLAN_RESCUE.md`, `Q2B_PROPOSAL.md`, `REVIEW.md`.
