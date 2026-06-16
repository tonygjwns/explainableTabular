# 지도교수 정렬 브리프 (1~2장, 2026-06-17)

> 목적: 현재까지 결과를 한눈에 보고 **타깃 학회·범위·프레이밍**을 정렬. 상세=OVERVIEW_V2.md/RESULTS.md.

---

## 0. 한 문단 결론

표 데이터 temporal drift를 "시간-인덱싱 프로토타입 메모리+검색"으로 다루려 했음. **원래의 성능 가설
(구조가 단순 시간-피처를 이긴다)은 반증됐다.** 그러나 *왜 틀렸는지*를 추적한 결과가 발견이 됐다:
**(A)** 현실 표 시간데이터의 drift는 압도적 covariate이고, 그 covariate가 공통 support를 무너뜨려
concept(P(y|x))을 표준 렌즈로 **측정조차 불가능**하게 만든다 — 단 within-overlap 프레임으로 support
있는 곳에선 복원된다(10데이터셋, ground-truth 검증). **(B)** 측정 가능한 곳에서도 시간-인덱싱 *구조*는
시간-*피처*를 유의하게 못 넘는다(교락 제거, 25시드). → **기여 장르가 method 논문 → 측정·분석 논문으로
이동.** 메커니즘이 고장나서가 아니라(합성 +87%, 충실성 robust PASS) 데이터 성질+외삽 불가능성 때문.

## 1. 가설은 틀렸나? (정직한 분해)

- 성능/구조 우위 가설: **반증(유의 음성)**.
- 암묵 전제 "실데이터에 착취 가능 concept 존재": **반증(측정불가/부재)**.
- "메커니즘은 concept 있으면 충실": **확인**(합성 +87%, Q1 복원 0.988).
- → 실패 위치가 *방법*이 아니라 *데이터 성질 + 외삽 불가*로 특정됨. **그 특정이 기여.**

## 2. 무엇을 확인했나 (검증된 숫자)

| 결과 | 근거 | 신뢰 |
|---|---|---|
| 파이프라인 재현 | TabM 8개 공개치 ±1% | solid |
| 메커니즘 작동(버그 아님) | 순수 concept 합성 +87% | solid |
| **Claim A — covariate 지배→측정불가** | TabReD 5/8 cov_AUC≈1.0·overlap 0(측정불가) / cooking·maps 측정가능 concept≈0 / elec2 +0.132·insects +0.144 | solid (10데이터셋) |
| DISDE 표준 재가중 퇴화 | elec2 ESS 0.55%; within-overlap은 복원 | solid |
| 도구킷 ground-truth 검증 | covariate×concept 4×4, 4/4 PASS(복원 ρ=+1, support 없으면 abstain) | solid |
| **Claim B — 구조≤피처** | time_tabr_t−tabr_t = −0.0067[−.012,−.001] / −0.0205[−.034,−.008], 25시드 paired CI<0 | solid |
| 왜 지나(기전) | in-dist(random) 도움 / 외삽(temporal) 해 = 외삽 장치 아님 | solid |
| Q1 충실성 robust | 2π 전회전+Fourier: 바닥0.894→0.017, 복원0.988(10/10 PASS) | solid |
| 판결 — Cai&Ye 이득 X-side | 변조 label-free(코드 확인)→P(y|x) 착취 구조적 불가 | 정의적 solid / 경험 inconclusive |

## 3. 신규성 (문헌 원문 검증 완료)

- **DISDE**(OR'25): overlap내 Y|X 측정 *프레임*의 선조 → 인용+**시간축·model-transfer 적응, 퇴화영역 확장**으로 차별화(재발명 아님).
- **WhyShift**(NeurIPS'23): 공간 표=Y|X 지배 ↔ 우리 시간 표=X 지배 (대조 우군).
- **TabReD**(ICLR'25): X/Y|X 분해 **안 함** → 분해는 우리 고유.
- **Cai&Ye**(NeurIPS'25): 변조로 TabReD 능가="concept" 주장 → 우리 판결로 X-side임을 보임(포섭).
- **린치핀**: TabReD에 분해/측정불가 주장한 선행 **없음** → **Claim A 코어 미선점**(프레임축만 DISDE와 PARTIAL).

## 4. 왜 이게 논문인가

"내 방법이 실패"가 아니라 **"분야 전체에서 시간-인지 방법이 왜 기대만큼 안 통하는지를 측정 가능하게 만들고,
설계가 값을 더할 수 있는 정확한 조건(concept 실재 + prior 필요 / online)을 구획"**. 선례: WhyShift·Wild-Time·
Grinsztajn("왜 트리가 이기나") — 측정·분석 논문으로 탑티어 입성. **도구킷이 ground-truth 검증까지 완료**된 게 D&B 강점.

## 5. 정직한 한계

- Claim B는 clean 2개(+elec2 보조)로 좁음 → A를 리드, B를 보조로.
- 측정 프레임이 DISDE와 겹침 → "적응+확장+실증"으로 정밀 포지셔닝 필수.
- Claim A 판결의 경험적 절반(R2.3)은 최소 재현이라 미확증 → LAMDA repo gold 재현은 future work.
- redundancy는 in-dist만 닫는 *가설*(증명 아님) — 외삽/online 경로는 열려 있음.

## 6. 지도교수 결정 요청 (정렬 포인트)

1. **타깃 학회**: 워크숍(지금 자산 충분, 우선권) / **NeurIPS D&B(주 타깃, 적당 범위확장)** / 메인트랙(15~20데이터셋·다중방법, 수개월)?
2. **범위 투자**: 데이터셋·방법 sweep을 어디까지? (D&B면 ~10-12데이터셋 × 8-12방법 + 도구킷 패키징.)
3. **R2.3 faithful 재현**: LAMDA repo를 직접 돌려 경험적 절반을 채울지(gold-standard) vs 정의적 논거로 충분한지?
4. **프레이밍 승인**: Claim A 리드 + Claim B 보조 + 판결을 X-side로. (B를 A보다 앞세우지 않음.)

## 7. 현재 상태 한 줄
R0(인프라)~R2(분석) **코드·실험 전부 완료**, 미실행 없음. 남은 건 정렬 + (정해지면) 범위확장. 모든 숫자 RESULTS §1–16.
