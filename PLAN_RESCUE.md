# PLAN_RESCUE — 사전등록 go/no-go (시간-인덱싱 메모리, 마지막 공정 기회)

> REVIEW.md(특히 §9)의 진단 위에서, **버리기 전 메커니즘에 공정한 기회**를 주는 bounded
> 계획. 3라운드 리뷰 피드백 반영. **floor = §6(다) 현상-분석**(강한 covariate는 이미 단단).

## 0. 청구 상태
- **확정**: 강한 covariate drift(시점분류기 AUC≈1.0; robust) · 구현 정확(합성 +87%) ·
  정적 in-dist 동률(underpowered; redundancy는 *허용*이지 증명 아님 — 표현적 우위 부재만).
- **미결(구제 대상)**: **Q1 충실성**(메인 청구, 한 번도 측정 안 함) · **Q2 외삽 우위**(기저 confound).
- **경고(F3)**: AUC≈1.0 = early/late x 공통 support 희박 → "x 고정·t 변경" concept 측정이
  **ill-posed일 수 있음**. "concept 없음"과 "측정 불가"를 혼동하면 안 됨.

---

## A. 즉시 (병렬, 싸고 결정적) — 지금 잠그고 실행

### Q1 — 기능적 충실성 (합성) [주 게이트] — 최종 프로토콜
- **세팅**: 합성 **이진** `y=sign(x·w(t)+noise)` (V_k=라벨임베딩 설계·Elec2와 정합) +
  **memory_only**(메모리 사용 강제) + **load-balance anti-collapse**(계수 최소+스윕) + **trend 기저**.
  학습 V_k 허용(시간은 drift_k로만 흐름; V_k·τ는 t-비의존 → 기능 충실성 격리).
- **★합성 drift는 trend-표현가능하게**(기저-불일치 confound 제거가 최우선): `w(t)` 각도를
  **0→π/2 단조 회전(≤반주기)** 또는 저차 다항 경로로. (전주기 cos는 deg-3가 못 담아 *충실해도 FAIL*.)
  → 게이트는 이 **trend-정합판**. periodic 합성+Fourier 기저(정합) 조합은 *충실성 기저-불변* robustness로 별도 1회.
- **주지표(기능공간, 게이지-고정 직접비교)**: `ŵ(t)=E_x[∂ŷ/∂x | t]`.
  recovery = t-그리드에서 **per-t `cos(ŵ(t), w(t))`의 집계(평균)**. ŵ와 w는 *같은 입력공간*
  그래디언트라 게이지가 이미 고정 → **Procrustes/CCA(자유회전) 금지**(false PASS·기하자유도 부활).
  **로짓이 x에 선형**(`logit=x·w(t)`, 이진 라벨) → ∂(logit)/∂x가 x-무관하게 `w(t)` 방향 → ŵ(t) 명확.
  (이질 그래디언트 비선형 합성은 게이트 후 robustness.)
- **보조지표(읽기성, 게이트 아님)**: z-공간 프로토타입 궤적 vs `span{w1,w2}` Procrustes/CCA.
- **임계 θ₁ (상대정의)**: 오라클 천장 hi(진짜 w(t) 쓰는 완벽/ time-feature MLP) + 셔플-t 바닥 lo
  먼저 측정. 두 기준선: `PASS선 = lo + 0.7·(hi−lo)`, `FAIL선 = lo + 0.4·(hi−lo)`.
- **결정규칙(칼날 대신 경계대; 오차 비대칭 — false PASS가 false FAIL보다 나쁨)**:
  - **PASS**: recovery의 **하한 신뢰구간 ≥ PASS선** → 충실성 PASS(필요조건). (점추정 0.69/0.71 다툼 제거)
  - **FAIL→(다)**: 점추정 < FAIL선 → 메인 청구 사망.
  - **[FAIL선, PASS선] 또는 CI 넓음**: 하드판정 아님 → 싼 후속(noise·난이도 스윕, 시드↑)으로 해소(합성이라 비용≈0).
  - (mem_gap은 memory_only에서 공허 → 게이트 제외.)
  - **CI는 시드-간이 주**: 하한을 **10시드 recovery 분포** 위(또는 "≥8/10 시드 PASS선 통과");
    t-그리드 부트스트랩은 보조(학습/init 노이즈를 못 잡음).
  - **진단 플롯**: `recovery(t)` 그리드 전체 — t→1 붕괴는 Q2 외삽 곤란의 예고편.
  - **load-balance 계수 스윕**: 강계수 FAIL은 정규화 아티팩트일 수 있으니 recovery(계수) 곡선 확인.
- **별도 진단(게이트 아님)**: concat 모드에서 z-탈출구가 있어도 메모리를 쓰는가 →
  배포용 concat이 z-지름길로 회귀하면 *불충실*(충실성⟂성능 모순)임을 열어둠.

### F3 — concept 측정-가능성 feasibility 프로브 [지금, Q1과 병렬]
- **질문**: AUC≈1.0 제약하에서 *어느 데이터셋이든* 충분한 공통 support로
  covariate-조정 concept을 잴 수 있는가?
- **방법**: early/late(median t) 분류기(HGB) → held-out `P(late|x)`. 측정 3종:
  **overlap mass**=`P∈[0.1,0.9]` 비율(+`[0.2,0.8]` 민감도 동반보고), **IW ESS**=`(Σw)²/Σw²`,
  **라벨-support**=overlap∩early / overlap∩late 각각의 (소수클래스) 사건수.
- **측정가능 판정**: overlap mass ≥ τ_m(=5%, 스크린) **AND** ESS ≥ N_m(스크린 500; 실측엔 1000~2000)
  **AND** 시간반쪽별 overlap 사건수 ≥ 하한. (covariate overlap이 있어도 불균형(homecredit 저부도율)이면
  overlap 양쪽에 양성이 거의 없어 P(y|x)를 못 잼.)
- **결정**: 측정가능 데이터셋 0개 → **Q2 데이터 게이팅 전제 사망 → (Q1 무관) §6(다) 가중 즉시 상향.**
  Q2는 공통 support 데이터(Elec2/Insects/Airlines) 중심.

---

## B. 게이트 통과 후에만 (Q1 PASS **그리고** F3 feasibility OK) — 지금 확정하지 않음

### Q2 — 외삽 우위 [큰 빌드, 연기]
- **메커니즘 F2(기본)**: **(b) 시간-조건 유사도, trend로 parametrize(외삽 가능).**
  ✗(a) `|t_q−t_i|` recency 커널 — 외삽서 모든 학습 이웃이 "오래됨" → 신호 소멸(서식지는 §6가 온라인).
- **요인설계**: 구조축 **3단계 {MLP+t, TabR(무시간), 시간-TabR}** × {기저: fourier, trend} × λ;
  **baseline의 t 인코딩을 같은 기저로 정합.** → 시간 기여 vs 비파라메트릭 검색 기여 분리.
- **데이터 F4**: Elec2/Insects/Airlines 우선(공통 support), TabReD 보조.
  Insects(abrupt/gradual/incremental)로 {기저}×{drift유형} 귀속. **스트리밍 출신 → 정적 split
  프로토콜 불일치(§4.1) 명시.** 3~4개, ≥2개는 *측정된* concept 보유.
- **검정력 F5**: n=10은 g≈0.6+만 신뢰 탐지 → **시드 25~30 / 또는 임계 g≥0.5 / 또는 단측** 중
  택1 사전명시(modest-real 효과를 놓쳐 조기 (다)行 방지).
- **결정규칙**: 시간-TabR(trend)이 기저-정합 baseline을 외삽에서 ≥2 데이터셋, 잠근 (g,p,power)로
  이김 → 생존. 아니면 → redundancy 폐쇄가 외삽까지 확장 → §6(다).

---

## C. 폴백 & 메타
- **floor = §6(다)**: "표 시간 벤치 = 강한 covariate, 시간방법 underdeliver" — 이미 단단.
  모든 부분결과가 여기로 흘러듦.
- **메타 플래그**: 메커니즘의 자연 이점이 recency/적응이면 서식지는 **§6(가) 온라인**이지
  정적 외삽이 아님 — Q2가 그쪽으로 기울면 프레이밍 재고.

## D. 잠금 상태
- **지금 잠금**: Q1 스펙(기능 주지표 + 상대 θ₁), F3 feasibility 프로브.
- **게이트 후 확정**: F2 최종, F4 최종, F5 검정력/임계.

## E. 사전 결정규칙 요약 (사후 재litigation 방지)
| 게이트 | PASS 조건 | FAIL 시 |
|---|---|---|
| Q1 충실성 | functional recovery ≥ θ₁(상대) | 메인 청구 사망 → (다) |
| F3 feasibility | ≥1 데이터셋서 support 있는 concept 측정 가능 | Q2 데이터전제 사망 → (다) 상향 |
| Q2 외삽 | 시간-TabR(trend) > 정합 baseline, ≥2 데이터셋, 잠근 검정력 | redundancy 외삽 확장 → (다) |
