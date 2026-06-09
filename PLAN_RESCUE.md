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

### Q1 — 기능적 충실성 (합성) [주 게이트]
- **세팅**: 합성 `y=x·w(t)`(회전) + **memory_only**(메모리 사용 강제) + 붕괴수정 + **trend 기저**.
  학습 V_k 허용(시간은 drift_k로만 흐름; V_k·τ는 t-비의존 → 기능 충실성 격리).
- **주지표(기능공간, 인코더-불변)**: `ŵ(t)=E_x[∂ŷ/∂x | t]`. recovery = t-그리드에서
  `ŵ(t)` 궤적과 `w(t)` 궤적의 회전 정렬(평균 cosine 또는 rotation-CCA R²).
- **보조지표(읽기성)**: z-공간 프로토타입 궤적 vs `span{w1,w2}` Procrustes. *게이트 아님*(해석 품질).
- **임계 θ₁ (상대정의)**: 오라클 천장 R²_hi(진짜 w(t) 쓰는 완벽 모델 / time-feature MLP) +
  셔플-t 바닥 R²_lo 먼저 측정 → `θ₁ = R²_lo + 0.7·(R²_hi − R²_lo)`.
- **결정규칙**: recovery ≥ θ₁ → **충실성 PASS(필요조건)**. 미달 → **메인 청구 즉사 → §6(다).**
  (mem_gap은 memory_only에서 공허 → 게이트에서 제외.)
- **별도 진단(게이트 아님)**: concat 모드에서 z-탈출구가 있어도 메모리를 쓰는가 →
  배포용 concat이 z-지름길로 회귀하면 *불충실*(충실성⟂성능 모순)임을 열어둠.

### F3 — concept 측정-가능성 feasibility 프로브 [지금, Q1과 병렬]
- **질문**: AUC≈1.0 제약하에서 *어느 데이터셋이든* 충분한 공통 support로
  covariate-조정 concept을 잴 수 있는가?
- **방법**: 후보별 early/late covariate overlap 점검(IW 밀도비 꼬리/유효표본수, 또는 매칭쌍 수).
  overlap 있는 곳에서만 covariate-조정 교차시점 초과손실 측정; 없으면 **측정불가**로 표기.
- **결정**: 어디서도 support 있는 concept 측정 불가 → **Q2 데이터 게이팅 전제 사망 →
  (Q1 결과와 무관하게) §6(다) 가중 즉시 상향.** Q2는 공통 support 데이터(Elec2/Insects/Airlines)로.

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
