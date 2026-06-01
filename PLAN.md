# PLAN: 실험 실행 계획 (Phase·일정·자원)

> ⚠️ **아키텍처는 EXPERIMENT_PLAN.md가 정본(authoritative)입니다.**
> 이 문서는 phase 구조·일정·자원·베이스라인 단계화 등 **실행 로지스틱스**를 담습니다.
> 아키텍처 상세(§3 구조), 설계 결정(§4), sanity check 기준(§8)은 중복을 피해 EXPERIMENT_PLAN.md를 가리킵니다.
> 실험 시작 전 마지막 점검은 PRE_REGISTRATION.md.

---

## 1. 메인 가설 (한 가지로 좁힘)

**H1**: Cai et al. NeurIPS 2025의 feature-aware modulation 베이스라인 위/옆에 **시간 인덱싱 프로토타입 메모리 + 검색** 구조를 추가하면, 다음 둘 중 하나가 달성된다:
- **(a)** 성능 면에서 통계적으로 유의한 추가 향상 (paired Wilcoxon p < 0.05, 8개 데이터셋 중 5개 이상)
- **(b)** 동등 성능 (no statistical degradation) + 정량 측정 가능한 해석가능성 향상 (프로토타입 궤적이 실제 분포 변화와 매칭)

조건 (a) 또는 (b) 어느 하나도 만족 못 하면 → null result로 정직 보고.

> 구조 정의: 시간에 따라 진화하는 프로토타입 메모리 `P_k(t) = P_k^base + drift_k(Fourier(t))`에서,
> 입력이 자기 시점의 어느 프로토타입에 가까운지 검색. **head가 아니라 백본-예측기 사이 메모리 레이어.**
> 상세는 EXPERIMENT_PLAN.md §1·§3.

---

## 2. 데이터셋 및 평가 프로토콜

### 데이터셋
- **TabReD 8개 (메인)**: Sberbank Housing, Homesite Insurance, Ecom Offers, HomeCredit Default, Cooking Time, Delivery ETA, Maps Routing, Weather
- **분할 방식**: Cai & Ye 2025 (ICML)의 개선된 시간 분할 (학습 lag 0, validation bias 최소화)

### 시드 정책 (차등 적용)
- Phase 0 (재현): 10 시드
- Phase 1 (sanity check): 5 시드 (빠른 결정 목적)
- Phase 2 (메인 결과): **20 시드** (통계 검정력 보장)

### 지표
- **분류**: ROC-AUC (이진), Accuracy (다중)
- **회귀**: RMSE
- **상대 지표**: MLP 대비 percentage improvement (TabM 논문 방식)
- **해석가능성 지표**: 프로토타입 궤적과 실제 분포 변화의 매칭도 (정의는 PRE_REGISTRATION.md)

### 통계 분석
- **Paired Wilcoxon signed-rank test** (시드 페어 기준)
- **Benjamini-Hochberg FDR 보정** (다중 비교)
- **Hedges' g** 효과 크기 (작은 표본 보정)
- 데이터셋이 진짜 단위, 시드는 pseudo-replicate임을 인지

---

## 3. 베이스라인 (단계적 접근)

### Phase 0 (필수, 처음부터)
1. **TabM 베이스라인 재현** — 우리 백본 검증 (1-2개 데이터셋에서 TabM 논문 수치와 ±1% 일치 확인)

### Phase 1 (Sanity check 단계)
2. **Cai et al. 2025 (NeurIPS) 재구현** — 우리 추가가 어디에 얹히는지 명확히
3. **Fixed memory (시간 무관 P_k)** — sanity check 대조군 (시간 인덱싱 효과 분리)

### Phase 2 (메인 결과, sanity 통과 후)
필요 시 추가:
4. **Cai & Ye 2025 (ICML)** — 시간 임베딩 단독 효과
5. **TabM-RW** (rolling-window 재학습, N ∈ {1, 3, 6 month}) — 산업 표준
6. **TabM-Wide** (파라미터 매칭) — capacity 효과 통제
7. **GBDT** (XGBoost, CatBoost, LightGBM) — 트리 비교

**원칙**: 이미 공개된 신뢰할 만한 베이스라인 수치는 인용 (TabReD/TabM 논문 표). 우리가 직접 재현하는 건 비교가 결정적인 것들만.

---

## 4. Phase 별 실험 계획

### Phase 0: 토대 구축 (3주)

**목표**: 작업 환경 셋업 + TabM 재현

**Sub-tasks**:
- [ ] TabM 코드 받기 (`yandex-research/tabm`)
- [ ] TabReD 8 데이터셋 다운로드 + Cai 분할 적용
- [ ] 작은 데이터셋(Sberbank Housing)에서 TabM 재현
- [ ] 재현 수치가 TabM 논문 ±1% 이내 확인
- [ ] (병행) Cai et al. NeurIPS 2025 코드 공개 여부 확인. 미공개 시 직접 구현

**통과 기준**: TabM 재현 성공 + Cai et al. 베이스라인 작동.

---

### Phase 1: 시간 인덱싱 메모리 + 검색 (최소 버전) 구현 + Sanity Check (3주)

**목표**: 시간 인덱싱 프로토타입 메모리 + 검색의 **최소 버전**을 TabM + Cai 백본 위에 구현하고,
메인 가설을 빠르게 검증.

**Sub-tasks**:

1. **최소 버전 구현** — 아키텍처 상세는 **EXPERIMENT_PLAN.md §6** 참조.
   요약: `P_k(t) = P_k^base + drift_MLP_k(Fourier(t))`, K=1000, 시간 슬라이스 KMeans 초기화,
   단순 softmax 검색 (**WTA·TabR 보정항·외적 게이팅은 Phase 1에 없음** — 결정 3),
   `V_k = 라벨분포 임베딩 + 학습벡터`, 손실 = L_main + λ·L_smooth.

2. **Sanity Check 실행**: 4 데이터셋, 5 시드, K=1000 (빠른 결정 목적).

3. **테스트와 PASS/FAIL 기준**: **EXPERIMENT_PLAN.md §8 + PRE_REGISTRATION.md가 정본.**
   - Test 1: 시간 인덱싱 메모리 vs 고정 메모리 (성능)
   - Test 2: 외삽 검증 (학습 70% → 미래 30% 분포 예측)
   - Test 3: 검색 의미성 + 궤적 시각화 (해석가능성)
   - Test 4: 시간 주입 위치 비교 (메모리/입력/둘 다 — 비-게이팅 진단)

**의사결정**: Test 1·2·3 모두 통과 → PASS(Stage 3a). 하나라도 미달/모호 → FAIL(Stage 3b).

**중요**: PASS/FAIL 기준은 실험 시작 전 PRE_REGISTRATION.md에 git commit. 결과 본 후 변경 금지.

---

### Phase 2a: PASS → 풀 평가 (6-7주)

**목표**: 시간 인덱싱 메모리 + 검색을 메인 청구로 풀 검증
(이 단계부터 EXPERIMENT_PLAN.md §9 ablation 로드맵 요소를 하나씩 추가)

**Sub-tasks**:
- [ ] 8 데이터셋 전체, 20 시드 학습
- [ ] 메인 비교: 우리 vs Cai et al. NeurIPS 2025
- [ ] **Leave-one-out ablation** (각 컴포넌트 제거 시 영향)
- [ ] **EXPERIMENT_PLAN §9 요소를 하나씩 추가**: TabR 보정항 / annealed WTA(aMCL) / 다단계 시간 주입 / K 스케일 / TabM 32 결합
- [ ] **해석가능성 분석**:
  - 학습된 프로토타입 궤적 PCA/UMAP 시각화 (전 데이터셋)
  - **1-2 데이터셋에서 정성 case study**:
    - 예: HomeCredit Default에서 "신용 위험 프로토타입"이 코로나 전후 어떻게 이동했는지
    - 가장 빠르게 움직인 프로토타입과 실제 분포 변화 매칭
- [ ] 통계 분석 (Wilcoxon + BH 보정 + Hedges' g)
- [ ] 필요 시 추가 베이스라인 (GBDT, TabM-RW 등)

---

### Phase 2b: FAIL → 진단 paper로 pivot (6-7주)

**목표**: "다양한 drift 모델링 메커니즘의 진단적 비교"로 재구성

**Sub-tasks**:
- [ ] 청구 재정의: "어떤 메커니즘이 어떤 데이터에서 효과 있는가"
- [ ] 비교 확대: Cai (ICML), Cai (NeurIPS), 우리 시간 인덱싱 메모리, 고정 메모리, FiLM-only 변형 등
- [ ] **메인 분석**: 데이터셋 특성과 메커니즘 효과의 매칭
  - 어느 데이터셋이 어느 방법으로 가장 큰 향상?
  - 향상 패턴에서 일반화 가능한 lesson 추출
- [ ] **정직한 결론**: 시간 인덱싱 검색이 무엇을 못 하는지, 왜인지
- [ ] Workshop 또는 short paper 형식 고려

---

### Phase 3: 글쓰기 (3주)

**Sub-tasks**:
- [ ] 논문 draft
- [ ] 그림/표 정리
- [ ] 동료 review
- [ ] 제출

---

## 5. 자원 및 일정 요약

```
주차    Phase                기간   누적
─────────────────────────────────────────
1-3     Phase 0             3주    3주
4-6     Phase 1 (sanity)    3주    6주
7-13    Phase 2a 또는 2b   6-7주  12-13주
14-16   Phase 3 (글쓰기)    3주    15-16주
```

**자원**: H100 ×2 (독립 병렬 실험, DDP 아님)

**예상 GPU-시간**:
- Phase 0: ~50 GPU-시간 (TabM 재현)
- Phase 1: ~80 GPU-시간 (sanity check)
- Phase 2: ~250 GPU-시간 (메인 평가)
- 총 ~380 GPU-시간 → wall-clock 약 8주 (×2 병렬 기준)

---

## 6. 위험 및 완화

| 위험 | 가능성 | 완화 |
|---|---|---|
| Sanity check FAIL | 50% | Stage 3b로 자연스럽게 pivot. 매몰비용 거의 없음. |
| Cai et al. 코드 미공개 | 30% | 우리 직접 구현 + 그들 표 수치와 비교 |
| TabM 재현 실패 | 10% | TabM 저자에게 issue 또는 이메일 |
| 시간 분포 변화가 데이터셋 너무 다양 | 50% | 데이터셋 그룹화 분석, 단순화 lesson 추출 |
| Null result | 30% | 정직 보고. "When does time-indexed retrieval help? Not always"가 의미 있는 메시지 |

---

## 7. 메인 청구 (글쓰기용)

```
Title 후보:
  "Time-Indexed Prototype Memory: Retrieval-Based Drift Modeling for Tabular Learning"
  또는 "Decision-Level Temporal Retrieval for Drift-Aware Tabular Learning"

Abstract 핵심:
  - 시간 분포 변화 + 표 데이터 문제 의식
  - Cai et al.(변수 측 변조)과 TabR(무시간 검색) 사이의 빈 교차점
  - 시간 인덱싱 프로토타입 메모리에서 입력이 자기 시점 어느 프로토타입에 가까운지 검색
  - 해석가능성: drift를 프로토타입 궤적으로 시각화 가능하게 노출
  - 결과: Cai et al. 위에 추가 향상 OR 동등 성능 + drift case study

Contribution:
  1. Time-indexed retrieval mechanism (Cai의 변조도 TabR의 무시간 검색도 안 한 교차점)
  2. Interpretable drift artifacts (prototype 궤적)
  3. Empirical evaluation on TabReD with rigorous methodology
```

---

## 8. 다음 단계 체크리스트

- [ ] PRE_REGISTRATION.md 완성 + 동료 입회 commit
- [ ] SETUP.md 따라 환경 구축
- [ ] Phase 0 시작
- [ ] 매주 진척 추적 (별도 progress.md 권장)
