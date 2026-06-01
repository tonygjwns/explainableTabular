# HANDOFF: ExplainableTab 프로젝트 컨텍스트

> 이 문서는 새 탭(또는 새 사람)이 프로젝트를 이어받을 때 첫 번째로 읽는 문서입니다.
> 어떻게 이 plan에 도달했는지의 흐름을 압축적으로 담고 있습니다.

---

## 1. 프로젝트 정체성

### 우리가 풀려는 문제

표 데이터 딥러닝에서 **temporal distribution shift** (시간에 따른 데이터 분포 변화). 산업 데이터에 흔하지만, 학술 벤치마크에선 거의 무시됨. TabReD (ICLR 2025)가 이 문제를 처음 본격 부각시켰고, 그들은 "복잡한 방법(검색·사전학습)이 시간 분할에서 실패한다"는 negative result를 보였음.

### 우리의 가설

**"Drift는 외부 노이즈가 아니라 학습 가능한 신호"**. 시간에 따른 분포 정보를 **시간 인덱싱된 프로토타입 메모리**에 저장하고, 입력이 자기 시점의 어느 프로토타입에 가까운지 **검색(retrieval)**하여 예측한다. 의미 진화를 명시적으로 학습하고 해석 가능한 형태로 노출시킨다.

> ⚠️ **아키텍처 진화 주의**: 초기엔 "head에 시간 궤적"이었으나, head는 정보를 *저장*하지 않는 *변환* 구조라 부족함이 드러남. Cai 증거(출력단 시간 주입 12.6%)와 TimeMCL 분석(WTA는 시간축 아닌 출력모드로 분화)이 이를 뒷받침. **최종 구조는 head가 아니라 백본-예측기 사이의 "시간 인덱싱 메모리 + 검색" 레이어.** 상세는 EXPERIMENT_PLAN.md.

### 한 줄 메시지

> "기존 작업(Cai et al. NeurIPS 2025)은 변수 측 분포 *변조(modulation)*로 의미 일관성을 달성했으나 검색이 없다. TabR은 검색하지만 시간 좌표가 없다. 우리는 그 빈 교차점 — **시간 인덱싱된 검색** — 을 차지한다. 시간에 따라 진화하는 프로토타입 메모리에서 입력이 어느 시점-프로토타입에 가까운지 검색하며, drift를 해석 가능한 궤적으로 노출한다."

---

## 2. 어떻게 여기 도달했나 — 사고 흐름

### 출발점: PFCT 프로젝트의 유산

기존 작업 PFCT(Patch-and-Vote)에서 두 핵심 아이디어:
- **Patching**: 상관관계로 변수 묶기 (현재 폐기됨)
- **Voting Head**: 학습된 K개 프로토타입과의 유사도로 분류 (메인 기여로 격상)

PFCT 자체는 성능이 강하지 않았지만, voting head의 **해석가능성**과 **압축된 검색**이라는 개념이 본질적 가치였음.

### 논문 정독으로 그려진 풍경

시간 순서로 읽은 논문들:

| 논문 | 핵심 | 우리에 미친 영향 |
|---|---|---|
| Revisiting Deep Learning (2021) | FT-Transformer, ResNet 베이스라인 | 단순한 베이스라인도 강력함 |
| onEmbeddingsfor (2022) | PLR/PLE 수치 임베딩 | 수치 임베딩은 보편적 도움 |
| Revisiting Pretraining (2022) | Target-aware 사전학습 | 사전학습도 효과적 |
| TabR (2023) | 검색 증강 모델 | 검색의 잠재력 |
| **TabReD (2024)** | **산업 벤치마크 + 시간 분할에서 검색/사전학습 실패** | **분야 전환점. 우리 출발 동기** |
| TabM (2025) | 효율 앙상블이 SOTA | **우리 백본 선택** |
| ModernNCA (2025) | 단순한 NCA가 강한 베이스라인 | 인접 작업 |
| Benchmarking Optimizers (2026) | Muon > AdamW | 직교 개선축 |
| **Understanding Limits (Cai & Ye ICML 2025)** | **학습 프로토콜 + Fourier 시간 임베딩으로 검색 모델까지 회복** | **Phase 1-2를 이미 그들이 했음** |
| **Feature-aware Modulation (Cai et al. NeurIPS 2025)** | **변수 분포 통계량 변조로 TabReD에서 처음 GBDT 능가** | **우리의 직접 경쟁작** |
| EvolveGCN (2020) | 그래프 가중치가 시간 진화 | 메커니즘 영감, 도메인 차별화 |
| Latent ODE (2019) | 연속 시간 잠재 동역학 | 연속 시간 원리 영감 |

### 외부 검토에서 받은 비판

세 LLM 채널에 다른 역할로 plan 검토 요청 → 보완적 결함 발견:

**Gemini (방법론)**:
- Welch's t-test 잘못 사용 → Wilcoxon signed-rank (페어드 데이터)
- 다중 비교 무보정 → Benjamini-Hochberg FDR
- Sequential additive ablation의 음의 상호작용 → Leave-one-out 필수
- Smoothness가 회귀 데이터셋에 작동 불가 (TabReD 8개 중 5개가 회귀)

**Claude (동료)**:
- 세 청구(성능 + 해석가능성 + 이론적 위치) 분산 → "Pick one"
- EvolveGCN, Latent ODE 인접 작업 인용 누락
- Rolling-window 재학습 베이스라인 누락
- "프로토타입" 개념 표 데이터에서 구체화 필요

**GPT (적대적, 가장 결정적)**:
- **구조적 공격**: Voting head가 TabReD가 실패라고 한 retrieval의 본질 가정("학습 인스턴스가 미래에 유용") 답습
- TabM의 32 implicit submodel을 평균 + voting head 얹는 옵션은 TabM의 핵심(앙상블 다양성) 파괴
- 결정적 발견: **NeurIPS 2025에 직접 경쟁작 존재** (Feature-aware Modulation)

### 결정적 발견: LAMDA 그룹의 두 논문

검토 후 직접 검색해서 발견:

**Cai & Ye, "Understanding the Limits..." (ICML 2025)**:
- TabReD 결과 분해: 학습 lag + validation bias 두 결함
- Fourier 시간 임베딩으로 검색 모델까지 회복
- → 우리 Phase 1-2 거의 그대로

**Cai et al., "Feature-aware Modulation" (NeurIPS 2025)**:
- 변수 의미가 시간 따라 변함을 분석
- 평균/std/왜도를 시간 함수로 변조 (Yeo-Johnson)
- **TabReD에서 처음으로 DL이 GBDT 일관 능가**
- → 우리의 직접 경쟁작

### 최종 청구 좁힘

원래 세 청구 → 하나로 좁힘:
- ~~성능 청구~~: Cai et al.를 절대값으로 이기기 어려움
- ~~이론적 위치~~: Cai et al.가 이미 일부 입증
- **해석가능성 + 보완적 메커니즘**: 그들이 안 한 시간 인덱싱 검색(retrieval) + drift 궤적 시각화

---

## 3. 전략 결정: C → B Sanity-Check Gate

### 세 가지 옵션이 있었음

> (당시엔 "voting head"라 불렀으나, 이후 "시간 인덱싱 메모리 + 검색"으로 구체화됨. 아래 옵션의 "우리 novel 구조"가 그것.)

| 옵션 | 설명 | 평가 |
|---|---|---|
| A. 전면 폐기 | 우리 novel 구조 버리고 단순 TabM + 시간 변조만 | Cai et al.와 거의 동일 → 새로움 부족 |
| B. 격하 | 우리 novel 구조를 여러 옵션 중 하나로, 비교 연구 | 안전하지만 청구 약함 |
| C. 유지 + 방어 | 우리 novel 구조를 메인으로, 사전 검증 | 강한 결과 가능, 실패 시 큰 손실 |

### 최종 선택: C → B Gate

**전략**: C를 시도하되, 빠른 sanity check로 검증. 실패 시 B로 자연 흡수.

```
Stage 1 (3주): 공통 토대 — TabM + Cai et al. + 시간 인덱싱 메모리(최소버전) 구현
Stage 2 (3주): Sanity Check — 4가지 테스트로 PASS/FAIL 결정
Stage 3a (6-7주): PASS → C 풀 진행, 메인 청구로 시간 인덱싱 검색
Stage 3b (6-7주): FAIL → B로 pivot, 진단 paper로 재구성
```

**핵심 장점**: Stage 1+2 작업이 B에서도 그대로 재사용 → 매몰비용 거의 없음.
**Phase 1은 의도적으로 최소 버전** (단순 softmax, WTA 없음 등 — EXPERIMENT_PLAN.md §6).

---

## 4. 즉시 행동 항목 (이번 주)

1. **EXPERIMENT_PLAN.md(아키텍처), HANDOFF.md, SETUP.md, REFERENCES.md 정독**
2. **GitHub repo (`https://github.com/tonygjwns-opt/explainableTabular`) clone 및 환경 셋업**
3. **TabM 코드 받기** (`https://github.com/yandex-research/tabm`)
4. **TabReD 데이터셋 다운로드**
5. **Cai et al. NeurIPS 2025 코드 공개 여부 확인** (`https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation`)
6. **Pre-registration 문서 (`PRE_REGISTRATION.md`) 동료와 함께 commit** (특히 결정 3: WTA 보류에 대한 합의)
7. **Phase 0 시작**: Sberbank Housing 같은 작은 데이터셋에서 TabM 재현 검증

---

## 5. 새 탭/사람을 위한 핵심 주의사항

### Don'ts (하지 말 것)
- ❌ 시간을 head(출력단)에만 주입 → 메모리(주)+입력(보조)로. head는 정보 저장 못 함
- ❌ Phase 1에 WTA/annealing 넣기 (결정 3 — 토대 검증 먼저, WTA는 시간축 아닌 출력모드로 분화)
- ❌ Phase 1에 TabR 보정항·외적 게이팅 등 화려한 요소 (ablation factory 함정)
- ❌ Welch's t-test 사용 → 페어드 데이터엔 Wilcoxon signed-rank
- ❌ 메모리를 TabM의 32 submodel 평균 후 얹기 (TabM 파괴 — Phase 2 ablation으로)
- ❌ Smoothness 손실의 라벨 무시 (다른 클래스끼리 가까이 강제하지 말 것)
- ❌ Sanity check 결과 후 PASS/FAIL 기준 변경
- ❌ Sequential additive ablation만 의존 (leave-one-out 병행)

### Do's (해야 할 것)
- ✅ **시간 인덱싱 메모리가 핵심**: P_k(t) = P_k^base + drift_k(Fourier(t)) (head 아님)
- ✅ **Phase 1은 최소 버전**: 단순 softmax 검색, KMeans 초기화, L_smooth만 (EXPERIMENT_PLAN §6)
- ✅ 페어드 통계 (Wilcoxon signed-rank) + FDR 보정 (Benjamini-Hochberg) + Hedges' g
- ✅ Cai et al. NeurIPS 2025 위/옆에 우리 추가 — 그들 베이스라인 위에서 sanity check
- ✅ Label-conditional smoothness (회귀는 `|y_i - y_j| < δ`)
- ✅ Pre-registration 결과 보기 전에 동료 입회
- ✅ 프로토타입 시간 궤적 시각화 — 우리 해석가능성 청구의 핵심 증거
- ✅ Null result도 정직 보고 — 어느 결과든 publishable한 구조
- ✅ 요소는 **하나씩** 추가 (Phase 2 ablation 로드맵, EXPERIMENT_PLAN §9)

---

## 6. 결정의 흐름 요약 (한 그림)

```
PFCT (corrpatch + voting + SupCon)
  ↓ 성능 약함, 분야 재탐색
TabReD 발견 → "drift는 학습 가능한 신호" 가설
  ↓ TabM 백본 + 시간 인지 컴포넌트 점진 추가 plan
외부 검토 3채널 (Gemini, Claude, GPT)
  ↓ 통계 결함 + 청구 분산 + 구조적 공격
Cai & Ye 두 논문 발견 (직접 경쟁작) + TimeMCL(WTA) 분석
  ↓ 청구 좁힘: 해석가능성 + 보완적 메커니즘
  ↓ "head로는 시간 부족" 깨달음 (Cai 12.6%, WTA는 시간축 아닌 출력모드 분화)
팀 토론: "시간 분포 정보를 가진 메모리가 있어야" 직관
  ↓ 아키텍처 구체화: head → "시간 인덱싱 분포 메모리 + 검색"
설계 결정 5개 승인 (단순 softmax / 라벨분포 V_k / WTA 보류 / 시간 메모리+입력 / KMeans 초기화)
  ↓ C → B Sanity-Check Gate (Phase 1 최소 버전)
  ↓ Pre-registration → Phase 0 시작
```

이게 우리가 지금 있는 자리입니다. **아키텍처 상세는 EXPERIMENT_PLAN.md**, phase/통계/자원은 PLAN.md.
