# REFERENCES: 주요 논문과 우리와의 관계

> 논문 정독 결과를 새 탭에서도 참조할 수 있도록 정리.
> 각 논문에 대해: 핵심 메시지, GitHub, 우리에게 미친 영향.

---

## A. 직접 경쟁작 (반드시 인용 + 차별화)

### A1. Feature-aware Modulation for Learning from Temporal Tabular Data
- **저자**: Hao-Run Cai, Han-Jia Ye (LAMDA, NJU)
- **발표**: NeurIPS 2025
- **arXiv**: 2512.03678
- **GitHub**: https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation

**핵심**:
- 변수의 의미가 시간 따라 변함 (객관적 vs 주관적 semantic)
- 분포 통계량 (평균/std/왜도)을 시간 함수로 변조 (Yeo-Johnson 비선형)
- 3-stage 변조 (입력/중간/출력) — 입력 단이 87.4% 효과
- TabM + 그들의 변조 = **TabReD에서 처음으로 DL이 GBDT 일관 능가**

**우리 관계**:
- 직접 경쟁작
- 우리가 그들 위/옆에 **시간 인덱싱 메모리 + 검색**을 얹어 추가 가치 입증해야
- 우리 차별화: 그들은 변수 측 **변조(modulation)**, 우리는 **검색(retrieval)** + 시간 인덱싱 + 해석가능성

### A2. Understanding the Limits of Deep Tabular Methods with Temporal Shift
- **저자**: Hao-Run Cai, Han-Jia Ye (LAMDA, NJU)
- **발표**: ICML 2025
- **GitHub**: (논문에서 명시 안 됨, 위 NeurIPS 코드에 흡수된 듯)

**핵심**:
- TabReD 시간 분할의 두 결함 분해: 학습 lag + validation bias
- Fourier 시간 임베딩 (yearly/monthly/weekly/daily 주기 prior + 선형 추세)
- 두 기여로 검색 기반 모델까지 회복 (TabR -1.32% → +0.49%)

**우리 관계**:
- 우리 Phase 1 (Time2Vec PE), Phase 2 (sliding window 학습)와 거의 일치
- 그들 학습 프로토콜 + Fourier 임베딩을 **우리 베이스라인의 일부로** 사용
- "Phase 1-2는 그들 contribution" 명시해야

---

## B. 직접 빌딩 블록 (우리 백본/방법)

### B1. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
- **저자**: Yury Gorishniy et al. (Yandex/HSE)
- **발표**: ICLR 2025
- **GitHub**: https://github.com/yandex-research/tabm

**핵심**:
- MLP 32개의 효율 앙상블 (BatchEnsemble 변형)
- 가중치 대부분 공유 + 작은 adapter (r, s, b)로 멤버 차별화
- 한 forward pass에서 32 predictions 동시 생성
- 개별 submodel은 weak, 평균은 strong

**우리 관계**:
- **우리 백본**. 모든 phase가 TabM 위에서.
- 메모리+검색 레이어 통합 시 주의: 32 submodel 평균하지 말 것 (TabM 핵심 파괴)
- Phase 1은 단일 백본 표현으로 시작, **Phase 2 ablation**으로 통합 방식 결정 (submodel별 메모리 vs 공유 메모리 + submodel별 query 등)

### B2. TabReD: Analyzing Pitfalls and Filling Gaps in Tabular DL Benchmarks
- **저자**: Ivan Rubachev et al. (Yandex/HSE)
- **발표**: ICLR 2025
- **GitHub**: https://github.com/yandex-research/tabred

**핵심**:
- 산업 표 데이터 + 시간 분할 + 변수 풍부 (평균 261개)
- 8개 데이터셋
- Negative result: 검색·사전학습 방법들이 시간 분할에서 무너짐
- 단순 MLP + 임베딩, GBDT가 살아남음

**우리 관계**:
- **우리 메인 벤치마크**. 모든 평가는 TabReD 8 데이터셋.
- 우리 연구의 출발 동기 (negative result에 대한 도전)

---

## C. 인접 작업 (인용 + 차별화)

### C1. EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs
- **저자**: Aldo Pareja et al. (MIT-IBM Watson)
- **발표**: AAAI 2020
- **GitHub**: https://github.com/IBM/EvolveGCN

**핵심**:
- GCN의 **가중치 행렬 W**가 시간 따라 RNN으로 진화
- 두 변형: -H (W가 hidden state), -O (W가 output)
- 모델 자체가 적응하므로 노드 변화에 견고

**우리 관계**:
- 메커니즘이 가장 유사 (model evolution over time)
- 그러나 도메인 (그래프 vs 표), 진화 대상 (가중치 vs 프로토타입), 시간 표현 (이산 vs 연속)에서 모두 다름
- 인용 + 차별화 필요

### C2. Latent ODEs for Irregularly-Sampled Time Series
- **저자**: Yulia Rubanova, Ricky Chen, David Duvenaud (Toronto)
- **발표**: NeurIPS 2019
- **GitHub**: https://github.com/YuliaRubanova/latent_ode

**핵심**:
- RNN의 hidden state를 Neural ODE로 일반화
- 관찰 사이의 state도 연속적으로 정의
- 외삽 가능 (학습 안 본 시점도)

**우리 관계**:
- 연속 시간 잠재 동역학의 원리 영감
- 우리는 ODE solver 없이 파라미터적 trajectory만 사용 → 가볍고 더 해석 가능
- 인접 작업으로 인용

### C3. TimeMCL: Winner-takes-all for Multivariate Probabilistic Time Series Forecasting
- **저자**: Cortés, Rehm, Letzelter (Télécom Paris / Valeo.ai)
- **발표**: ICML 2025
- **GitHub**: https://github.com/Victorletzelter/timeMCL

**핵심**:
- Multiple Choice Learning + Winner-Takes-All(WTA): K개 head가 backbone 공유, 승자만 backprop
- 이론적으로 각 head = 자기 Voronoi cell의 centroid → 조건부 K-Means (gradient 기반)
- dead head 문제 → aMCL annealing(softmin 온도 감소)으로 완화
- score head로 추론 시 어느 head 승자인지 예측

**우리 관계 (중요한 주의)**:
- 팀원이 제안한 "WTA prototype" 메커니즘의 출처
- **그러나 WTA는 출력 분포 모드로 분화하지 시간축으로 분화하지 않음** → 시간은 P_k(t)의 t 입력이 담당
- **Phase 1에선 WTA를 쓰지 않음 (결정 3)**. 프로토타입 전문화가 부족하다는 증거가 나오면
  Phase 2에서 aMCL annealing만 차용 (dead prototype 방지)
- centroid 해석은 우리 해석가능성 청구의 이론적 우군

---

## D. 분야 맥락 (배경 인용)

### D1. Revisiting Deep Learning Models for Tabular Data
- **저자**: Yury Gorishniy et al.
- **발표**: NeurIPS 2021
- **GitHub**: github.com/yandex-research/tabular-dl-revisiting-models

**의미**: 표 데이터 DL 분야의 베이스라인 정리. FT-Transformer 제안.

### D2. On Embeddings for Numerical Features in Tabular Deep Learning
- **저자**: Yury Gorishniy et al.
- **발표**: NeurIPS 2022

**의미**: PLR/PLE 수치 임베딩이 보편적 도움. 우리도 TabM과 함께 사용.

### D3. Revisiting Pretraining Objectives for Tabular Deep Learning
- **저자**: Ivan Rubachev et al.
- **발표**: NeurIPS 2022 (워크샵)

**의미**: Target-aware 사전학습이 효과적. 우리는 사전학습은 안 함 (TabReD가 시간 분할에서 사전학습 실패라고 보임).

### D4. TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023
- **저자**: Yury Gorishniy et al.
- **발표**: ICLR 2024
- **GitHub**: github.com/yandex-research/tabular-dl-tabr

**의미**: 검색 기반 표 데이터 DL — **우리 검색 메커니즘의 직접 선조**. 단, 시간 좌표가 없어 TabReD 시간 분할에서 실패. 우리는 그 검색에 **시간 인덱싱**을 추가해 빈자리를 메움. (검색 모듈의 유사도 `s = -‖k - k_i‖²`, 값 모듈 `V = W_y(y_i) + T(...)`가 우리 설계의 참고. 단 우리 검색 대상은 인스턴스가 아닌 시간 인덱싱 프로토타입.)

### D5. Revisiting Nearest Neighbor for Tabular Data (ModernNCA)
- **저자**: Han-Jia Ye et al. (LAMDA)
- **발표**: ICLR 2025
- **GitHub**: https://github.com/qile2000/LAMDA-TALENT

**의미**: NCA를 modern DL 기법으로 강화. TabR보다 단순하고 빠른 SOTA. 같은 LAMDA 그룹 (이 사람들이 시간 분포 변화 연구 라인 주도).

### D6. Benchmarking Optimizers for MLPs in Tabular Deep Learning
- **저자**: Yury Gorishniy et al.
- **발표**: 2026
- **GitHub**: github.com/yandex-research/tabular-dl-optimizers

**의미**: Muon > AdamW. 우리 학습에 적용 가능한 직교 개선.

---

## E. 인용 그래프 시각화 (논문 쓸 때 도움)

```
                    Revisiting DL (2021)
                         |
                         v
                  PLR/PLE (2022)
                  /            \
                 v              v
        Pretraining (2022)    TabR (2023)
                                  |
                                  v
                              TabReD (2024) ←── 우리 출발 동기
                                  |
                                  v
                              TabM (2025) ←── 우리 백본
                                  |
                                  v
                  ModernNCA (2025), Optimizers (2026)
                                  |
              ┌───────────────────┴───────────────────┐
              v                                       v
    Cai & Ye ICML 2025          Cai et al. NeurIPS 2025
    (학습 프로토콜 + Fourier)    (Feature modulation = 변조)
              └───────────────────┬───────────────────┘
                                  v
              우리 작업: 시간 인덱싱 프로토타입 메모리 + 검색
              (Cai의 변조와 TabR의 무시간 검색 사이 빈 교차점)
              + Cai 베이스라인 위/옆에 build

  TabR (2024) ───── 검색 메커니즘 선조 (시간 좌표만 추가)
  EvolveGCN (2020) ─┐
  Latent ODE (2019) ├─→ 시간 진화 메커니즘 인접 작업 인용
  TimeMCL (2025) ───┘   (WTA centroid — Phase 1엔 미사용, Phase 2 aMCL만 차용)
```

---

## F. 인용 우선순위 (글쓰기 시)

**Abstract에서 언급할 핵심 인용**:
1. TabReD (negative result 출발점)
2. TabM (백본)
3. Cai et al. NeurIPS 2025 (직접 경쟁작)

**Related Work 메인 인용**:
- 카테고리 1: 표 데이터 DL (Revisiting, TabM, ModernNCA, FT-T 등)
- 카테고리 2: 시간 분포 변화 (TabReD, Cai 두 논문, Wild-Time)
- 카테고리 3: 시간 진화 메커니즘 (EvolveGCN, Latent ODE, TimeMCL)
- 카테고리 4: 검색 기반 / 프로토타입 (TabR, ModernNCA, PFCT line)

**Method 섹션 인용**:
- Fourier 시간 임베딩 (Cai & Ye ICML 2025)
- BatchEnsemble (Wen et al. 2020) — TabM의 토대
- 검색/유사도 모듈 (TabR — 인스턴스 검색을 프로토타입 검색으로 변형)
- WTA centroid / aMCL annealing (TimeMCL — Phase 2 전문화 시)
- Prototype 학습 (ProtoPNet 등 해석가능 프로토타입 계열)

---

## G. 우리 작업의 narrative arc

논문 쓸 때 이런 흐름:

1. **문제**: 산업 표 데이터의 시간 분포 변화 + TabReD가 보인 negative result
2. **기존 접근의 한계**:
   - 정적 모델은 적응 불가
   - 적응 모델(Cai et al.)은 변수 측 **변조**만 — 검색이 없음
   - 검색 모델(TabR)은 시간 좌표가 없음 → 시간 분할에서 실패
3. **우리 통찰**: 시간에 따른 분포 정보를 **메모리에 저장**하고, 입력이 자기 시점의
   어느 프로토타입에 가까운지 **검색**한다. (head=변환이 아니라 memory=저장+retrieval=조회)
4. **우리 방법**: 시간 인덱싱 프로토타입 메모리 `P_k(t)` + 단순 softmax 검색
5. **결과**:
   - Cai et al. 위에 추가 향상 OR
   - 동등 성능 + drift 해석가능성 증거 (프로토타입 궤적 시각화)
6. **차별화**:
   - Cai et al.: 같은 도메인, **변조 vs 검색** + 시간 인덱싱
   - TabR: 검색 같지만 **시간 좌표 없음** (우리가 추가)
   - EvolveGCN: 다른 도메인(그래프), 다른 진화 대상(가중치), 이산 RNN vs 연속 Fourier
   - Latent ODE: 다른 문제(시계열 모델링), ODE solver vs 파라미터적
   - TimeMCL: WTA는 출력 모드로 분화(시간축 아님) — 우리는 t 입력으로 시간 처리
7. **함의**: 시간 분포 변화 대응 = "변조(alignment) vs 검색(retrieval)" + 시간 인덱싱.
   우리는 "시간 인덱싱된 검색"이라는 빈 교차점을 정의.

---

이 문서는 새 탭에서 reference 검색할 때 첫 stop. 더 깊은 정보는 원 PDF를 `~/Desktop/PFCT논문들/` 폴더에서 확인.
