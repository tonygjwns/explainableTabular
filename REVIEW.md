# REVIEW — 프로젝트 전체 비판적 리뷰 (신규 멤버 온보딩 겸)

> 이 문서는 **새 멤버가 프로젝트를 처음부터 이해**하고, 동시에 **우리가 도달한
> (대체로 음성인) 결론을 *신선한 눈으로 재검토*** 하도록 돕는 것이 목적입니다.
> 결론을 방어하려는 게 아니라 **틀린 곳을 찾아달라**는 초대입니다.
> 더 깊은 일자별 로그는 `progress.md`, 증거 요약은 `FINDINGS.md` 참고.

---

## 0. 한 장 요약 (TL;DR)

- **하려던 것**: 표 데이터의 시간 분포 변화(temporal drift)를, **시간에 따라
  움직이는 프로토타입 메모리 + 검색(retrieval)** 으로 다룬다. TabM 백본 위에 얹음.
  메인 청구는 *성능*이 아니라 *해석가능성*이라고 (문서상) 적혀 있었음.
- **결론(현재)**: **제안한 메커니즘은 예측기로서 정당화되지 않는다.**
  - 합성(통제) 데이터에선 작동 → **구현은 정확함**(버그 아님).
  - 그러나 실데이터에서 (a) TabReD는 잡을 concept 드리프트가 없고(covariate-only),
    (b) Elec2는 시간이 유효하지만 **"시간을 그냥 입력 피처로 주는 것"이 메모리보다
    낫고**, 메모리는 외삽에서 오히려 해롭다.
  - 근본 이유: **시간을 입력으로 받는 신경망은 임의의 시간 의존성을 표현**할 수 있어,
    "시간을 *추가*하는" 구조가 표현적 우위를 갖기 어렵다.
- **그래서 지금**: 아키텍처 패치가 아니라 **연구 질문 재정식화**가 필요한 지점.
  §6에 후보들.
- **새 멤버에게 바라는 것**: §4(왜 안 됐나)·§5(무엇을 믿어도 되나)·§6(다음 질문)을
  비판적으로 읽고, **우리가 놓친 대안 해석이나 결함**을 지적해 주세요.

---

## 1. 배경 & 원래 가설 + 용어집

### 1.1 문제
표 데이터(tabular)는 시간이 지나며 분포가 변한다(temporal distribution shift).
TabReD는 이런 *현실적 시간 분할* 벤치마크인데, **단순 MLP/GBDT가 이기고 화려한
딥러닝/시간-인지 방법이 잘 안 통한다**는 것이 알려져 있다.

### 1.2 우리의 베팅 (원래 가설)
- 백본 **TabM** 위에, **시간-인덱싱 프로토타입 메모리** `P_k(t) = P_k^base +
  drift_k(Fourier(t))` 를 두고, 입력 표현 z가 *자기 시점의 어느 프로토타입에
  가까운지* 검색(softmax)하여 예측에 쓴다.
- 동기: "Cai et al.= 시간 *변조*, TabR= *검색*(시간 없음) → **시간-인덱싱된 검색**은
  아무도 안 한 빈 교차점" 이라는 *방법-공간의 빈칸* 논리.
- 메인 청구: **해석가능성**("이 입력은 2021Q3 프로토타입 #7에 가까웠고, 그게 이 방향으로
  이동") — 성능은 동등이면 족하다고 봄.

### 1.3 용어집 (새 멤버용)
- **Covariate drift**: 입력 분포 P(x)가 시간에 따라 변함. (예측 규칙은 그대로일 수 있음)
- **Concept drift**: 예측 규칙 P(y|x)가 시간에 따라 변함. ← *시간 모델이 도울 수 있는 곳*
- **time-indexed memory**: 프로토타입(대표 벡터)들이 시간 t의 함수로 위치가 변함.
- **retrieval**: 입력 표현 z를 프로토타입들과 비교(softmax(-‖z-P_k‖²/τ))해 가중합.
- **mem_gap**(우리 진단 지표): `loss(메모리 끔) - loss(메모리 켬)`. >0이면 모델이
  메모리를 *실제로 쓴다*, ≈0이면 *장식*(z만으로 푼다 = "z-지름길").
- **predictor_mode**: `concat`([z;메모리]) / `memory_only`(메모리만) / `residual`(z헤드+메모리헤드).
- **inject_time_input**: 시간을 *입력 피처로도* 넣을지 토글. (off면 시간은 메모리로만 들어옴)

---

## 2. 무엇을 만들었나 + 코드맵

### 2.1 아키텍처 (Phase1Model)
```
x ─(+선택적 Fourier(t) 입력주입)─► TabM.encode (32 submodel 평균) ─► z (B,d)
t ─► P_k(t)=P_base+drift_k(Fourier(t))   (K=1000 프로토타입)
검색: w_k = softmax(-‖z-P_k(t)‖²/τ)          (단순 softmax, WTA 없음)
집계: aggregated = Σ_k w_k·V_k,  V_k=라벨임베딩+학습벡터
예측: predictor_mode에 따라 [z;aggregated] / aggregated / z헤드+메모리헤드
손실: L_main + λ·L_smooth(궤적 매끄러움)
```

### 2.2 코드맵 (직접 탐색용)
- `src/models/`: `temporal_embedding`(Fourier t), `prototype_memory`(P_k(t), drift),
  `value_module`(V_k), `retrieval`(검색+예측, ablate/모드), `tabm_wrapper`(TabM),
  `phase1_model`(조립), `proto_init`(시간슬라이스 KMeans init)
- `src/training/`: `trainer`(Phase0), `phase1_trainer`(Phase1, 진단 훅), `diagnostics`(mem_gap 등)
- `src/analysis/`: `drift_measure`(covariate/concept 측정), `retrieval_trajectory`(Test3),
  `extrapolation`(Test2)
- `src/data/`: `tabred_loader`, `splits`(Cai resplit), `elec2_loader`
- `scripts/`: `run_phase0`, `run_phase1_sanity`(Test1, --predictor-mode), `run_test2/3`,
  `run_drift`, `run_conceptdrift`, `run_synth_control`(합성 양성대조), `run_elec2`, `smoke_test_*`
- `configs/`: `tabm_baseline.yaml`(Phase0), `phase1.yaml`
- 문서: `EXPERIMENT_PLAN`(아키텍처 정본), `PRE_REGISTRATION`(사전 기준), `FINDINGS`, `progress`

---

## 3. 무엇을 했고 무엇이 나왔나 (사실 기록)

| # | 실험 | 결과 | 함의 |
|---|---|---|---|
| 0 | **Phase 0**: TabM 재현 (8개) | 공개치 ±1% 내 (sberbank rmse 0.257, homecredit auc 0.852 등) | 파이프라인 신뢰 |
| 1 | **Test 1**: 시간 vs 고정 메모리 (격리, inject off, 10시드) | clean null (|g|≤0.17, 4/4 비유의) | 시간-인덱싱 이득 0 |
| 2 | **학습 진단** (mem_gap 등) | concat에서 mem_gap≈0, 메모리 grad 100~1000×↓, 검색 1개로 붕괴 | **메모리 장식(z-지름길)** |
| 3 | **합성 양성대조** (y=x·w(t) 회전, 순수 concept) | time 0.13 vs fixed 1.03, +87%, mem_gap+0.93 | **구현 정확(버그 아님)** |
| 4 | **드리프트 분해(G3)** | covariate AUC≈1.0 pervasive, label ρ(t,y)≤0.13 | TabReD=강한 covariate, 약한 label-시간 |
| 5 | **concept 측정** (early vs late→future) | sberbank +36%, homecredit +4%, 나머지 작음 | 단 covariate 외삽에 오염 |
| 6 | **engaged time-vs-fixed** (sberbank, residual/memory_only) | residual g=−0.30(나쁨), memory_only 불안정 | sberbank +36%는 exploit 불가 |
| 7 | **Elec2 random, inject off** | time 0.954 vs fixed(무시간) 0.911 | *시간*은 유효 (메모리 아님 주의) |
| 8 | **Elec2 random, inject on (시간=피처)** | fixed(피처) **0.9615** > 메모리; 메모리 추가 +0.0017(p=.11) | **메모리가 시간-피처를 못 이김** |
| 9 | **Elec2 temporal (외삽)** | time 0.875 vs fixed 0.894 (g=−0.67), 불안정 | **외삽에서 메모리가 해로움** |

---

## 4. 비판적 검토 — 왜 안 됐나 (3층)
> *신규 멤버: 이 진단들이 맞는지, 빠진 게 없는지 봐주세요.*

### 4.1 가설/프레이밍 층
- **(치명) 시간-조건화의 표현적 redundancy**: 시간을 입력으로 받는 NN은 임의의
  P(y|x,t)를 표현 → "시간을 *추가*하는" 구조는 in-distribution 예측에서 표현적
  우위가 없음. 성능-양성 전제가 표현이론과 충돌.
- **해석가능성 청구가 조작적 정의/성공기준 없이 설정됨** → 반증 불가능(ill-posed).
  게다가 충실한 해석은 z-지름길 제거(병목)를 요구 → 성능과 trade-off.
- **동기가 현상-주도가 아니라 방법-공간 빈칸**. 빈칸은 *지배당해서* 비었을 수 있음.
- **프로토콜 불일치 가능성**: 드리프트 대응이 가치 있는 무대는 *온라인/스트리밍*인데
  우리는 *정적 train/test*에서 평가. (Elec2의 원래 출신지는 스트리밍)

### 4.2 설계 층
- **(핵심) 학습 프로토타입이 retrieval의 진짜 힘(실제 라벨-이웃)을 버림** → V_k가
  학습 벡터라 예측기가 가질 수 있던 파라미터일 뿐, 새 정보 없음 → 사실상 파라미터 레이어.
- **z-지름길**: predictor가 z로 풀어 메모리 무시(mem_gap≈0).
- **untrained 백본의 z에서 KMeans init** → 움직이는 표적.
- **Fourier 주기-정규화 불일치**(패치했으나 주기성 포착 제한적).

### 4.3 방법론/평가 층
- **1차 baseline이 틀림**: "시간 메모리 vs 고정 메모리"는 *"메모리가 단순 시간-조건화를
  이기나"*를 안 물음. 올바른 baseline = **시간을 피처로**(Elec2에서 뒤늦게 깔자 지배 드러남).
- **단일 시드 불안정**(Test2 r −0.087↔−0.404), **sberbank test로 튜닝(누수)**,
  **Test2 지표 임의성**(8슬라이스 centroid 상관).

---

## 5. 결론의 견고함 등급
> *무엇을 믿어도 되는가 — 이 리뷰의 핵심 산출.*

- **견고 (재현·통제·올바른 baseline로 뒷받침)**
  - 구현 정확 (합성 +87%).
  - Elec2에서 **시간-피처 ≥ 시간-인덱싱 메모리** (메모리 구조는 불필요).
  - 외삽(temporal)에서 메모리가 해로움/불안정.
  - TabReD = 강한 covariate, 약한 label-시간.
- **취약/미확정**
  - concept 측정의 +36%(sberbank)는 covariate 외삽에 오염 → "진짜 concept"인지 불명.
  - Elec2 단일 벤치 (일반성 미확인). Test2 노이즈.
- **측정한 적 없음 (그래서 주장 불가)**
  - 해석가능성의 실제 payoff/사용처.
  - 온라인/continual 프로토콜에서의 가치.
  - 강한 drift-aware baseline(ARF 등) 대비.

---

## 6. 열린 질문 & 가능한 방향
> §5에서 "성능-양성 전제는 (대체로) 닫혔다"가 나옴. 그럼 **질문을 다시 세운다면?**
> *아키텍처 패치가 아니라 새 베팅 후보들.* 새 멤버 의견 환영.

- **(가) 온라인/continual 세팅**: 데이터가 흐르며 모델이 *계속 적응*(실배포).
  질문: "시간-구조가 *적응 속도/샘플효율*에서 이기나?" → 표현이 아닌 *귀납편향* 주장.
  경쟁자: stream-learning(ARF 등).
- **(나) 산출물 = 드리프트 탐지/귀속**: 더 잘 맞히려 하지 말고, *언제·어디서 분포가
  변했는지*를 산출. 알려진 drift 시점 대비 평가 → 해석가능성을 *반증가능*하게 만든 버전.
- **(다) 현상-분석 논문**: 새 방법 없이 **"표 시간 벤치 = covariate≠concept, 그래서
  시간 방법이 안 통한다 + 진단 도구"** 자체가 기여. *현상-주도 + 증거 보유.* 가장 단단.
- **(라) 새 메커니즘**: 학습 프로토타입 대신 **실제 라벨 인스턴스에 대한 시간-인지 검색**
  (TabR+시간). 비파라메트릭+라벨 운반이라 시간-피처를 넘을 *여지*. **단 §4.1 표현 redundancy
  벽에 똑같이 부딪힐 수 있음** — 신중.
- **(마) 보류 / 더 큰 프로젝트에 흡수.**

판정 도구: 각 후보가 **§4.1 표현 redundancy 함정을 어떻게 피하는지**가 핵심.
(피하지 못하면 또 같은 결말.)

---

## 7. 재현 가이드 & 자산

### 7.1 환경 (서버, conda)
- 학습 env `explaintab311`(py3.11): `pip install torch(cu121) -r requirements.txt tabm`
- 전처리 env `tabred`: `conda env create -f ~/external/tabred/tabred-env.yaml`
  (+ lightgbm 바이너리휠, `kaggle>=1.8`; 상세 SETUP.md §4 / CLAUDE.md gotchas)
- 데이터: TabReD 8개(`prepare_all_data.sh`), Elec2는 `fetch_openml(151)` 자동.

### 7.2 핵심 재현 명령
```bash
python scripts/run_phase0.py --config configs/tabm_baseline.yaml        # 재현
python scripts/run_synth_control.py                                     # 구현 정확성(양성대조)
python scripts/run_phase1_sanity.py --config configs/phase1.yaml        # Test 1 (시간 vs 고정)
python scripts/run_drift.py --config configs/phase1.yaml --all          # covariate 드리프트
python scripts/run_conceptdrift.py --config configs/phase1.yaml --all   # concept 드리프트
python scripts/run_elec2.py --config configs/phase1.yaml --split random          # Elec2 (무시간 대비)
python scripts/run_elec2.py --config configs/phase1.yaml --split random --inject # Elec2 (시간-피처 대비)
python scripts/run_test3.py --config configs/phase1.yaml --dataset sberbank_housing --diag-every 5  # 내부 진단 보기
```

### 7.3 자산 (방향 무관 재사용)
- 검증된 파이프라인(로더/TabM/트레이너), 진단 도구(`diagnostics.py` mem_gap 등),
  covariate/concept 측정(`drift_measure.py`), 합성 양성대조 방법론, 음성 결과 자체.

---

## 8. 교훈 (메타)
- **통한 것**: 합성 양성대조(구현 검증), 내부 계측(mem_gap), *올바른 baseline*(시간-피처)
  → 환상("+4.3") 대신 진짜 답에 하루 만에 도달.
- **다음엔 다르게**:
  1. **성공 기준을 *사전에 반증가능*하게** 정의 (특히 "해석가능성").
  2. **가장 단순한 대안부터 baseline** (시간-피처를 Day 1에).
  3. **현상이 사는 프로토콜**을 고른다 (드리프트면 온라인).
  4. 동기를 *방법 빈칸*이 아니라 *현상*에서.

---

## 부록 — 포인터
- 증거 요약: `FINDINGS.md` · 일자 로그: `progress.md` · 아키텍처 정본: `EXPERIMENT_PLAN.md`
- 사전등록 기준: `PRE_REGISTRATION.md` · git 이력: `git log --oneline` (실험별 커밋 메시지에 근거 기록)
