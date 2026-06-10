# CLAUDE.md — 프로젝트 메모리 (매 세션 자동 로드)

> 이 파일은 Claude Code가 세션 시작 시 자동으로 읽습니다. 새 탭은 이것만 읽어도
> 핵심 컨텍스트를 잡습니다. 더 깊은 내용은 아래 "상세 문서"를 참조.

## 한 줄 정체성
표 데이터의 **temporal distribution shift**를 다루는 연구. TabM 백본 위에
**시간 인덱싱 프로토타입 메모리 + 검색(retrieval)** 레이어를 얹어, drift를
해석 가능한 형태로 노출. (Cai et al.의 변조 ≠ 우리 검색. TabR의 검색엔 시간 좌표 없음 → 그 빈 교차점.)

## 워크플로우 (중요)
- **로컬(이 repo, Windows `C:\Users\joon\Desktop\ExplainableTab`)에서 코드 작성 → git push**
- **서버(H100, `~/explainableTabular`, conda env `explaintab311`, py3.11)에서 git pull → 실험**
- 서버에선 Claude를 못 씀. **그래서 로컬에서 최대한 완성된 코드를 만들어 push**해야 함.
- GitHub: https://github.com/tonygjwns/explainableTabular

## 현재 상태 (2026-06-05) — ⚠ 새 탭은 먼저 `NEXT_TAB.md` 읽을 것
**핵심 문서(읽는 순서)**: `NEXT_TAB.md`(인계·다음 행동) → `RESULTS.md`(결과 ledger) →
`FINDINGS.md`(증거 사슬) → `PLAN_RESCUE.md`(사전등록 프로토콜·결정규칙) → `Q2B_PROPOSAL.md`(현 빌드).

한 줄: Phase 0 8/8 PASS. Phase 1 메모리 메커니즘은 **TabReD에서 성능 이득 0(검증된 음성)**,
**합성/Elec2 concept에선 충실·작동**. Q1 충실성 **PASS**, Elec2 measurable concept +0.132 확정.
**Q2b(인스턴스 time-TabR) 실행·판정 완료 → 구조 청구 = 견고한 음성, ≥2 데이터셋 잠금**:
elec2(time_tabr−mlp_t=−0.018) + Insects(−0.011, 비-trivial multiclass) **둘 다 구조 ≤ 시간-피처**.
**사전등록 "구조 우위(≥2 데이터셋)" 미충족 → 방향 = §6(다) 음성/분석 paper(A' method 경로 닫힘).** 지도교수 정렬만 남음.
- ✅ Q1 게이트 PASS, Elec2 concept 확정, **Q2b 음성 확정(elec2+Insects)**(RESULTS §10·§11, FINDINGS "Q2b ANSWERED").
- ✅ Q2b 코어: `src/models/tabr.py`(TimeTabR+TimeTabRModel, (t_q,t_i,y_i) 훅, dropout), `src/training/tabr_trainer.py`
  (in-batch 학습/고정-context eval, train-loss 기록·step-eval·min_epochs), `scripts/run_elec2_q2.py`(요인설계 + `--diag`/`--report-grid`/`--dataset`).
- ✅ **Insects 인프라**: `src/data/insects_loader.py`(river, multiclass), `run_elec2_q2.py --dataset insects`, `smoke_test_insects.py`.
- ✅ **Insects 실행·판정 완료**: 구조 음성 일관(§11). ≥2 데이터셋 결정규칙 충족 → §6(다) 확정.
- 🔄 **다음 작업**: **지도교수 정렬**(RESULTS §10·§11 / FINDINGS "Q2b ANSWERED" 지참) → §6(다) 음성/분석 paper 작성 착수.
  (선택 보강: Insects abrupt variant·random 대조로 음성 견고화.)

## 아키텍처 (요지)
- 백본: **TabM** (`pip install tabm`; `from tabm import TabM`). k=32 submodel.
- 메모리: `P_k(t) = P_k^base + drift_k(Fourier(t))` — 시간에 따라 진화하는 프로토타입.
- 검색: `w_k = softmax(-‖z - P_k(t_x)‖²/τ)`, 집계 `Σ w_k·V_k`, `V_k = W_y(label)+value`.
- **head가 아니라 백본-예측기 사이 메모리 레이어.** (head는 정보 저장 못 함)

## 절대 하지 말 것 (Don'ts)
- ❌ 시간을 head(출력단)에만 주입 (Cai 증거 12.6%) → 메모리(주)+입력(보조)
- ❌ Phase 1에 WTA/annealing/TabR 보정항/외적 게이팅 (결정 3 — ablation factory 함정; Phase 2로)
- ❌ TabM의 32 submodel 평균 후 메모리 얹기 (앙상블 다양성 파괴; Phase 2 ablation에서 결합)
- ❌ Welch's t-test (페어드 데이터 → Wilcoxon signed-rank)
- ❌ sanity check 결과 본 뒤 PASS/FAIL 기준 변경

## 반드시 (Do's)
- ✅ Phase 1 = 최소 버전: 단순 softmax 검색, KMeans 초기화, L_smooth만 (EXPERIMENT_PLAN §6)
- ✅ 통계: paired Wilcoxon + Benjamini-Hochberg FDR + Hedges' g (`src/utils/stats.py` 구현됨)
- ✅ Cai et al. NeurIPS 2025 베이스라인 위/옆에 우리 추가, 그 위에서 sanity check
- ✅ 회귀 NaN 주의: TabReD X_num에 결측 있음 → quantile 후 `nan_to_num` (이미 trainer에 반영)
- ✅ 요소는 하나씩 추가 (Phase 2 ablation 로드맵 EXPERIMENT_PLAN §9)

## C → B Gate 전략
Phase 1 sanity check(Test 1~3 게이팅, Test 4 진단)로 PASS → C(메인 청구) / FAIL → B(진단 paper).
기준은 `PRE_REGISTRATION.md` §3에 사전 commit. 모호하면 FAIL(안전).

## 코드 맵 (구현 완료)
```
src/models/
  temporal_embedding.py   Fourier τ(t)
  prototype_memory.py     P_k(t) (time_indexed 토글, KMeans init, smoothness)
  value_module.py         V_k (라벨표현+학습벡터)
  retrieval.py            softmax 검색 + MemoryRetrievalLayer (predictor 포함)
  tabr.py                 TimeTabR(인스턴스 검색, (t_q,t_i,y_i) 훅) + TimeTabRModel(3-arm) ← Q2b
  tabm_wrapper.py         TabM 래퍼 (encode d_out=None / predict)
src/data/
  tabred_loader.py        실물 포맷 로더 (X_meta[:,0]=timestamp, split 선택)
  elec2_loader.py         Elec2(real concept-drift) → TabReDDataset (split random/temporal)
  insects_loader.py       INSECTS(river, designed drift, multiclass) → TabReDDataset ← 2nd dataset
  splits.py               cai_resplit (Cai lag=0/bias-min, 검증 필요)
src/training/
  trainer.py              Phase 0 학습 (mean-loss over k, quantile+nan_to_num, grad clip, tqdm)
  phase1_trainer.py       Phase 1 메모리 모델 학습 (L_main + λ·L_smooth)
  tabr_trainer.py         Q2b: train_timetabr (in-batch 검색 학습 / 고정-context eval) ← NEW
src/utils/                stats(Wilcoxon/FDR/Hedges'g), metrics, seed
scripts/
  run_phase0.py           Phase 0 (8개, 데이터 없으면 skip, tqdm)
  run_elec2_q2.py         Q2b 요인설계(+`--diag`/`--report-grid`/`--dataset elec2|insects`); diagnostics.jsonl 누적
  smoke_test_tabr.py      TimeTabR 모델 CPU 배선 (서버 통과)
  smoke_test_tabr_trainer.py  train_timetabr CPU 배선 (합성 라벨-flip)
  smoke_test_insects.py   insects 로더 + 멀티클래스 Q2b 배선 (river 없으면 skip) ← NEW
  smoke_test_memory.py    novel 모듈 CPU 배선 검증 (통과 확인됨)
  prepare_all_data.sh     TabReD 8개 전처리 일괄
  run_overnight.sh        전처리→Phase0 한 번에 (nohup)
```

## 다음 작업 (Phase 1) 구현 항목
1. **Phase 1 모델**: `MemoryRetrievalLayer`(이미 있음)를 TabM `encode()`(d_out=None, reduce='mean') 출력 위에 연결한 end-to-end 학습 모델. 입력에도 시간 약하게 주입.
2. **Phase 1 트레이너**: trainer.py 확장 or 신규 — L_main + λ·L_smooth, Cai 분할(`cai_resplit`) 또는 TabReD `default`/`random` split로.
3. **베이스라인**: Cai et al. NeurIPS 2025 재구현 + Fixed memory(시간무관) 대조군.
4. **Sanity check Test 1~4** (EXPERIMENT_PLAN §8, PRE_REGISTRATION §3):
   - T1 시간인덱싱 vs 고정 메모리(성능) / T2 외삽 / T3 검색의미+궤적시각화 / T4 시간주입위치
5. `scripts/run_phase1_sanity.py` + `src/analysis/` (extrapolation/retrieval/trajectory_viz/time_injection).

## 서버 환경 gotchas (재현 시 참고)
- **공유 H100 서버라 `~/miniconda3`·`~/external`·데이터가 통째로 사라질 수 있음** (2026-06-02 실제 발생;
  디스크 압력 시 누군가/무언가가 큰 디렉토리를 정리). git repo만 살아남음 → **env 재구축 절차를 알아둘 것**(아래).
- **두 env로 분리**: 전처리=`tabred`(TabReD 공식 env), 학습=`explaintab311`(우리 repo+tabm). 서로 deps 안 겹침.
- **전처리 env = TabReD 공식 yaml로 한 방에** (deps 하나씩 깔지 말 것 — `import lib`가 torch/faiss/rtdl 전체 스택을 끌어옴):
  - `cd ~/external/tabred && conda env create -f tabred-env.yaml` (env명 `tabred`, py3.11, numpy 1.26.4 등 핀)
  - ⚠️ yaml의 `lightgbm`이 GPU 소스빌드(`--no-binary`)라 OpenCL 없으면 pip 단계 실패 → 그냥 `pip install lightgbm`(바이너리휠)로 보강. 끊긴 pip deps도 보강: `pip install delu==0.0.23 rtdl_num_embeddings==0.0.9 rtdl_revisiting_models==0.0.2`
  - 검증 관문: `PYTHONPATH=. python -c "import lib; print('OK')"` 통과해야 전처리 가능.
- kaggle: 1.8+ (KGAT 토큰)은 py3.11 필요. 인증 `kaggle auth login`(OAuth, user-level이라 env간 공유). `tabred` env는 kaggle 1.6.11 핀(=kaggle.json 요구)이라 **`pip install "kaggle>=1.8"`로 올려야** 기존 OAuth 토큰을 씀. (구형 `KAGGLE_KEY`는 무시)
- 전처리 실행: `TABRED_REPO=~/external/tabred bash scripts/prepare_all_data.sh` (`tabred` env에서; tmp 데이터셋별 자동정리 내장 — 누적 시 디스크풀 Errno 28).
- 데이터 경로: `ln -s ~/external/tabred/data ~/explainableTabular/data` (config `data.root: data`). 폴더는 하이픈(sberbank-housing), 로더가 키↔폴더 매핑.
- Phase 0 학습은 **반드시 `explaintab311`에서** (`run_overnight.sh`는 prep+phase0를 한 env에서 돌리니 env 분리 상황에선 쓰지 말고 둘을 따로 실행).

## 상세 문서 (읽는 순서)
1. `HANDOFF.md` — 사고 흐름 전체 (왜 이 설계에 도달했나)
2. `EXPERIMENT_PLAN.md` — 아키텍처 정본 (§3 구조, §4 결정, §6 최소버전, §8 sanity, §9 ablation)
3. `PLAN.md` — phase/일정/자원
4. `PRE_REGISTRATION.md` — sanity 기준 사전 commit
5. `SETUP.md` — 환경/데이터 절차
6. `REFERENCES.md` — 논문/차별화
7. `progress.md` — 일자별 진행 로그 (가장 최신 상태)
