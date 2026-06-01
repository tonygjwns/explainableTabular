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

## 현재 상태 (2026-06-02)
- ✅ **Phase 0 (TabM 재현) sberbank 검증 완료**: rmse 0.2572±0.0046 (공개 수치 ~0.24-0.26과 일치)
- ✅ 전 코드가 서버에서 실증됨 (로더/트레이너/TabM 래퍼/novel 모듈/통계)
- 🔄 **8개 데이터셋 전처리 + Phase 0 overnight 실행 중** (`scripts/run_overnight.sh`)
- ⬜ **다음 작업 = Phase 1**: novel 모듈을 TabM 위에 조립한 학습 모델 + sanity check Test 1~4 스크립트
- ⬜ PRE_REGISTRATION 동료 입회 commit (특히 결정 3: WTA 보류 합의)

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
  tabm_wrapper.py         TabM 래퍼 (encode d_out=None / predict)
src/data/
  tabred_loader.py        실물 포맷 로더 (X_meta[:,0]=timestamp, split 선택)
  splits.py               cai_resplit (Cai lag=0/bias-min, 검증 필요)
src/training/trainer.py   Phase 0 학습 (mean-loss over k, quantile+nan_to_num, grad clip, tqdm)
src/utils/                stats(Wilcoxon/FDR/Hedges'g), metrics, seed
scripts/
  run_phase0.py           Phase 0 (8개, 데이터 없으면 skip, tqdm)
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
- kaggle 1.8+ (KGAT 토큰)은 **py3.11 필요** → conda env `explaintab311`.
- 인증: `kaggle auth login` (OAuth) 또는 `KAGGLE_API_TOKEN` 환경변수. (구형 `KAGGLE_KEY`는 새 CLI가 무시)
- TabReD 전처리: `cd ~/external/tabred && PYTHONPATH=. python preprocessing/<script>.py`,
  의존성 `polars==0.20.19`(핀 필수), xlsx2csv, plotnine, delu 등. 4개 competition 규칙 수락 필요.
- 데이터 경로: `ln -s ~/external/tabred/data ~/explainableTabular/data` (config `data.root: data`).

## 상세 문서 (읽는 순서)
1. `HANDOFF.md` — 사고 흐름 전체 (왜 이 설계에 도달했나)
2. `EXPERIMENT_PLAN.md` — 아키텍처 정본 (§3 구조, §4 결정, §6 최소버전, §8 sanity, §9 ablation)
3. `PLAN.md` — phase/일정/자원
4. `PRE_REGISTRATION.md` — sanity 기준 사전 commit
5. `SETUP.md` — 환경/데이터 절차
6. `REFERENCES.md` — 논문/차별화
7. `progress.md` — 일자별 진행 로그 (가장 최신 상태)
