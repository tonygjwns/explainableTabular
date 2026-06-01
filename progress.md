# Progress Log

> 매일 짧게 기록. SETUP.md §7 형식.

## 2026-06-01 — repo 셋업 + Phase 0 스캐폴딩
- git repo 초기화, remote 연결 (tonygjwns/explainableTabular)
- 핸드오프 문서 7종 커밋 (아키텍처: 시간 인덱싱 메모리 + 검색)
- Phase 0 코드 스캐폴딩 생성:
  - `src/utils/stats.py` — Wilcoxon/BH-FDR/Hedges' g (완전 구현, 환경 독립)
  - `src/utils/seed.py`, `src/utils/metrics.py` — 완전 구현
  - `src/data/tabred_loader.py` — 스켈레톤 (Cai 분할 TODO, 데이터 필요)
  - `src/models/tabm_wrapper.py` — 스켈레톤 (external/tabm 필요)
  - `configs/tabm_baseline.yaml`, `scripts/run_phase0.py` — 오케스트레이션 스켈레톤
- **우리 novel 모듈 완전 구현** (external API 의존 없음, 서버에서 Claude 못 쓰니 로컬서 완성):
  - `src/models/temporal_embedding.py` — Fourier 시간 임베딩 tau(t)
  - `src/models/prototype_memory.py` — P_k(t)=P_k^base+drift_k(tau(t)), time_indexed 토글, KMeans init, smoothness penalty
  - `src/models/value_module.py` — V_k = W_y(label) + value (해석가능성용 라벨 표현)
  - `src/models/retrieval.py` — 단순 softmax 검색 + MemoryRetrievalLayer (predictor 포함)
  - `scripts/smoke_test_memory.py` — CPU 스모크 테스트 (shape/grad 검증)
  - ⚠️ 이 문서 머신엔 torch 없어 미실행 — **torch 있는 환경에서 `python scripts/smoke_test_memory.py` 먼저 돌려 검증할 것**
- 다음: GPU 머신에서 SETUP.md 따라 환경 구축 + smoke test

## Phase 0 체크리스트 (PLAN.md §4)
- [ ] conda 환경 구축 (SETUP.md §4)
- [ ] external/tabm, external/tabred clone (SETUP.md §2)
- [ ] TabReD 데이터 다운로드 (SETUP.md §3) — 작은 4개 우선
- [ ] `tabred_loader.py` 구현 — Cai & Ye ICML 2025 분할 적용
- [ ] `tabm_wrapper.py` 구현 — 공식 TabM 연결 (k=32 축 보존)
- [ ] Sberbank Housing에서 TabM 첫 학습
- [ ] 재현 수치가 TabM 논문/TabReD 표 ±1% 이내 확인
- [ ] (병행) Cai et al. NeurIPS 2025 코드 공개 여부 확인

## 통과 기준
TabM 재현 성공 + Cai et al. 베이스라인 작동 → Phase 1로.

## 미해결/결정 대기
- 결정 3 (WTA 보류) 동료 합의 — PRE_REGISTRATION commit 전
- TabReD timestamp 형식 통일 방법 (데이터 받은 후)
- Cai et al. NeurIPS 코드 공개 여부 → 미공개 시 직접 구현 일정 추가
