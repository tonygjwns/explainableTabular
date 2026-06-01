# ExplainableTab: Drift-Aware Tabular Deep Learning

> **프로젝트**: 표 데이터의 temporal distribution shift를 해석 가능한 방식으로 다루는 새 방법 개발
> **목표 학회**: NeurIPS 2026 또는 ICLR 2027
> **자원**: H100 ×2
> **기간**: 약 4개월 (실험 12-13주 + 글쓰기 3주)
> **GitHub**: https://github.com/tonygjwns-opt/explainableTabular

---

## 📂 이 폴더에 있는 문서들 (읽는 순서)

1. **`HANDOFF.md`** — 새 탭에서 작업 시작 시 **가장 먼저 읽을 문서**. 프로젝트 컨텍스트, 결정 요약, 즉시 행동 항목.
2. **`EXPERIMENT_PLAN.md`** — ⭐ **정식 실험 계획 (아키텍처 구체화 버전)**. 시간 인덱싱 메모리 + 검색 구조, 설계 결정, Phase 1 최소 버전, sanity check.
3. **`PLAN.md`** — 초기 phase/프로세스 구조. EXPERIMENT_PLAN.md가 아키텍처 부분을 갱신함 (phase 일정·통계 방법은 여전히 유효).
4. **`SETUP.md`** — 환경 셋업, repo clone, 데이터셋 다운로드 등 실용 가이드.
5. **`PRE_REGISTRATION.md`** — Sanity check 기준 사전 commit (실험 시작 전 lock).
6. **`REFERENCES.md`** — 인용할 주요 논문과 그들의 위치.

> ⚠️ **PLAN.md vs EXPERIMENT_PLAN.md**: PLAN.md는 초기 버전으로 "trajectory voting head"라는 막연한 표현을 씀.
> 이후 아키텍처가 **"시간 인덱싱 분포 메모리 + 검색"**으로 구체화되어 EXPERIMENT_PLAN.md에 정리됨.
> **아키텍처는 EXPERIMENT_PLAN.md를 따를 것.** Phase 일정·통계·자원은 두 문서가 일치.

---

## 🎯 한 줄 요약

표 데이터의 시간 분포 변화를, **시간에 따라 진화하는 프로토타입 메모리**에 저장하고,
입력이 **자기 시점의 어느 프로토타입에 가까운지 검색**하여 예측하는 구조.
TabM 백본 위에서, Cai et al.(변조)과 TabR(무시간 검색)이 둘 다 안 한 **"시간 인덱싱된 검색"**의 빈 자리를 차지.
청구: (1) 추가 성능 향상 또는 (2) 동등 성능 + 해석가능한 drift 분석.

---

## 🚦 현재 상태

- [x] 분야 논문 정독 (TabM, TabR, TabReD, ModernNCA, Cai & Ye ICML 2025, Cai et al. NeurIPS 2025, EvolveGCN, Latent ODE, TimeMCL 등)
- [x] 외부 검토 3채널 받음 (Gemini, Claude, GPT)
- [x] 전략 결정: **C → B Sanity-Check Gate**
- [x] 메인 청구 좁힘: **해석가능성 + 보완적 메커니즘**
- [x] **아키텍처 구체화: 시간 인덱싱 분포 메모리 + 검색** (EXPERIMENT_PLAN.md)
- [x] **설계 결정 5개 승인** (단순 softmax / 라벨분포 V_k / WTA 보류 / 시간 메모리+입력 / KMeans 초기화)
- [ ] 동료 검토 1라운드 (특히 결정 3: WTA 보류)
- [ ] Pre-registration commit
- [ ] Phase 0 시작

---

## ⚡ 새 탭에서 가장 먼저 할 일

```bash
# 1. 이 폴더의 모든 문서를 순서대로 읽기
cat HANDOFF.md PLAN.md SETUP.md REFERENCES.md

# 2. GitHub repo clone
cd ~/Desktop
git clone https://github.com/tonygjwns-opt/explainableTabular.git

# 3. SETUP.md의 환경 셋업 따라가기
```

새 탭의 LLM에게 작업 컨텍스트 부여할 때: **이 폴더의 모든 .md 파일 내용을 첫 메시지로 전달**.
