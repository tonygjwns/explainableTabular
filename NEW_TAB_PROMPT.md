# 새 탭에서 LLM에 전달할 첫 프롬프트

> 새 탭(Claude 또는 다른 LLM)에서 작업 시작 시, 이 프롬프트를 첫 메시지로 전달합니다.

---

## 옵션 1: 짧은 부트스트랩 (LLM이 폴더 접근 가능한 경우)

```
이 프로젝트는 표 데이터 딥러닝의 temporal distribution shift를 해석 가능한 방식으로
다루는 연구입니다. 4개월 plan을 짜둔 상태이고, 곧 실험을 시작합니다.

다음 폴더의 모든 .md 파일을 읽고 컨텍스트를 파악해주세요:
C:\Users\joon\Desktop\ExplainableTab\

읽는 순서:
1. README.md
2. HANDOFF.md (컨텍스트·사고흐름)
3. EXPERIMENT_PLAN.md (아키텍처 정본 — 가장 중요)
4. PLAN.md (phase·일정·자원 로지스틱스)
5. SETUP.md
6. PRE_REGISTRATION.md
7. REFERENCES.md

다 읽은 후, "현재 상태에서 가장 시급한 다음 단계 3가지"를 한국어로 답해주세요.
```

---

## 옵션 2: 긴 부트스트랩 (LLM이 폴더 접근 불가, 직접 컨텍스트 주입)

아래 텍스트를 그대로 복사해서 첫 메시지로 사용. (각 .md 파일 내용을 첨부)

````
# 프로젝트 인계 컨텍스트

다음은 새 작업 세션 시작 전, 이전 세션에서 작성한 프로젝트 핸드오프 문서들입니다.
모두 읽고 컨텍스트를 완전히 파악한 후, 작업을 이어가주세요.

---

## 프로젝트: ExplainableTab

표 데이터 딥러닝의 temporal distribution shift를 해석 가능한 방식으로 다루는 연구.
GitHub: https://github.com/tonygjwns-opt/explainableTabular

## 현재 상태
- 4개월 실험 plan lock 직전
- C → B Sanity-Check Gate 전략 채택
- 메인 청구: 해석가능성 + Cai et al. NeurIPS 2025 위의 보완적 메커니즘

## 핵심 결정사항

1. **백본**: TabM (Gorishniy et al. ICLR 2025)
2. **베이스라인**: Cai et al. NeurIPS 2025 "Feature-aware Modulation" (직접 경쟁작, 변조)
3. **우리 추가**: **시간 인덱싱 프로토타입 메모리 + 검색** (head 아님 — 백본/예측기 사이 메모리 레이어)
   `P_k(t) = P_k^base + drift_k(Fourier(t))`, 입력이 자기 시점 어느 프로토타입에 가까운지 검색
4. **벤치마크**: TabReD 8 데이터셋 + Cai & Ye ICML 2025 분할 프로토콜
5. **통계**: Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR + Hedges' g
6. **시드**: 차등 적용 (Phase 0/1: 5-10, Phase 2 메인: 20)
7. **Phase 1은 최소 버전**: 단순 softmax 검색, KMeans 초기화, WTA·보정항·외적 없음

## 가장 시급한 행동

1. PRE_REGISTRATION.md의 sanity check 기준(4 테스트)을 동료와 함께 commit (특히 결정 3: WTA 보류 합의)
2. TabM 환경 셋업 + 작은 데이터셋에서 재현 시작
3. Cai et al. NeurIPS 2025 코드 (github.com/LAMDA-Tabular/Tabular-Temporal-Modulation) 확인

## 절대 하지 말 것

- 시간을 head(출력단)에만 주입 (Cai 증거 12.6%, WTA는 시간축 아닌 출력모드로 분화) → 메모리+입력으로
- Phase 1에 WTA/annealing/TabR 보정항/외적 게이팅 넣기 (ablation factory 함정 — Phase 2로)
- Welch's t-test (페어드 데이터엔 Wilcoxon)
- TabM의 32 submodel을 평균 + 메모리 레이어 얹기 (TabM 파괴 — Phase 2 ablation에서 결합 결정)
- 결과 본 후 sanity check 기준 변경
- TabReD 표 수치 인용 (Phase 0에서 직접 재현)

---

[여기에 README.md, HANDOFF.md, EXPERIMENT_PLAN.md, PLAN.md, SETUP.md, PRE_REGISTRATION.md,
REFERENCES.md 내용 순서대로 첨부]

---

위 모든 문서를 읽고 이해했다면, "현재 상태에서 다음 단계 3가지"를 한국어로 답해주세요.
````

---

## 옵션 3: 코딩 task 시작용

```
이전 세션에서 만든 ExplainableTab 프로젝트의 핸드오프 문서들이 다음 폴더에 있습니다:
C:\Users\joon\Desktop\ExplainableTab\

README.md, HANDOFF.md, EXPERIMENT_PLAN.md, SETUP.md 정도만 빠르게 읽고,
Phase 0의 첫 작업인 "TabM 환경 셋업 + Sberbank Housing에서 재현"을 시작해주세요.

작업 디렉토리: ~/Desktop/explainableTabular/ (이미 clone되어 있다고 가정)
외부 의존성: ~/Desktop/external/tabm/ (TabM 공식 repo)

다음 단계를 수행:
1. 우리 repo에 src/ 구조 만들기 (SETUP.md §5 참고)
2. src/data/tabred_loader.py 작성 (TabReD 데이터 로더)
3. src/models/tabm_wrapper.py 작성 (TabM을 우리 파이프라인에 래핑)
4. scripts/run_phase0.py 작성 (재현 entry point)

코드 작성 시 EXPERIMENT_PLAN.md §7의 통계 방법 (Wilcoxon, FDR, Hedges' g) 반영 필수.
아키텍처(메모리+검색)는 EXPERIMENT_PLAN.md §3·§6이 정본.
```

---

## 부록: LLM에 컨텍스트 부여 시 팁

### 효과적인 컨텍스트 주입 순서

1. **프로젝트 정체성 한 줄** ("표 데이터 + temporal shift + 해석가능성")
2. **현재 단계** ("Phase 0 시작 직전, plan lock 직후")
3. **결정 사항 핵심 5개** (백본, 베이스라인, 우리 추가, 통계, 시드)
4. **다음 행동 3개**
5. **금기 사항 4개**
6. (선택) 전체 문서 첨부

### LLM이 처음 잡아야 할 핵심 5가지 사실

1. **우리 구조는 "시간 인덱싱 메모리 + 검색"** — head 아님. 시간 분포 정보를 메모리에 저장하고, 입력이 어느 시점 프로토타입에 가까운지 검색
2. **Cai et al. NeurIPS 2025**가 직접 경쟁작 (변조 vs 우리 검색). 모르면 차별화 못 함
3. **시간은 메모리 + 입력에**, head 출력단에만 넣으면 약함 (Cai 12.6%, WTA는 시간축 아님)
4. **C→B Gate 전략**에서 sanity check(4 테스트)가 결정 포인트. Phase 1은 최소 버전(WTA 없음)
5. **메인 청구는 성능이 아니라 해석가능성 + 보완적 메커니즘**

이 5개 모르면 잘못된 방향으로 갈 가능성 큼.

### 컨텍스트 검증 질문 (새 LLM이 제대로 이해했는지 확인)

작업 시작 전 LLM에게 물어볼 것:
1. "우리 메인 청구가 뭐였지?" → "해석가능성 + 보완적 메커니즘" 답해야
2. "왜 우리 구조는 head가 아니라 메모리인가?" → "head=변환은 정보 저장 못 함, 시간 분포를 저장·검색하려면 메모리 필요" 답해야
3. "왜 TabM의 32 submodel을 단순 평균하면 안 되지?" → "TabM의 ensemble diversity 파괴" 답해야
4. "Phase 1 sanity check FAIL 시 어떻게 하지?" → "Stage 3b로 pivot, 진단 paper" 답해야
5. "왜 Phase 1에 WTA를 안 넣지?" → "WTA는 출력 모드로 분화하지 시간축 아님 + ablation factory 함정. 시간은 P_k(t)의 t가 담당" 답해야

위 5개 다 맞으면 컨텍스트 잘 잡힘.
