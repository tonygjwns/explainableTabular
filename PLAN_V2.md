# PLAN_V2 — 재건 계획 (외부 감사 반영, 2026-06-12)

> **배경**: 외부 코드 감사 + 문헌 대조 결과, (i) Q2b의 "구조 ≤ 피처" 음성은 **코드가 그 가설을
> 검정하지 않았음**이 확인되어 증거로서 무효화됨, (ii) Claim A의 측정 프레임은 DISDE/WhyShift와
> 겹쳐 재포지셔닝 필수, (iii) Cai & Ye(ICML/NeurIPS 2025)의 양성 결과와의 긴장 미해소.
> 이 문서가 **유일한 현행 계획**. PLAN_RESCUE.md의 게이트 철학(사전등록·결정규칙)은 계승하되
> 결정규칙은 §R1.4에서 재등록한다.

---

## 0. 무엇이 죽고 무엇이 사나

**죽음 (결론 폐기, 재실행 대상)**
- ~~"Q2b 구조 음성 확정, ≥2 데이터셋 잠금"~~ (RESULTS §10·§11의 *해석*): value 훅이 선형이라
  `Σw_k·Linear(Δτ_k)=Linear(Σw_k·Δτ_k)` — 집계된 Δt 피처로 붕괴, `y_i`와 무관 → "stale label
  보정" 가설을 표현 불가. 실제 검정된 것 = "검색-가중 Δt 피처 vs 직접 t 피처".
- ~~"검색 substrate가 본질적으로 parametric에 짐"~~: 기판이 sub-TabR 상태로 비교됨
  (온도/스케일링 없음, key projection 없음, train 255 vs eval 4096 context 불일치, eval context=
  train의 13% 균일 샘플, topk/context/온도 전부 미튜닝 — lr만 튜닝).
- ~~"빈 교차점(시간×검색)"~~ 수사: 스트리밍 문헌에 FISH(Žliobaitė ~2011), SAM-kNN(ICDM 2016)
  존재 → "**미분가능 딥 표 학습 내**"로 한정 + 인용.

**생존 (재사용 자산)**
- 파이프라인·Phase 0 재현(8/8), Q1 충실성 게이트(PASS), 합성 양성대조(+87%),
  within-overlap concept 측정(elec2 +0.132)과 F3 이분법, 진단 도구(mem_gap 등),
  INSECTS 로더/러너 인프라, 시드-페어링·paired CI·oracle 라벨링 규율,
  학습곡선 drift 지문(①), val→test Spearman 진단.
- **숫자 자체는 유효** (mlp_t 0.9027 등) — 무효인 것은 그 숫자에서 "구조 ≤ 피처"를 읽는 해석.

---

## R0. 코드 수정 (로컬, ~1주) — 전부 push 후 서버 실행

### R0.1 `src/models/tabr.py` — 가설을 실제로 검정하게
1. **value 훅 비퇴화화** (`value_hook ∈ {linear(레거시 보존), mlp, gate}`):
   - `mlp`: `val = val + value_time_mlp(cat([lab, dtime]))` — 2-layer MLP, **마지막 layer zero-init**
     (time_tabr ≡ tabr 초기 등가 보존). 라벨×시간 상호작용이 집계에서 살아남음 → "stale label 보정" 표현 가능.
   - `gate`: `val = val * sigmoid(gate_mlp(dtime))`, bias init으로 σ≈1에서 시작.
2. **검색 스케일링**: `sim = -‖k_q-k_i‖²/τ`, τ = 학습 스칼라(`log_tau` param) 또는 `/√d` 옵션.
   **key projection `W_K`** 추가(검색용 k와 predictor용 z 분리 — TabR 정품 정합).
3. **metric 훅을 top-k 선택 *전*에**: 시간 변조가 *어떤 이웃을 가져올지*를 바꿀 수 있게
   (현재는 시간-무관 top-32 안에서 재가중만 가능).
4. **4번째 arm `tabr_t`**: predictor 입력 `[z_q, τ(t_q), agg]` — "검색이 직접 시간 피처 *위에*
   무엇을 더하나"를 격리. **이것이 청구가 실제로 요구하는 비교** (현 설계는 구조×시간주입위치 교락).
5. (선택, anchor용) TabR 보정항 `T(k_q−k_i)` 옵션 — 정품 TabR 재현 근사.

### R0.2 `src/training/tabr_trainer.py` — train/eval 정합
6. **context 정합**: `--train-context {inbatch, sampled-4096}` / `--eval-context {4096, full}`.
   기본 = train sampled-4096(배치 제외) + eval full-train (계산 사소: 45k×128).
7. eval `ctx_idx`를 **전용 `torch.Generator(seed)`**로 — 같은 시드에서 모든 arm이 같은 context
   (현재 arm별 RNG 소비량 차이로 달라져 time_tabr−tabr 대비에 노이즈).
8. `n_classes`/카디널리티는 train∪val에서만 (test-peek 제거).
9. <2 샘플 배치 skip을 전 arm 공통으로 (현재 mlp_t만 학습하는 비대칭).

### R0.3 통계·러너
10. `src/utils/stats.py`: **paired Hedges' g** `g_z = mean(d)/std(d,ddof=1)·J(n)` 추가,
    기존 unpaired g는 명시 라벨 시에만.
11. `scripts/run_elec2_q2.py`: 4-arm 지원, `--report-grid`에 **val-fair 선택과 oracle 병기**,
    retrieval knob 그리드(`--topk`, `--context`, τ), `--max-samples` 사용 여부 record에 명기.
12. `scripts/run_anchors.py` (신규): **LightGBM(+t)** / **kNN(+t)** / **no-change(persistence)**
    (elec2 — 문헌상 ~85%, Kappa-Temporal 근거) / (보정항 켠) **정품-근사 TabR**. 같은 split·시드.
    → "내부적으로 공정하나 외부적으로 미보정" 반론을 표 하나로 차단.
13. `scripts/smoke_test_tabr*.py` 갱신 (새 훅/arm/context 배선, CPU).

### R0.4 문서 동기화
14. RESULTS §10·§11에 *"해석 무효화(훅 선형 붕괴) — PLAN_V2 재실행으로 대체 예정"* 주석,
    FINDINGS "Q2b ANSWERED" 동일, CLAUDE.md 현재상태·NEXT_TAB.md를 PLAN_V2 포인터로 갱신.
15. **PREREG_V2.md** 작성 (아래 R1.4를 서버 실행 *전* commit, hash 기록) — 사전등록 문서 단일화
    (PRE_REGISTRATION→PLAN_RESCUE→Q2B_PROPOSAL로 표류한 기준을 한 표로 재구성, 각 잠금 시점 hash 명기).

---

## R1. Claim B 재검정 (서버, ~1–2주)

### R1.1 튜닝 단계 (val-기반, 오염 방지)
- 3시드, val로 retrieval knob {topk 8/32/128 × context × τ(학습/√d)} 데이터셋별 선택 → **고정**.
- 점검: 튜닝된 `tabr − mlp_t` 격차가 −0.038에서 절반 이상 줄면 "substrate 적자" 프레이밍 전면 수정.

### R1.2 본 실행 (사전등록 검정력 준수)
- **25시드**(사전등록값; 현행 10시드는 INSECTS CI 상단 +0.001로 미달) × **4 arm**(mlp_t/tabr/tabr_t/
  time_tabr-mlp훅) × lr 그리드 × temporal(+random 대조).
- 데이터: **INSECTS incremental + abrupt + gradual**(clean n=3; val→test ρ 확인) + elec2(보조 강등,
  val→test 0.07 명기). (선택 stretch: Airlines 로더 추가 — streams 3rd bench.)
- anchor 표(R0.3-12) 동시 실행. elec2에서 no-change가 4-arm을 다 이기면 그 사실 자체를 보고
  (비교가 trivial baseline 아래라는 적신호 — 숨기지 않음).

### R1.3 백본 정합 (1셀만)
- 최강 세팅 1개(INSECTS incremental)에서 인코더를 TabM으로 교체한 재실행 1회 —
  "MLP 인코더라 그렇다" 반론 차단용 스팟체크.

### R1.4 사전등록 결정규칙 (PREREG_V2에 잠금; 골자)
- **주 대비 = time_tabr − tabr_t** (둘 다 직접 t 피처 보유 → 순수 "시간-인덱싱 검색의 추가 가치").
  보조 대비 = time_tabr − mlp_t, tabr_t − mlp_t (검색 자체의 가치).
- **구조 우위(부활)**: temporal에서 paired 95% CI > 0, clean 데이터셋 ≥2/3, 25시드.
- **음성(이번엔 진짜)**: CI가 0 포함 또는 <0, clean ≥2/3 → "비퇴화 훅 + 정품화 기판 + oracle을
  줘도 구조가 피처를 못 넘음" — 현재보다 훨씬 강한 음성.
- random split 이득만 있으면 적신호(자기상관 누수) — 기존 규칙 유지.
- 어느 쪽이든 논문에 들어감: 부활 → 좁은 양성 method 결과(+분해), 음성 → Claim B 완성.
  (참고: 현 INSECTS에서 선형 훅조차 time_tabr−tabr=+0.028 → 비선형 훅으로 뒤집힐 확률 무시 못 함.)

---

## R2. Claim A 재포지셔닝 (R1과 병렬 가능, ~3–4주) — 논문 등급 결정 구간

### R2.1 문헌 검증 (웹, 즉시 — 모든 후속 작업의 전제)
- DISDE(Cai·Namkoong·Yadlowsky, arXiv:2303.02011), WhyShift(NeurIPS 2023), Webb 2016/2018,
  Gama 2014, FISH/SAM-kNN, Drift-Resilient TabPFN(NeurIPS 2024), Cai & Ye ICML/NeurIPS 2025 —
  **원문 확인** (이번 대조는 웹 차단 상태에서 로컬 문서+모델 지식 기반이었음). 관련연구 절 재작성.
- "TabReD에 X/Y|X 분해를 적용한 2025–26 선행이 없는지" 최종 확인 (있으면 즉시 전략 수정).

### R2.2 DISDE 퇴화 실험 (도구킷의 존재 이유를 DISDE의 언어로)
- DISDE식 분해(density-ratio = 기존 시점분류기 재사용)를 TabReD에 적용 → 고-covariate 5개에서
  ESS 붕괴/분산 폭발(우리 ESS=20 현상이 바로 이것)을 정량 시연 ↔ within-overlap 프레임은 작동.
- 프레이밍: "DISDE의 overlap-제한 Y|X 측정을 시간축에 model-based로 옮기고, support 붕괴
  영역에서도 동작하게 만든 것" — 인용+적응, 재발명으로 보이지 않게.

### R2.3 ★ Cai & Ye 판결 실험 (최대 위협 → 최고 결과 전환; 최우선 EV)
- LAMDA 공개 코드(Tabular-Temporal-Modulation)로 TabM+변조를 TabReD 2~3개에서 재현.
- 그 **이득을 우리 분해 렌즈에 통과**: concept≈0이 *측정으로 확인된* cooking/maps에서 이득이
  그대로 나오면 → 이득은 X-side 적응(정의상 concept 착취 불가). overlap-매칭 영역 내/외 이득
  분해 병행.
- 성공 시 헤드라인: "시간-인지 방법이 이기는 것은 맞다 — 단 이긴 이유는 covariate 적응이며,
  concept은 그곳에서 측정조차 불가" → 경쟁 서사 포섭. (반대로 나오면 Claim A 범위 축소 — 그것도
  사전 명시.)

### R2.4 도구킷 1급화 + ground-truth 검증 (D&B 핵심 요건)
- 합성 generator: (covariate 강도 × concept 강도) 2×2+ 그리드 → 도구킷이 비율을 복원하는지,
  overlap→0에서의 실패 모드/추정 분산 문서화.
- INSECTS 6 variant(설계 drift, 위치 기지)로 실데이터 검증.
- 패키징: API 정리, README, 재현 파이프라인, 라이선스 (Croissant/datasheet는 D&B 제출 시).

### R2.5 잔여 보강
- Q1 큰-회전(≥180°, 기저 정합) robustness 1회 (기존 미결).
- elec2 +0.132에 no-change/Kappa-Temporal 통제 명기 (Žliobaitė 2013 비판 선제).
- WhyShift 대조 문장 확정: "공간 표 shift = Y|X-지배 ↔ 시간 표 shift = X-지배".

---

## R3. 정렬·학회 게이트 (R1 결과 손에 쥐고)

- **G3 (지도교수 정렬)**: R1 재검정 결과 + R2.1 문헌 지형 들고 감. *현행 10시드 결과로 정렬 금지*
  (코드 읽는 사람에게 같은 지적 받음).
- 타깃 사다리 (기존 판단 유지, 문헌 선례로 보강):
  - **워크숍 (지금)**: unmeasurability 발견 우선권. R1 완료 전엔 Claim B 빼고 Claim A만.
  - **NeurIPS D&B (주 타깃)**: TabReD 8 + elec2 + INSECTS variants(+Airlines) ≈ 10–12 데이터셋 ×
    방법 8–12개(GBDT/MLP+t/TabM/Cai 변조/TabR/time-TabR/TabPFN류/no-change/스트리밍) + 도구킷
    릴리스 + DISDE·WhyShift 직접 비교. (WhyShift·Wild-Time의 "depth" 경로.)
  - **메인트랙**: 위 + 15–20 데이터셋 + R2.3을 중심 결과로.
  - **Claim B 단독 제출 금지** (2–3 데이터셋·1방법은 어느 바도 미달 — Wild-Time조차 5×13).
- 논문 골격: Claim A 리드(분해+측정불가+도구킷, DISDE/WhyShift 위에 위치) → R2.3 판결 →
  Q1 충실성 → Claim B(재검정판, 분해·paired) 보조 → 한계(redundancy는 가설로만).

---

## 일정·의존성 요약

| 주차 | 작업 | 산출 |
|---|---|---|
| 1 | R0 전체(로컬) + R2.1 문헌 검증(병렬) | push 완료, PREREG_V2 hash, 관련연구 지형 확정 |
| 2–3 | R1.1 튜닝 → R1.2 본 실행(서버) ‖ R2.2 DISDE 실험 | Claim B 판정(유효판), anchor 표 |
| 3–5 | R2.3 판결 실험 + R2.4 도구킷 검증 ‖ R1.3, R2.5 | 헤드라인 확정 재료 |
| 5–6 | R3 정렬 → 타깃 확정 → 집필 착수 | 워크숍 즉시 / D&B 본격 |

**경로 의존성**: R0.1-1(훅)·R0.1-4(tabr_t)가 모든 것의 전제. R2.3이 논문 등급을 결정.
R2는 R1 결과와 무관하게 가치 있음(어느 판정이든 Claim A가 리드).

## 메타 규칙 (이번 사이클에서 추가)
- **가설→수식→코드 등가성 체크**: 메커니즘 구현 시 "이 모듈이 표현 *못 하는* 것"을 docstring에
  명시하고 smoke test에 표현력 체크 1개 포함 (선형 붕괴류 재발 방지).
- 사전등록은 **단일 문서(PREREG_V2) 누적** — 기준 변경은 새 섹션+hash+사유로만.
- 헤드라인 문장에는 반드시 (시드 수, CI, 데이터셋 수) 병기 — "잠금/확정" 단독 표현 금지.
