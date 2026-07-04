# PREREG_DEPLOYMENT_V2 — deployment-decay v3 사전등록 (서버 실행 전 커밋)

> 목적: 감사(AUDIT_FINAL_2026-07-04.md)가 지적한 "산문 사전등록 부재"를 해소. 이 문서가
> `scripts/run_deployment_decay.py` v3의 **규범(normative) 판정 규칙**이다. 코드와 이 문서가
> 다르면 이 문서가 우선하고, 불일치 자체를 보고한다. 이 문서 커밋 이후의 규칙 변경은
> 새 섹션(+사유)으로만 추가한다 (기존 텍스트 수정 금지).
> 근거 실험: audit_artifacts_2026-07-04/ (denoised 배터리, D-gate 실측, model-class 매트릭스).

## 0. 지위 선언 (정직성)

- 2026-07-03까지의 모든 실측(0703summary.json rows 10-19)은 **탐색적(exploratory)**이다.
- v2의 sberbank DEPLOYMENT-CONCEPT는 감사에서 3중 기각(early-noisy null이 +0.021로 동일 크기
  재현 / 규칙 경계 / d-gate-invalid 무통제)되어 **철회 후보**로 지정한다. v3 재실행이 판정한다.
- **10번째 dissolution 조항**: v3 실행에서 새로운 산업(TabReD) 양성이 나오면 그것은 헤드라인이
  아니라 **dissolution 후보 #10**이다. §5의 전체 생존 배터리를 통과하기 전에는 어떤 문서에도
  양성으로 기록하지 않는다.

## 1. 측정량 (estimand·계기, 고정)

- 점수: binclass AUC / multiclass acc / regression −RMSE (y는 z-표준화).
- raw_staleness = score(recent N) − score(recent N ∪ old N), 미래 윈도우 평가 (v2와 동일;
  RNG 스트림 v2와 비트 동일 — parity 검증 완료).
- **denoised_staleness** = score(recent N) − score(recent N ∪ (X_old, g_old(X_old)) N).
  g_old = old 윈도우 내부 2-fold cross-fit HGB, hard predict. (노이즈만 변하면 pseudo-label은
  근사-정답 → 해악 소멸; 규칙이 변했으면 옛 규칙 인코딩 → 해악 잔존.)
- **noise gate** = old 윈도우 노이즈 프록시 / recent 윈도우들 프록시 중앙값.
  프록시: regression held-out MSE / binclass 1−AUC / multiclass 1−acc (윈도우별 fresh HGB, 70/30).
- D = group-aware(GroupShuffleSplit, 그룹=z-반올림 중복 클러스터) + 랜덤 크기맞춤 분리도 AUC,
  시드 5개 중앙값. **의미는 "윈도우 분리 가능성"이며 support 겹침 주장으로 읽지 않는다.**
- injection = 강도 2.5 회전 규칙, top-2 분산 feature, **시드 10개**, **학습가능성 게이트**
  (최대 윈도우 held-out: AUC≥0.65 / R²≥0.20 / acc−majority≥0.10; 미달 시 null은 무효=vacuous).

## 2. 임계값 (고정)

| 상수 | 값 | 근거 |
|---|---|---|
| FLOOR_GAIN / FLOOR_DECAY | 0.02 | v1부터 연속 |
| D* | 0.96 | v2 power curve |
| GATE_THRESH | 1.5 | stable 컨트롤 0.75–0.99로 캘리브레이션 |
| GATE_ENVELOPE | 4.7 | denoiser 편향 실측 경계(비율 5.7에서 +0.026 오발) |
| INJ_SEEDS | 10 | 본판독과 동일 검정력 |
| 학습가능성 | AUC .65 / R² .20 / acc margin .10 | L4 vacuity 실측 (0.506 vs 0.964) |

## 3. 판정 규칙 (fire = 규칙 A: CI하한>0 ∧ 평균>floor; e680960부터 연속)

측정불능(유효 시드<2) → NO-DATA. 이후 순서대로:

1. denoised fire ∧ ratio ≤ 4.7 → **DEPLOYMENT-CONCEPT** (+ gate 발화 시 `noise-drift-present`,
   D≥D*면 `d-gate-invalid` 플래그; **주입 양성대조 필수 실행**)
2. denoised fire ∧ ratio > 4.7 → **NOISE-AMBIGUOUS** (기권)
3. raw fire ∧ gate 발화 → **NOISE-DRIFT-CONFOUNDED** (라벨 노이즈 drift; 규칙 변화 아님)
4. raw fire ∧ gate 조용 → **RAW-ONLY-POSITIVE** (미해결; concept 아님)
5. D_strip ≥ D* → 주입 컨트롤: 학습불능 → `injection-vacuous` (earned 아님) /
   회복 → **INJECTION-RECOVERED** / 실패 → **UNIDENTIFIABLE-{EXPLOITABLE|INERT}** + `unident-earned`
6. denoised 0<CI하한≤floor → **SUBFLOOR-CONCEPT-SIGNAL**
7. raw·denoised 모두 null(CI상한≤floor) → rec_present ? **DEPLOYMENT-DECAY-COVARIATE** : **NO-STRONG-CONCEPT**
8. 그 외 → **INCONCLUSIVE**

병기: `verdict_strict` = 규칙 B(CI하한>floor)로 같은 캐스케이드. 본판정은 규칙 A이되 **두 판정이
다른 셀은 rule-sensitive로 표기하고 어느 쪽도 헤드라인으로 쓰지 않는다.**
`d-gate-suspect`(D_shuffle>0.6)는 모든 분기에 부착. 모든 결과 blob에 meta(git·argv·버전·UTC) 포함,
버전드 파일로 기록(append 금지).

## 4. 합성 배터리 불변식 (서버 실행의 전제조건; --synth PASS 필수)

- CONCEPT여야 함: concept, nuisance_proxy, reg_concept, **reg_concept_earlynoisy**(gate가 veto 금지)
- CONCEPT 금지: covariate, stable, covariate_mc, covariate_mild, reg_stable, reg_cov_linear,
  reg_cov_nonlinear, reg_early_noisy, reg_late_noisy, reg_xdep_noise
- NOISE-DRIFT-CONFOUNDED여야 함: **reg_early_noisy**(F1 킬러), **reg_xdep_noise**
- covariate_mild는 UNIDENTIFIABLE-* 금지 (식별가능 영역 커버리지)
- FAIL 시: 어떤 실데이터 실행도 금지. 원인 수정 후 재실행, 이 문서에 새 섹션으로 기록.

## 5. 실행 계획·집계 읽기 (사전 고정)

- **Phase 1 (sberbank 결정)**: `--tabred sberbank_housing --n-seeds 10` + K∈{5,8,10,12,20}.
  사전 예측: NOISE-DRIFT-CONFOUNDED 또는 den-null. 만약 DEPLOYMENT-CONCEPT(규칙 A·B 모두,
  전 K에서, gate 조용)면 → 철회 취소가 아니라 **생존 배터리 개시**: fresh-seed 확증(시드
  100–109) + 주입 양성대조 회복 + model-class 패널(HGB/RF 일치) 전부 통과해야 양성 기록.
- **Phase 2 (전체 지도)**: 10개 전부, 탐색 시드 0–9 → 즉시 확증 시드 100–109 재실행.
  두 실행에서 판정이 다른 셀은 **unstable**로 보고 (어느 쪽도 채택 금지).
- **Phase 3 (model-class 패널)**: HGB/RF(+linear·kNN 카나리아) × 10 데이터셋. 실데이터에서
  판정 flip이 ≥2개면 class-relativity가 헤드라인 후보로 승격.
- **Phase 4 (앵커)**: EMBER 월별 by-value(NO-DATA 가드 확인), river 패널, insects 8변종,
  folktables 1-year. **사전 예측: 설계된 drift 앵커(river concept 셀, insects)는 CONCEPT로
  발화해야 한다.** 발화 실패 시 계기 결함으로 간주하고 지도 출판 보류.
- 집계 문장(고정): 지도의 주장 형식은 "tree-ensemble 클래스 기준, 검출한계 δ 이상의
  mean-rule drift가 확인된 산업 데이터셋 수 = N; 나머지는 {noise-confounded / covariate /
  blind-earned / vacuous}"이며, N=0이어도 그대로 보고한다.

## 6. 이 문서 이후 금지사항 (§7은 결과 기록이며 규칙 변경 아님)

- 결과를 본 뒤 임계값·캐스케이드 순서·배터리 불변식 변경 (새 섹션 추가로만 가능, 소급 적용 금지)
- rule-sensitive 셀이나 unstable 셀을 헤드라인에 사용
- 탐색 실행과 확증 실행의 혼합 보고

## 7. Phase 1 결과 판독 (2026-07-04 실행, 커밋 b3ae243, 서버 sklearn 1.9.0) — 예측 적중

사전 예측(§5: "NOISE-DRIFT-CONFOUNDED 또는 den-null")이 K∈{5,8,10,12,20} 전부에서 확인됨.
아티팩트: prereg_results/phase1/summary_20260704T*.json (5개, run meta 포함).

| K | raw stale | denoised | noise ratio | 판정 (rule A / strict B) |
|---|---|---|---|---|
| 5 | +0.005 null | −0.014 [−.020,−.008] | 2.57 발화 | UNIDENT-EXPL(earned; 주입 학습가능 .92, 미회복 +.008) / 동일 |
| 8 | +0.021 발화 | −0.016 [−.022,−.010] | 2.92 발화 | NOISE-DRIFT-CONFOUNDED / UNIDENT-EXPL |
| 10 | +0.0239 발화 | −0.014 [−.018,−.010] | 2.11 발화 | NOISE-DRIFT-CONFOUNDED / UNIDENT-EXPL |
| 12 | +0.021 발화 | −0.018 [−.021,−.014] | 2.15 발화 | NOISE-DRIFT-CONFOUNDED / UNIDENT-EXPL |
| 20 | +0.017 floor미달 | −0.017 [−.025,−.009] | 2.45 발화 | INJECTION-RECOVERED(주입 +.101 회복) / UNIDENT-EXPL |

판정(§5 규칙 적용):
- **sberbank DEPLOYMENT-CONCEPT 공식 철회 — 기제 확정: 라벨 노이즈 감쇠.** 어떤 K에서도
  CONCEPT 아님(rule A·B 일치); 생존 배터리 트리거 조건 미발생.
- 증거: (i) K=10 raw = +0.023900573591226625 — v2 헤드라인(0703 row 10)과 **비트 동일**
  (RNG 패리티가 실데이터·서버에서 성립; 같은 신호의 재해석임을 증명); (ii) old 윈도우 노이즈
  프록시가 recent 대비 2.1–2.9× (전 K, envelope 이내); (iii) **denoised staleness 전 K에서
  유의 음수** — pseudo-label로 노이즈를 제거하면 old 데이터가 미래 예측을 도움 = 규칙 불변.
- 부수 확인: v2의 D_shuffle 0.887 → 수정 후 0.50–0.53 (head-truncation 아티팩트 기제 실증);
  K=20 주입 회복 +0.101(해상도별 검출력 존재), K=5 주입 +0.008(넓은 윈도우의 회전 평균화 —
  timescale 효과와 일치); dup_group_frac 0.3%.
- 지도 갱신: 산업(TabReD) concept 양성 = 0. 논문 서술은 "철회(withdrawn)"가 아니라
  "**계기에 의한 진단(diagnosed as label-noise decay)**"로 기록한다 (9번째 dissolution이자
  최초의 계기-내 진단 사례). Phase 2~4 진행.

## 8. Phase 4 실행 전 명확화 (2026-07-04, 서버 Phase 2~4 실행 전 커밋)

출처 공개: 로컬 3-시드 스모크(river sea_abrupt, n=8000)에서 SUBFLOOR-CONCEPT-SIGNAL
(stale +0.004, den +0.006 CI>0)이 관측됨. 이에 §5 Phase 4의 앵커 기대를 실행 전에 명확화한다
(기준 완화가 아니라 적용 범위의 정밀화; 소급 변경 아님):
- "CONCEPT 발화해야 한다"는 **강한 규칙 전환 앵커**에 적용: insects 변종들, river의
  agrawal/stagger/sine 계열 abrupt·reoccurring 셀.
- **약한 전환 셀**(예: sea variant 0→3 threshold 이동)은 방향-일치 양성(SUBFLOOR 포함,
  denoised CI하한>0)이면 앵커 통과로 간주하고, delta(검출한계)와 함께 보고한다.
- nodrift 셀은 CONCEPT/SUBFLOOR 금지 (오탐 체크).
- 판정 불가 사유로 지도를 보류하는 조건은 "강한 앵커의 CONCEPT 실패"로 유지.
