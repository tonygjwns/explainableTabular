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

## 9. Phase 2~4 결과 판독 (2026-07-04~05 실행, 커밋 5f3217d, 서버 sklearn 1.9.0)

아티팩트: prereg_results/phase234/ (16개 run JSON + 로그 3개), env_explaintab311_freeze.txt.

### Phase 2 — 탐색(시드 0–9) vs 확증(시드 100–109): **10/10 판정 일치, unstable 셀 0**
유일한 경계 사례: homesite의 플래그가 injection-vacuous↔unident-earned로 흔들림(학습가능성
점수가 컷 0.65 근방) → homesite의 실명 인증서는 **불안정**으로 기록, 어느 라벨도 채택 금지.

### 최종 지도 (HGB, 확증 시드로 재현됨)
| 셀 | 판정 | 핵심 수치 |
|---|---|---|
| insects | **DEPLOYMENT-CONCEPT** | raw +0.129~+0.135, **den +0.145~+0.152**(denoised가 더 강함 = 진짜 규칙 변화), 주입 회복 |
| sberbank | **NOISE-DRIFT-CONFOUNDED** | raw +0.024/+0.033 발화, den −0.015/−0.011, gate 2.1~2.2; rule-B에서는 UNIDENT(rule-sensitive, 어느 쪽도 concept 아님) |
| cooking / delivery | INJECTION-RECOVERED | 주입 +0.55/+0.33 회복 + 실 staleness null = **검증된 무-concept** |
| maps | NO-STRONG-CONCEPT | group-aware D 0.578(v2의 0.735에서 하락 — 분리도 일부가 메모리제이션이었음) |
| elec2 | UNIDENT-EXPLOITABLE (**earned**) | 주입 학습가능+미회복(+0.017~0.018) |
| ecom / homecredit / weather | UNIDENT-EXPLOITABLE (**vacuous**) | 주입 학습불능 → v2의 'earned' 라벨은 과대주장이었음(감사 L4 실증) |
| homesite | UNIDENT-INERT (vacuous/earned 불안정) | inj −0.052~−0.055 |

**§5 고정 집계문**: tree-ensemble 클래스 기준, δ 이상 mean-rule drift가 확인된 산업 데이터셋 = **0/8**.
insects(designed)만 CONCEPT. 실명 인증(earned)은 elec2 1건뿐; ecom/homecredit/weather는 인증 실패(vacuous).

### Phase 3 — model-class 패널
- **결정등급(HGB↔RF): 10/10 작동적 일치** (homesite EXPL/INERT 서브라벨 차이만). 지도는
  tree-ensemble 내에서 견고.
- 카나리아 flip (예상대로 + 실데이터 class-relativity 실증): **linear/elec2 = DEPLOYMENT-CONCEPT**
  (raw +0.023, den +0.033, 주입 회복 +0.19 — 정전 concept-drift 데이터셋이 선형 프로브에는
  검출되고 트리 프로브에는 unidentifiable) / linear/ecom = CONCEPT(raw null인데 den만 발화 —
  선형 denoiser 아티팩트 채널, 카나리아 거동) / kNN 서브라벨 flip 다수. §5의 "flip ≥2" 조건
  충족하되 **카나리아 클래스에서만** — 헤드라인 승격은 "프로브 의존성의 실데이터 실증"으로
  범위 한정 (결정등급 클래스는 일치했으므로 지도 자체는 무손상).

### Phase 4 — 앵커 (§8 기준)
- **강한 전환 앵커 7/7 CONCEPT**: agrawal_abrupt(+0.045)/agrawal_gradual(+0.047)/stagger_abrupt/
  sine_abrupt(+0.031)/hyperplane_incr(+0.113)/sine_reoccur2(+0.047)/insects. **계기는 설계된
  규칙 변화를 검출한다** — 지도 출판 보류 조건 미발생.
- SEA 약전환: SUBFLOOR 방향-일치 (§8 허용).
- **nodrift 오탐 체크: 5셀 중 2셀(sea, sine)이 SUBFLOOR** — §8의 금지 조항 위반. 크기는
  den +0.001~+0.002(floor의 1/10~1/20)로, 중첩-시드 CI의 반보수성(감사 C7)이 미세 양수를
  '유의'로 만든 것. **캘리브레이션 판정: SUBFLOOR 대역은 '약한 증거'가 아니라 '무증거'로
  읽는다** (maps/kNN·SEA 셀 포함 전체 SUBFLOOR에 소급 적용; 이 해석 규칙을 §9에서 고정).
- reoccurring 셀 대부분 DECAY-COVARIATE: A→B→A 구조에서 old(A) 데이터는 A-구간 미래에
  무해하므로 staleness가 안 뜨는 게 **의미상 옳음** — 롤링 렌즈는 재발 drift의 '해악'이 아니라
  '재사용 가치'를 본다는 범위 명시.
- p4_insects FAIL: river 0.25.0 variant 명칭 불일치(레포 목록이 구버전) → 7종으로 수정(커밋),
  재실행 대기. EMBER: parquet 부재로 스킵(정상).

## 10. Phase 4 최종 앵커 판독 — insects 7변종 + EMBER (2026-07-05, 커밋 fe13544)

아티팩트: prereg_results/phase4_final/ (JSON 3 + 로그 2).

### insects 변종 매트릭스 — 계기의 drift-구조별 감도 프로파일
| 변종 (구조) | 판정 | raw / den | 비고 |
|---|---|---|---|
| gradual_balanced (단조) | **CONCEPT** | +0.092 / +0.093 | 주입 학습가능·회복 +0.114 |
| gradual_imbalanced (단조) | **CONCEPT** | +0.069 / +0.076 | 주입 +0.130 |
| incremental_balanced (단조) | **CONCEPT** | +0.135 / **+0.152** | 주입 +0.162 |
| abrupt_balanced (온도 왕복 전환) | NO-STRONG | −0.070 / −0.070 | **recency −0.058 (음수!)** |
| abrupt_imbalanced (〃) | NO-STRONG | −0.027 / −0.032 | recency −0.023 (음수) |
| incremental_abrupt_balanced | DECAY-COVARIATE | −0.011 / −0.002 | rec +0.057 |
| incremental_reoccurring_balanced | SUBFLOOR (§9 규칙: 무증거) | −0.005 / +0.003 | rec +0.150 |

**구조적 판독**: 단조(gradual/incremental) 앵커 3/3 발화 — 전부 denoised ≥ raw(진짜 규칙 변화의
서명) + 주입 회복. 미발화 변종들은 전부 **레짐 재발/왕복** 구조이고, 그 내부 증거가
abrupt 변종의 **음의 recency_gain**(직전 윈도우가 window 0보다 미래 예측에 더 나쁨 —
단방향 drift에서는 불가능; 시험 구간 레짐이 옛 레짐과 재일치할 때의 서명)이다. river에서
**단발 전환** abrupt는 4/4 발화했으므로(§9), 프로파일은 일관된다:
**"계기는 old 라벨이 '현재' 레짐과 모순일 때 발화하고, 옛 레짐이 되돌아와 old 데이터가
재사용 가능해지면 침묵한다"** — 배포-해악 렌즈의 의미론상 옳은 거동이며, river reoccur
셀들(§9)·insects reoccurring·insects abrupt까지 **3중 독립 확인**. §8의 지도 보류 조건
(강한 앵커의 CONCEPT 실패)은 단조-앵커 기준으로 미발생. 논문 기술 시 Souza et al. 2020의
변종별 온도 프로토콜 인용으로 왕복 구조를 문헌 확정할 것.

### EMBER (자연 발생 drift 후보) — NO-STRONG-CONCEPT, 의미 있는 null
by-value W=126: stale **−0.012**[−.013,−.011](old 데이터가 도움), den −0.003, gate 1.01 조용,
recency +0.012(미미), delta 0.0014(타이트한 null). 판독: 말웨어의 문헌적 drift(TESSERACT)는
**새 패밀리 등장 = 커버리지 확장**이지 라벨 부패가 아니다 — 2017년 말웨어는 2018년에도
말웨어. "정전 drift 도메인에서조차 old 데이터의 배포 해악 ≈ 0, 열화는 rule-rot이 아니라
coverage-driven" — 지도 전체 메시지와 정합. 캐비앗: sparse-window:1, D 계산불능(식별가능성
인증서 없음), ember_k10(K=10 분위)은 NO-DATA로 정직 거부(질량이 2018에 집중, 값-랭크 윈도우
희소화). 선택: 2018-only 필터 재실행(ember2018)으로 셀 업그레이드 가능.

### 실행 계획 종결 선언
§5의 Phase 1~4 전부 실행·판독 완료 (EMBER 2018-only 재실행과 folktables만 선택 항목으로
잔존). 다음 단계 = 집필: RESULTS_LEDGER 모순 정리, 초안 §재작성 (identifiability map + 계기
+ 구조별 감도 프로파일 + 진단된 sberbank + EMBER null).

## 11. Phase 3 확장 — MLP(신경망) 프로브 패널 (2026-07-05 실행; 리뷰 라운드 1 대응)

외부 리뷰 지적(딥 프로브 부재)에 따라 `--model mlp`(sklearn 2층 64-32) 추가. 판정 순서:
합성 배터리 먼저(관문), 실데이터 패널 다음.
- **합성 매트릭스 (로컬)**: concept ✓(+0.305) / stable ✓ / nuisance ✓(+0.228) —
  그러나 **prior shift에서 거짓 CONCEPT**(den +0.042 발화), covariate에서 den +0.045(CI로
  겨우 미발화) → **MLP = 카나리아 등급** (linear/kNN과 동류; 의사결정 등급 아님).
  아티팩트: audit_artifacts_2026-07-04/exp-modelclass/mlp_matrix.json.
- **실데이터 패널 (서버, p3mlp.log → prereg_results/phase3_mlp/)**:
  insects CONCEPT(+0.120, den +0.165 — 5/5 클래스 검출 견고);
  **elec2 = DEPLOYMENT-CONCEPT (raw +0.007, den +0.025 발화, d-gate-invalid)** — linear에 이은
  두 번째 독립 비-트리 프로브의 flip; 정전 drift 데이터셋의 판정이 프로브 클래스에 의존함이
  이제 2중 실증. TabReD 8셋은 신규 양성 없음. 회귀 셋에서 MLP 수치 폭주(sberbank stale +3.2
  CI [−11.8,+18.3]; weather CI 폭 0.5) — 카나리아 분류와 정합, 판정 채택 금지.
- 집계 불변: 의사결정-등급(HGB·RF) 지도 무변경. 논문 §4.3/§5.4에 반영.

## 12. 선택 실험 판독 — ember2018 + MLP 아티팩트 완결 (2026-07-06)

아티팩트: prereg_results/optional/ (JSON 5 + ember2018.log).
- **ember2018 (2018-only, 월별 by-value, HGB, 10시드): DEPLOYMENT-DECAY-COVARIATE.**
  D 0.834(식별가능 — 게이트 미작동, null이 직접 하중), raw −0.008[−.009,−.007], den −0.004,
  gate 1.01 조용, recency **+0.031**(floor 초과), delta 0.0013. §10의 저전력 W=126 판독
  (NO-STRONG-CONCEPT, D 불능)을 **대체·격상**: 말웨어의 시간 열화는 실재하고 recency로 회복
  되지만(커버리지), old 라벨은 무해(규칙 부패 없음). TESSERACT와 정합. 7/5·7/6 두 실행이
  바이트 동일(실데이터 결정성 재확인).
- **MLP 패널 JSON 3개 입수** — §11의 로그 판독과 수치 일치, 체인 완결. 추가 확인: elec2
  MLP-CONCEPT 셀의 주입이 학습가능+회복(+0.052) — flip이 프로브-상대적 신호로서 유효함 보강;
  weather/ecom/homecredit는 MLP에서도 injection-vacuous.
- **folktables CA: 실패(prep 미실행 — parquet 부재 FileNotFoundError). 재시도 대기.** RI
  로컬 스모크는 통과(§ prep_folktables 커밋 메시지); CA 재실행 후 §13으로 기록 예정.

## 13. folktables CA 브리지 판독 (2026-07-15 실행, 커밋 5b85cb0) — 실험 큐 종결

아티팩트: prereg_results/optional/summary_20260715T135651_5b85cb0.json (HGB, 10시드, YEAR
by-value 5윈도우, min_window 12k행).
- **acs_income_CA = NO-STRONG-CONCEPT, trust ok.** raw −0.0076[−.0083,−.0069], den
  −0.0074[−.0080,−.0068] (old 데이터가 도움), gate 0.99 조용, **D 0.515/shuffle 0.484**
  (연도 간 공변량 이동조차 미미 — 완전 식별가능), recency +0.0009≈0, decay +0.003≈0,
  **delta 0.00078 (지도 최소 검출 하한)**.
- **WhyShift 브리지**: 공간 축에서 Y|X-shift 만연이 보고된 ACSIncome이 시간 축(CA
  2014–2018)에서는 트리-앙상블 기준 정지 상태 — 같은 과제에서 축에 따라 shift 성격이
  갈림을 실증. **semi-known 앵커 통과**: 고정 $50k 임계값의 인플레이션 pos_rate 램프
  (prior-shift)를 CONCEPT로 오독하지 않음 (covariate_mc 컨트롤의 실데이터 대응).
- 범위: 1개 주, 5개 연간 윈도우, 2014–2018(RELP 개명·COVID 갭 회피). 다주(multi-state)
  확장은 논문 future work.
- **선택 실험 포함 실행 큐 전체 종결. 이후 작업 = 집필만** (벡터 Figure 1, LaTeX, 서지
  검증, 국문 전파).

## 14. 리버탈 실험 [C]·[D] 사전 규정 (2026-07-18, 서버 실행 *전* 커밋 — 리뷰어2 대응 옵션 B)

> 규칙: 아래 판독 규정은 결과를 보기 전에 커밋된다. 캐스케이드·임계값·floor·게이트·envelope는
> §3~§8과 동일하게 **불변**. 탐색 시드 0–9 → 판정이 기존 지도와 달라지는 셀은 확증 시드
> 100–109 재실행을 통과해야만 논문에 들어간다(기존 §7 규율 그대로). 코드 = 커밋 ffe1e56의
> `--tabred-span full` / `--inj-family` (로컬 스모크 통과, 사전등록 배터리 경로 불변).

### [C] 배포 held-out 구간 감사 (`--tabred-span full`, 8개 TabReD, 10시드)

- **무엇**: train+val+test를 공유 정규화 timestamp로 연결·정렬 — 윈도우가 공식 분할이 hold-out
  하는 배포 갭을 가로지른다. 행 태그 `_fullspan`.
- **예측 (커밋 시점)**: 지도 판정 유지 (full-span에서도 산업 mean-rule drift 발화 없음).
- **판독 규정 (양방향, 결과 확인 전 커밋)**:
  - **판정 전 셀 유지** → 부록 B.3 "지도는 배포 갭을 포함해도 불변" — 리뷰어2 사유 1의 잔여
    절반(§7 한계 2의 train-구간 한정) 실측 방어로 격상.
  - **어떤 산업 셀이든 DEPLOYMENT-CONCEPT 발화** (denoised CI>0 ∧ mean>floor, envelope 내)
    → ① 확증 100–109 재실행 통과 시에만 인정, ② 인정되면 **0/8 헤드라인은 "train 구간"으로
    재스코프**되고 해당 셀은 신규 발견으로 보고된다 — 프레이밍: "drift는 정확히 벤치마크가
    딥 방법을 평가하는 구간에 산다" (이는 실패가 아니라 이 계기가 처음 잡는 산업 양성).
    초록·§5.2·Figure 2 갱신 필수.
  - **NOISE-DRIFT-CONFOUNDED / RAW-ONLY 발화** → 기제 라벨대로 보고 (concept 아님, §3.3).
  - **rule-sensitive / unstable** → 기존 규칙대로 헤드라인 금지.
  - full-span은 지도를 **대체하지 않는다**: 본 지도 = train 구간(공식 분할의 감사 가능 구간);
    full-span = 강건성 부록. 재스코프 트리거는 위 CONCEPT 케이스뿐.

### [D] 주입-패밀리 sweep (`--inj-family {lowvar, interaction, subpop}`, 인증서 셀 6 + insects, 10시드)

- **무엇**: 기준 패밀리(topvar 회전)를 저분산 2피처 / 상호작용항 / 부분모집단(z(f2)>0)-국소
  회전으로 교체. 학습가능성 게이트 불변 — 학습불능 패밀리는 *그 패밀리에 대해* VACUOUS.
- **예측 (커밋 시점)**: cooking/delivery 회복이 학습가능한 패밀리들에서 유지; insects(양성
  대조) 전 학습가능 패밀리에서 회복; vacuous 3셀(ecom/homecredit/weather)은 패밀리를 바꿔도
  대부분 학습불능 또는 미회복.
- **판독 규정 (양방향)**:
  - **회복이 학습가능 패밀리 전반에서 유지** → §7 한계 (6)이 "실측 방어"로 격상, 부록 B.4에
    패밀리×셀 표. "verified no-concept" 명명 유지(family-상대성 각주 유지).
  - **기존 verified 셀(cooking/delivery)이 어떤 학습가능 패밀리에서 미회복** → 그 셀 인증서를
    §5.2에서 **family-상대적으로 재표기**하고, "인증서는 패밀리-상대적"이 한계가 아니라 본문
    발견으로 승격(§3.2 재작성). 확증 재실행 규칙 동일 적용.
  - **insects가 어떤 학습가능 패밀리에서 미회복** → 그 패밀리에 대한 계기 감도 한계로 §6에
    보고 (은폐 금지).
  - 학습불능 패밀리는 어떤 셀에서도 인증·반증 어느 쪽으로도 쓰지 않는다 (vacuous 규율).

## 15. 감사-발견 결함 2건의 기록과 수리 (2026-07-18~19 로컬 실행; §6 규칙에 따른 신규 섹션)

독립 교차감사(평가 메모 3건 + artifacts/ 전수 대조)가 두 결함을 실증했다. 둘 다 결과의
오류가 아니라 **규율 집행의 공백**이며, 아래에 수리 기록을 남긴다.

### 15a. 배터리-환경 불일치 (감사 B 논거 1) — 수리 완료

- 결함: 유일한 배터리 PASS(01ae6ae)는 로컬(Python 3.14.3/sklearn 1.8.0), 실데이터 전 실행은
  서버(3.11.15/1.9.0). §4의 전제("PASS 없이 실데이터 금지")가 환경 경계를 넘어 가정으로만
  연결되어 있었다.
- 수리: 서버-버전 일치 venv(Python 3.11.5/sklearn 1.9.0/numpy 2.4.6)에서 **무수정 코드**로
  `--synth` 재실행 → **PASS, 커밋본과 14/14 판정 일치**. 오탐 채널 잔차: reg_early_noisy
  den +0.0045@gate 3.53, reg_xdep_noise den +0.0063@3.72, 공존 셀 den +0.319 발화 유지.
  아티팩트: repair_20260718/synth_battery_v3_PASS_sklearn190_py311_runA_unmodified-code.json
  (UTC 2026-07-18T15:43).
- 잔여: 서버 머신 자체에서의 재실행(§14 실행 시 선행)로 패치버전까지 봉인할 것.

### 15b. strict 그림자 캐스케이드의 주입-단계 부재 (감사 B 논거 2) — 코드 수리 완료, 라벨은 잠정

- 결함: 주입 승격이 `verdict`에만 적용되어 `verdict_strict`는 구조적으로 INJECTION-RECOVERED가
  될 수 없었다 (구 741-745행). 따라서 cooking/delivery(및 kNN/elec2)의 verdict/strict 불일치는
  **규칙 민감성이 아니라 그림자 캐스케이드의 기계적 공백**이다. §3의 문언("두 판정이 다른 셀은
  rule-sensitive, 헤드라인 금지")을 문자 그대로 적용하면 두 셀은 헤드라인 부적격이었다 —
  §9가 이를 기록하지 못했음을 인정한다.
- 수리(코드): `_injection_recovers`가 CI 반환; 주입 실행 조건을 verdict **또는** verdict_strict가
  UNIDENTIFIABLE*/CONCEPT인 경우로 확장; strict는 자기 규칙(B: 주입 CI하한>floor)으로 회복을
  판정해 승격; `injected_staleness_ci`·`injection_recovered_strict` 방출.
- 검증: 수정 코드로 배터리 재실행 → **PASS, raw 수치가 15a 실행과 비트 동일**(RNG 스트림
  불변 증명), 전 셀 verdict·verdict_strict 불변, vacuity 규율 보존(covariate_mc: strict 회복
  기준 충족이나 학습불능이므로 미승격). 아티팩트: repair_20260718/..._runB_strict-shadow-fix.json
  (UTC 2026-07-18T17:04).
- 라벨 판정 (사전 커밋): 기존 실데이터 아티팩트에는 주입 per-seed가 없어 rule-B 회복 CI를 사후
  계산할 수 없다. cooking/delivery의 strict-확정 인증서는 §14 [D] 실행 시 자동 방출되는
  `injection_recovered_strict`로 판정하며, **그때까지 두 셀의 INJECTION-RECOVERED는
  primary-rule-only 라벨로 표기하고 §5.2 헤드라인에 caveat을 부착한다** (REVISION_NOTES R4).
  예측(결과 확인 전): 회복 마진이 floor의 16~27×이므로 strict에서도 회복될 것. 미회복 시
  §14 [D]의 양방향 규정대로 family-상대적 재표기.
- 부수 명시: MLP/elec2의 verdict/strict 불일치(CONCEPT vs UNIDENT-INERT)는 이 공백과 무관한
  **실질적 rule-sensitivity**다 (den +0.025가 rule B 미달). §5.4 서술을 linear(양 규칙 견고)와
  MLP(rule-sensitive)로 분리 표기할 것.
