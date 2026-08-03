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

## 16. [D] 주입 스윕의 설계 보정과 판독 규정 (2026-07-31, 서버 실행 *전* 커밋)

> 지위: §14 [D]의 **확장이며 대체가 아니다.** 캐스케이드·판정 임계값(floor 0.02 / D\* 0.96 /
> gate 1.5 / envelope 4.7 / 학습가능성 0.65·0.20·0.10)·시드 프로토콜(탐색 0–9, 확증 100–109)은
> §2~§3과 동일하게 **불변**. 이 섹션이 추가하는 것은 (a) 스윕 축의 보정, (b) §14가 규정하지
> 않은 채 남긴 판독 분기, (c) 다중비교·검정력 회계, (d) 실행 전 예측이다. 전부 결과를 보기
> 전에 커밋된다.
>
> 계기: 독립 외부 적대 검토 2건 + 저장소 전수 대조(`INJFAMILY_SWEEP_PLAN_2026-07-31.md`).

### 16.1 설계 보정 — 스윕은 1축이 아니라 2축이다 (family × cols)

**결함**: §14가 등록한 4개 패밀리 중 **3개(topvar / interaction / subpop)가 같은 열을 읽는다**
(`np.argsort(-Xf.std(0))`의 상위 2열). 그런데 ecom / homecredit / weather의 주입 공허 원인으로
진단된 것이 바로 **그 상위-분산 열의 두꺼운 꼬리**다(감사 L4): heavy tail을 z-표준화하면 행의
~99%가 0 근처에 몰리고 `N(0,.3)` 항이 score를 지배해 심어진 라벨이 동전던지기가 된다
(윈도우 내 AUC 0.51–0.53 실측). 따라서 **그 열 위에서 interaction/subpop을 시험하면
"상호작용 규칙을 못 본다"와 "이 열이 아무것도 못 나른다"를 구분할 수 없다** — 경화 단계가
위음성을 낸다.

**보정**: 열 선택을 규칙 기하와 분리해 `--inj-cols {auto,hi,lo}`로 노출한다.
- `hi` = 분산 내림차순(사전등록 기준 캐리어) / `lo` = 비퇴화 분산 오름차순
- `auto` = v1의 결합(lowvar→lo, 그 외→hi). **`auto`에서 topvar·lowvar·subpop의 주입 라벨은
  v1과 비트 동일** — 커밋된 배터리와 지도가 그대로 재현된다.
- `interaction`은 **의도적으로 변경**: v1은 `z(f0·f1)`을 `z(f1)`에 대해 회전시켜 두 축이
  종속이었다(두 번째 축이 첫 축의 인자). 이제 독립한 제3열에 대해 회전한다. 이 기하는 실데이터에서
  **한 번도 실행된 적이 없으므로** 커밋된 아티팩트가 바뀌지 않는다.

**실행 조합 (4)**: `topvar@hi`(= 기존 지도, 대조) / `lowvar@lo` / `interaction@lo` / `subpop@lo`.
`interaction@hi`·`subpop@hi`는 진단용으로만 기록하고 판정에 쓰지 않는다(16.7).

**검증**: `scripts/smoke_test_inj_family.py` — 파리티(3 task × 2 기하에서 라벨 비트 동일),
축 독립성(|corr| v1 0.769 → 0.013), 캐리어 분리(heavy-tail 기하에서 hi 실패 AUC 0.566 /
lo 통과 0.957), 게이트 상수 불변. 산출물을 커밋한다 — v1 스모크가 산출물을 남기지 않아
검증 자체가 산문으로만 존재했던 결함(감사)을 여기서 닫는다.

### 16.2 범위 명시 — [D]가 할 수 없는 것 (사전 고정)

`--inj-family` / `--inj-cols`는 `_injection_recovers`에만 들어가고, 실 판정을 결정하는
`staleness_harm` / `denoised_staleness`는 그 전에 **실 라벨**로 계산된다. 따라서:

> **패밀리·캐리어를 바꿔도 실데이터 판정 수치는 비트 단위로 불변이다.**
> [D]는 산업 셀을 발화시킬 수 없고, §7 한계 3(자연 발생 산업 양성 부재)을 해소할 수 없다.
> [D]가 바꿀 수 있는 것은 **인증서의 상태**뿐이다.

이 문장을 사전 고정하는 이유: 외부 검토 2건이 독립적으로 "셀이 발화하면 실데이터 양성을
얻는다"고 서술했고, 이는 기제상 불가능하다. 결과 해석 시 같은 오독을 막는다.

### 16.3 조합 라벨 (조합 = 셀 × family × cols)

| 라벨 | 조건 (아티팩트 필드) | 의미 |
|---|---|---|
| **VACUOUS** | `injection_learnable == false` | 그 조합에 대해 인증·반증 어느 쪽으로도 사용 금지 |
| **EARNED-BLIND** | `learnable == true` ∧ rule-A 미회복 | 그 신호 클래스에 대해 기하가 진짜로 눈멀었다. **분모를 복구하지 않는다** |
| **RECOVERED** | `learnable == true` ∧ rule-A 회복 (`injected_staleness_ci[0] > 0` ∧ `mean > 0.02`) | 기하에 검정력이 있었다 → 그 셀의 실 null이 정보량 있음 = verified no-concept |

rule-B(strict) 회복은 `injection_recovered_strict`로 별도 기록한다. rule-A만 회복하면
§15b 규율대로 **primary-rule-only** caveat를 유지한다.

### 16.4 셀 라벨 집계 — §14의 "전반에서 유지" 정의

§14는 "학습가능 패밀리 전반에서 유지"의 "전반"을 정의하지 않았다. 아래로 고정한다.

- **인증서 유지(무조건)**: 학습가능한 **모든** 조합이 RECOVERED. 근거: "verified no-concept"은
  전칭 주장이므로 학습가능한 반례 하나가 무조건적 읽기를 반증한다. §14의 기존 문언
  ("**어떤** 학습가능 패밀리에서 미회복 → family-상대적으로 재표기")과 정합.
- **부분 인증**: 학습가능 조합 중 일부만 RECOVERED → §5.2 라벨을
  `verified no-concept (m/n families)`로 재표기하고, 어느 조합이 실패했는지 본문에 명시.
- **셀 라벨에는 항상 분모를 붙인다**: `EARNED-BLIND (2/4 조합 학습가능)` 형식. 학습불능 조합을
  분모에서 빼고 보고하는 것을 금지한다(16.6의 방향성 편향 대응).

### 16.5 분모 복구 분기 — §14가 규정하지 않은 판독 (신규)

§14는 "현재 공허한 셀이 새 조합에서 **회복되는** 경우"의 판독 규정을 두지 않았다. 그것이
헤드라인의 판독 가능 분모를 늘리는 유일한 분기이므로 여기서 사전 규정한다.

1. 탐색(0–9)에서 어떤 조합이 `learnable ∧ rule-A 회복` → **확증(100–109) 재실행 필수.**
2. 확증에서도 동일 → 그 셀의 판정을 `INJECTION-RECOVERED`로 확정하고, §5.2에서
   *vacuous* → **verified no-concept (family-상대 표기)** 로 승격한다. **헤드라인의 판독 가능
   셀 수가 늘어난다** (예: "감사 8, 판독 가능 4" → "판독 가능 5").
3. 탐색/확증 불일치 → **unstable**, 어느 라벨도 채택 금지 (§9 homesite 선례 그대로).
4. **동시 명시(필수)**: 회복은 *그 셀의 실 null이 정보량을 갖는다*는 뜻이지 *drift가 없다는
   새 증거*가 아니다. `staleness_harm` / `denoised_staleness` 수치는 불변이다(16.2).
   특히 homecredit이 회복될 경우, 그 셀의 denoised +0.005(floor의 1/4)는
   **여전히 floor 미만이며 판정을 바꾸지 않는다** — 다만 그 null이 처음으로 "못 봤다"가 아니라
   "봤는데 없었다"가 된다. 이 구분을 §5.2 본문에 명시할 것.

### 16.6 다중비교·검정력 회계 (기존 프로토콜에 없던 항목)

임계값을 바꾸지 않는다. 대신 **회계를 공개하고, 확증 재실행을 다중성 통제로 명시**한다.

| 방향 | 규칙 | 단일 조합 위양성 | 조합 4개 max-over | 확증 재실행 후 |
|---|---|---|---|---|
| 인증서 **상실** (어떤 학습가능 조합에서 미회복) | 16.4 | ≤2.5% (rule-A 단측) | ≈9.6% | ≈0.25% |
| 인증서 **획득** (어떤 학습가능 조합에서 회복) | 16.5 | ≤2.5% | ≈9.6% | ≈0.25% |

- 확증 후 수치는 탐색·확증이 **독립 시행일 때의 값**이다. 두 실행은 같은 데이터에서 서브샘플·
  윈도우 경계·학습셋만 다시 뽑으므로 완전 독립이 아니다. **상한이 아니라 방향 지시로만 읽는다.**
- 따라서 **판정이 지도와 달라지는 모든 셀에 확증 재실행을 의무화**한다(§14 규율 유지). 이것이
  이 스윕의 다중성 통제이며, 별도의 α 보정을 도입하지 않는다(임계값 불변 원칙).
- 검정력: 조합 4개는 진짜 family-의존성 검출력을 크게 올린다. 그것이 이 스윕의 **목적**이며,
  위 표의 위양성 상승은 그 대가로 사전 수용된다.

### 16.7 경계 조항 2건 (실행 전 고정)

**(a) 하이브리드 캐리어 — `interaction@hi` 귀속 금지.** 보정된 `interaction`은 제3열을 두 번째
축으로 끌어온다. 스모크 실측: heavy-tail 기하에서 `topvar@hi`·`subpop@hi`는 게이트 탈락
(AUC 0.566 / 0.527)인데 `interaction@hi`는 통과(0.945)했다 — 제3열이 깨끗했기 때문이다.
따라서 **`interaction@hi`의 회복은 어떤 캐리어에도 귀속하지 않는다.** 기록만 하고 16.4의
셀 라벨 집계에서 제외한다. 실행 조합에서 `interaction@lo`만 쓰는 이유가 이것이다.

**(b) 학습가능성 경계값 — 컷 ±0.05는 단독 사용 금지.** homesite의 vacuous↔earned 불안정은
`injection_learn_score`가 0.516↔0.694로 컷 0.65를 걸쳐 흔들린 것이 전부다(§9). 앞으로
`|lscore − 0.65| ≤ 0.05`(회귀는 `|R² − 0.20| ≤ 0.05`)인 조합은 **boundary**로 표기하고,
인증·반증 어느 방향으로도 단독 근거로 쓰지 않는다. 다른 조합이 명확히 학습가능일 때만 셀
라벨이 결정된다. 이는 임계값 변경이 아니라 **실증된 불안정 구간에 대한 사용 제한**이다.

### 16.8 사전 예측 (결과 확인 전 커밋)

기제 가설: 공허 3셀의 실패 원인은 상위-분산 열의 heavy tail이므로, **`lo` 캐리어만이 뒤집을
후보**이고 `hi`를 쓰는 조합은 topvar의 실패를 그대로 물려받는다.

| 셀 | 최빈 예측 | 확률 | 근거 |
|---|---|---|---|
| ecom_offers | VACUOUS | 45% (EARNED-BLIND 40 / RECOVERED 15) | lscore 0.527/0.516/0.516 전 프로브 무작위 수준; D=1.0000 |
| homecredit_default | EARNED-BLIND | 45% (VACUOUS 35 / RECOVERED 20) | lscore 0.633/0.637/0.617 — 컷에 가장 근접; D=0.9998 |
| weather | VACUOUS | 55% (EARNED-BLIND 35 / RECOVERED 10) | 유일하게 lscore 음수(R² −0.022); 회귀 게이트 R²≥0.20까지 거리가 멀다 |
| homesite_insurance *(탐색 추가)* | EARNED-BLIND | 45% (VACUOUS 30 / 불안정 20 / RECOVERED 5) | 컷을 걸침; topvar 주입이 **음수 회복**(−0.052) |
| cooking_time *(대조)* | 인증 유지 | **70%** | 회복 +0.546~+0.577 = floor의 27×; D 0.9656으로 인증서 셀 중 최저 |
| delivery_eta *(대조)* | 인증 유지 | **60%** | 회복 +0.318~+0.332 = floor의 16×; D 0.9990 |
| elec2 | EARNED-BLIND 유지 | 75% | lscore 0.976(최고 학습가능)인데 회복 +0.018(floor 미달) |
| insects *(양성대조)* | 전 학습가능 조합 회복 | 85% | D 0.8436, 회복 +0.162 |
| river_agrawal_abrupt *(구현대조)* | 4/4 회복 | 95% | D≈0.498, lscore 0.956 |

집계 예측: 공허 3셀 중 **적어도 1개가 학습가능해질 확률 ≈70%**, **적어도 1개가 회복(분모 복구)될
확률 ≈35%**, **cooking/delivery 중 적어도 하나가 인증서를 잃을 확률 ≈50%**.

**사전 커밋된 분기(중요)**: cooking·delivery가 모두 인증서를 잃으면 판독 가능 분모가 0/2로
내려간다. 그 경우 **지도를 구하려는 어떤 사후 조정도 하지 않는다.** §14의 규정대로
family-상대 재표기 후, "인증서는 신호 클래스에 상대적"을 본문 발견으로 승격한다.

### 16.9 실행 전 관문 (순서 규범적)

1. **서버 머신에서 `--synth` 배터리 PASS** — 14/14 판정이 커밋본과 일치. §15a의 잔여 항목이자
   §4의 게이트. 실패 시 실데이터 실행 금지. (로컬 회귀 실행은 이 게이트를 **대체하지 않는다**.)
2. **구현 대조**: `river_agrawal_abrupt` × 4 조합 (수 분). 4/4 회복이 아니면 그 조합의 구현을
   의심하고, 해당 조합의 실데이터 실행을 보류한다.
3. `[D1]` 스윕 (탐색 0–9) → 4. 판정 이동 셀만 확증 100–109.

### 16.10 실행 명령 (tmux; `nohup` 금지)

```
# 관문 1
python scripts/run_deployment_decay.py --synth 2>&1 | tee logs/synth_server_$(date -u +%Y%m%dT%H%M%S).log
cp results/phase1/deployment_decay/synth_summary.json \
   results/phase1/deployment_decay/synth_battery_PASS_server_$(git rev-parse --short HEAD).json

# 관문 2 (조합별, 수 분)
for C in "topvar hi" "lowvar lo" "interaction lo" "subpop lo"; do set -- $C
  python scripts/run_deployment_decay.py --river agrawal_abrupt --n-seeds 10 \
    --inj-family $1 --inj-cols $2 2>&1 | tee logs/ctrl_$1_$2.log
done

# [D1] — 조합당 별도 tmux 창, 완료 즉시 산출물을 조합별 폴더로 복사(스탬프 초 충돌 회피)
#   --tabred cooking_time delivery_eta ecom_offers homecredit_default weather homesite_insurance
#   + --elec2 + --insects,  각각 --n-seeds 10 --inj-family F --inj-cols C
```

`homesite_insurance`는 §14 목록 밖의 **탐색 추가**이며, 결과 보고 시 반드시 그렇게 표기한다.

## 17. [C]·[D] 실행 결과 판독 (2026-08-01 실행, 커밋 0577684·738c024·f373aa2) — §16 규정 적용

아티팩트: `results/phase1/deployment_decay/summary_20260801T*.json` (34개), `logs/night_driver.log`,
`logs/day2_driver.log`, `results/phase1/deployment_decay/synth_battery_PASS_mapenv_dd82e0a.json`,
`smoke_inj_family_*.json`. 환경: **py 3.11.15 / sklearn 1.9.0 / numpy 2.4.6** (지도 환경).

### 17.1 §4 게이트 — map 환경에서 통과 (2026-08-01 04:29Z)

`--synth` 배터리 **GROUND-TRUTH PASS**, 14셀 판정이 커밋본 01ae6ae와 **14/14 일치**.
스모크(`smoke_test_inj_family.py`) 4개 검사 전부 PASS — 파리티(주입 라벨이 v1과 비트 동일,
3 task × 2 기하), 축 독립성(|corr| 0.769 → 0.013), 캐리어 분리(heavy-tail 기하에서 hi 게이트
탈락 AUC 0.566 / lo 통과 0.957), 게이트 상수 불변.

**부수 확인**: 배터리 판정이 이제 **네 환경**(sklearn 1.6.1 / 1.8.0 / 1.9.0 × 로컬·서버)에서
동일하다. §8의 "판정 수준 안정성" 주장을 뒷받침한다.

### 17.2 §16.9 구현 대조 — 4/4 통과

| 조합 | 학습가능 | lscore | 회복 | rule-B |
|---|---|---|---|---|
| topvar@hi | ✓ | 0.956 | **+0.2007** | ✓ |
| lowvar@lo | ✓ | 0.926 | **+0.2110** | ✓ |
| interaction@lo | ✓ | 0.930 | **+0.1557** | ✓ |
| subpop@lo | ✓ | 0.959 | **+0.0769** | ✓ |

`river_agrawal_abrupt`(D≈0.50) 기준. `topvar@hi`가 7월 실행의 +0.201을 재현했다.
**4조합 전부 채택** — §16.9의 구현-의심 분기 미발동. `subpop@lo`가 가장 낮은 것은 회전이
절반 모집단에서만 도는 설계상 당연한 감쇠다.

### 17.3 [C] full-span — §14 예측 적중, 헤드라인 재스코프 없음

산업 8셀에서 **DEPLOYMENT-CONCEPT 발화 0건.** §14의 재스코프 트리거 미발생 → 지도의 "0/8"은
train 구간 한정으로 재스코프되지 **않는다**. 확증 시드(100–109)에서 **8/8 판정 일치**,
raw·den 수치가 0.004 이내로 재현.

span 간 이동 3건은 기록한다(불안정이 아니라 구간 차이):
- **sberbank**: train NOISE-DRIFT-CONFOUNDED → full-span UNIDENT (raw +0.006~+0.009로 floor
  미달이라 발화 자체가 없어 노이즈 분기로 라우팅되지 않음). **§5.3의 진단은 train 구간 한정임을
  본문에 명시할 것.**
- **cooking**: D 0.966 → 0.927로 **게이트 아래로 하락** → 인증서가 필요 없는 식별가능 null.
  구간이 길어지면 D가 오를 것이라는 사전 예상과 반대 방향.
- **homecredit / homesite**: 인증서가 *vacuous* → **earned**로 격상(full-span에서 주입이
  학습가능해짐).

**주목**: `homecredit_default_fullspan`의 denoised가 **+0.0188 / +0.0177**(floor의 94% / 89%)로
탐색·확증 양쪽에서 살아남았다. 패널 전체에서 가장 큰 양수 방향 수치이며, 발화하지는 않았으나
(mean이 floor 미달) 인증서가 earned이므로 "못 봤다"가 아니라 "봤는데 floor 아래"다.

### 17.4 [D1] 주입 스윕 — 대조 셀 유지, 분모 복구 1건

**대조 셀 (§16.8이 50% 상실을 예측한 곳):**

| 셀 | topvar@hi | lowvar@lo | interaction@lo | subpop@lo | §16.4 판정 |
|---|---|---|---|---|---|
| cooking_time | +0.546 | +0.467 | +0.357 | +0.131 | **4/4 회복 — 무조건 유지** |
| delivery_eta | +0.332 | 학습불능 | +0.274 | 학습불능 | **학습가능 2/2 회복 — 유지** |

두 셀 모두 `injection_recovered_strict = true` → **§15b의 primary-rule-only caveat 해소.**
§5.2/Table 4의 caveat를 strict-확정 라벨로 교체한다.

**공허 4셀:**

| 셀 | 학습가능 | 결과 |
|---|---|---|
| ecom_offers | 0/4 | **VACUOUS 유지** |
| homecredit_default | 0/4 (0.633 / 0.494 / 0.512 / 0.498) | **VACUOUS 유지** (train 구간) |
| **weather** | 3/4 | **§16.5 분모 복구 — 아래** |
| homesite_insurance | 2/4 (0.691, 0.803), 둘 다 미회복 | **EARNED-BLIND, 안정화** |

**★ weather = §16.5 분모 복구 분기 (확증 통과).**

| 조합 | 탐색 0–9 | 확증 100–109 |
|---|---|---|
| lowvar@lo | learn 0.674 · **+0.183** · ruleB ✓ | learn 0.664 · **+0.193** · ruleB ✓ |
| interaction@lo | learn 0.437 · **+0.083** · ruleB ✓ | learn 0.498 · **+0.078** · ruleB ✓ |
| subpop@lo | learn 0.244 (**boundary**) · 미회복 | learn ✗ · 미회복 |
| topvar@hi | 학습불능 (R² −0.022) | — |

학습가능·회복 조합 2개가 탐색·확증에서 일치하고 둘 다 rule-B까지 회복 → §16.5 조건 충족.
**vacuous → verified no-concept (family-상대)** 승격. `subpop@lo`의 학습가능↔공허 뒤집힘은
§16.7(b)가 사전에 배제해 둔 boundary 케이스(회귀 게이트 0.20, lscore 0.244)이며, 판정에 영향을
주지 않았다 — **사전등록 조항이 실제로 작동했다.**

**homesite = §9가 봉인한 불안정의 해소.** 원인은 `topvar@hi`의 lscore가 컷 0.65를 걸친 것
(0.516↔0.694). `lo` 캐리어에서는 0.687~0.804로 컷에서 충분히 떨어져 네 실행 전부 일관되며,
회복은 −0.0005~−0.0040으로 일관 음수 → **earned blindness 확정.**

**elec2**: 4조합 전부 학습가능, 전부 미회복 → earned blindness가 네 신호 클래스에서 확인.
**insects(양성대조)**: 4/4 회복 (+0.162 / +0.265 / +0.229 / +0.047) — §14의 은폐-금지 조항
미발동.

### 17.5 full-span 패밀리 스윕 — 인증서의 패밀리-상대성이 *실증*됨

| 셀 | topvar@hi | lowvar@lo | interaction@lo | subpop@lo | 판정 |
|---|---|---|---|---|---|
| homecredit_fs | 학습가능·미회복 | 불능 0.495 | 불능 0.510 | 불능 0.493 | earned blind **(1/4 학습가능)** |
| **weather_fs** | 불능 | **+0.128** ✓ | **+0.046** ✓ | **학습가능 0.560 · −0.050 미회복** | **부분 인증 (2/3)** |
| homesite_fs | 학습가능·미회복 | 0.933 · −0.007 | 불능 | 0.818 · −0.004 | earned blind (3/3 미회복) |

**★ `weather_fs`의 `subpop@lo`가 결정적이다.** train 구간에서는 lscore 0.244로 boundary라
배제됐으나, full-span에서는 **0.560으로 게이트(0.20)에서 충분히 떨어진 채 −0.050으로 명확히
미회복**이다. 배제할 구실이 없다. 따라서:

> 같은 데이터셋·같은 강도(2.5 rad)에서, 계기는 **저분산 회전(+0.128)과 상호작용 규칙(+0.046)에는
> 검정력이 있고, 부분모집단 국소 규칙 변화에는 없다.**

§7 한계 6이 "미인증"으로만 적어둔 것이 **측정된 사실**이 되었다. §3.2를 "인증서는 패밀리-상대적"
에서 "**어느 패밀리에 검정력이 있고 어느 패밀리에 없는지 측정했다**"로 승격할 것.

### 17.6 패널 완성 — sberbank의 strict 그림자가 검정력 인증서를 얻음

| 조합 | 학습가능 | lscore | 회복 | verdict | verdict_strict |
|---|---|---|---|---|---|
| **topvar@hi** | **✓** | 0.933 | **+0.0864** (ruleB ✓) | NOISE-DRIFT-CONFOUNDED | **INJECTION-RECOVERED** |
| lowvar@lo | ✗ | −0.234 | (+0.086) | 〃 | UNIDENT-EXPL |
| interaction@lo | ✗ | −0.057 | (+0.104) | 〃 | 〃 |
| subpop@lo | ✗ | −0.234 | (−0.095) | 〃 | 〃 |

§16.3 공허 규율 적용: `lo` 3조합은 학습불능이므로 **회복값이 크더라도 어느 방향으로도 쓰지
않는다**(괄호 표기). 유효한 것은 `topvar@hi` 하나이고, 그것이 학습가능·회복·rule-B 전부 충족.

결과: 주 판정(NOISE-DRIFT-CONFOUNDED, 기제=라벨 노이즈 감쇠)과 strict 그림자
(INJECTION-RECOVERED, 검증된 무-concept)가 **서로를 보강한다.** §5.3의 진단에 지금까지 없던
것이 붙는다 — *denoised가 음수인 것은 검정력 부족이 아니다. 같은 기하가 심어진 규칙을 +0.086으로
회복한다.* §15b가 고친 strict-그림자 배선이 실제 값을 만들어냈다.

### 17.7 갱신된 집계문

**§5 고정 집계문 "0/8"은 변경되지 않는다** (사전등록된 판독이며 결과를 본 뒤 바꾸지 않는다).
아래는 그 옆에 함께 보고하는 인증서 회계이며, 분해 자체는 §9에 이미 등록되어 있다.

| | 7월 판독 | 2026-08-01 판독 |
|---|---|---|
| 감사 | 8 | 8 |
| **판독 가능(정보량)** | 4 | **5** — sberbank(진단+검정력인증) · cooking(4/4) · delivery(2/2) · maps(식별가능) · **weather(2/3)** |
| 인증-실명(blind) | 0 | **1** — homesite |
| 인증서 거부 | 4 | **2** — ecom · homecredit |
| **그중 mean-rule drift** | **0** | **0** |

**→ 0/8 감사 · 0/5 판독 가능.** 인증서 없는 셀이 4개에서 2개로 줄었다.

### 17.8 실행 순서 이탈 기록 (§16.9와 다르게 실행한 부분)

§16.9는 river 구현 대조 뒤에 **사람이 멈춰서 판단**하도록 규정했다. 실제 실행은 밤샘 배치라
4조합을 전부 계산한 뒤 아침에 판독했다. **계산한 것과 채택한 것을 구분하면 규율은 보존된다**
(§16.3의 vacuity 규율이 무엇을 채택할지를 이미 규정한다). 결과적으로 4/4가 통과해 배제된
조합은 없었으나, **실행 순서가 사전등록과 달랐다는 사실을 여기 남긴다.**

### 17.9 환경 사고 기록 (§15a와 같은 종류의 결함, 재발)

첫 배터리 실행은 **잘못된 인터프리터**에서 돌았다. 셸 프롬프트가 `(explaintab311)`이었으나
`which python`이 `/home/tonyhuh/credit-dml-validation/.venv/bin/python`(py 3.12.9 / sklearn
1.6.1)을 가리켰다 — 앞서 activate한 다른 프로젝트의 venv가 PATH 앞자리에 남아 있었다.
meta의 ENV 필드를 확인하지 않았다면 그 PASS로 실데이터를 돌렸을 것이다.

**§15a가 기록한 결함(배터리와 실데이터가 서로 다른 환경)이 다른 형태로 재발했고, 이번에는
프롬프트 라벨이 거짓이었다.** 조치: 이후 모든 실행에 절대경로
`$HOME/miniconda3/envs/explaintab311/bin/python`을 사용하고, 배치 스크립트 첫 줄에
`assert (python, sklearn) == ('3.11.15','1.9.0')` 환경 게이트를 넣었다(`run_night.sh`,
`run_day2.sh`). 잘못된 환경의 PASS는 폐기하지 않고 **버전 견고성 증거**로 보존한다
(sklearn 1.6.1에서도 14/14 동일).

## 18. full-span 패밀리 스윕 확증 (2026-08-02 실행, 커밋 1fa4b6f) — §17.5 봉인

§17.5는 탐색 시드(0–9)만으로 읽혔다. §14 규율("판정이 기존 지도와 달라지는 셀은 확증
100–109를 통과해야 논문에 들어간다")에 따라 세 `lo` 조합을 확증 시드로 재실행했다.
아티팩트: `logs/a3_fsconf_{lowvar,interaction,subpop}_lo.log`,
`results/phase1/deployment_decay/summary_20260802T*.json`.

| 조합 | `weather_fullspan` 탐색 | 확증 100–109 | |
|---|---|---|---|
| lowvar@lo | INJECTION-RECOVERED (+0.128) | **INJECTION-RECOVERED** | ✅ |
| interaction@lo | INJECTION-RECOVERED (+0.046) | **INJECTION-RECOVERED** | ✅ |
| subpop@lo | UNIDENT-INERT (학습가능 0.560 · **−0.050 미회복**) | **UNIDENT-INERT** | ✅ |

**3/3 일치.** 두 가지가 동시에 봉인된다:

1. **`weather_fullspan`의 verified no-concept 승격이 확증 안정**이다.
2. **`subpop@lo`의 미회복도 확증 안정**이다 — 즉 §17.5가 보고한 **"계기는 저분산 회전과
   상호작용 규칙에는 검정력이 있고 부분모집단 국소 규칙 변화에는 없다"가 새 시드에서 재현된다.**
   인증서의 패밀리-상대성은 이제 탐색 1회의 관측이 아니라 **확증된 측정**이다.

부수 확인(전부 탐색과 일치): `homecredit_default_fullspan` den **+0.018**[+0.015,+0.021]로
안정, 세 `lo` 조합 모두 `injection-vacuous`(그 셀의 earned 인증서는 기준 패밀리에서만 나온다);
`homesite_insurance_fullspan`은 lowvar·subpop에서 `unident-earned`, interaction에서
`injection-vacuous` — 탐색과 동일한 패턴.

---

## 19. day-4 큐 실행 결과 판독 (2026-08-02 실행, 커밋 a8f1549) — `run_day4.sh` 헤더 사전 예측 적용

> §6 규칙에 따라 **신규 절로만** 추가한다. §1–§18은 손대지 않았고 임계값·캐스케이드·시드
> 프로토콜은 변경하지 않았다. E1의 ACS 부분은 `PREREG_ACS_EXTENSION_2026-07-31.md` §12.

### 19.0 게이트 (PREREG §4 재게이트) — 통과, 비트 동일

신규 opt-in 플래그 2종(`--metric`, `--mi-k`)이 들어갔으므로 배터리 재게이트가 요구됐다.
`smoke_test_inj_family.py` SMOKE PASS → `--synth` **GROUND-TRUTH PASS 14/14**. 추가로
map-env 참조(`audit_repair_2026-07-18/synth_battery_v3_PASS_server_mapenv_sklearn190_py31115.json`,
py3.11.15 / sklearn 1.9.0)와 전 셀·전 필드(`staleness_harm`·`denoised_staleness`·`noise_ratio`
·`injected_staleness`·`injection_learn_score`·`D_strip`) **부동소수 비트 동일**, 판정 14/14 일치.
→ 두 플래그는 기본 경로를 건드리지 않았다. 되돌림 불필요.

### 19.1 E1 (산업 3셀, proper score) — 지표를 바꿔도 vacuity는 풀리지 않는다

판독 규칙(사전 커밋): 판정 라벨 폐기, 팔 크기만. floor·게이트·envelope가 전부 AUC 단위로
보정돼 있어 proper score에서 라벨은 구성상 무의미하다.

| 셀 | brier raw / den | logloss raw / den | 주입 learn (brier / logloss) |
|---|---|---|---|
| ecom_offers | +0.002 / +0.009 | −0.027 / −0.008 | −0.252 / −0.698 |
| homecredit_default | −0.000 / +0.000 | −0.089 / +0.005 | −0.133 / −0.436 |
| homesite_insurance | −0.005 / −0.003 | −0.038 / −0.018 | −0.001 / −0.022 |

**판독**: 인증서 없는 두 셀(ecom·homecredit)은 proper score에서도 인증서를 얻지 못한다 —
주입이 네 지표 전부에서 학습불능(learn 전부 음수, 게이트 R² 0.20 근처도 못 간다). 지표 교체는
**vacuity를 풀지 못한다**. ecom brier den +0.009는 floor(0.02) 아래이고 그 floor는 여기서 무효라
판정이 아니다. homesite logloss 주입 −0.383(별표 없음: 학습불능 주입의 회복값은 해석 불가)은
공허 규율의 재확인이다.

### 19.2 E3 (mi-k 내부 D 사다리) — 예측 1건 부호 반대, 1건 측정 불가

| k | delivery_eta D / den | homecredit_default D / den |
|---|---|---|
| 5 | 0.521 / −0.002 | 0.721 / **+0.013 [+0.010, +0.016]** |
| 10 | 0.540 / −0.004 | 0.796 / +0.006 |
| 20 | 0.555 / −0.005 | 0.833 / +0.006 |
| 50 | 0.801 / −0.013 | 0.912 / −0.010 |

- **예측 ①(D가 k와 함께 하락, 80%) = 부분 반증.** 단조성은 맞으나 **부호가 반대**로 D는 k와
  함께 *상승*한다(피처가 많을수록 윈도우가 더 잘 갈린다). 사후 재해석하지 않고 그대로 기록한다.
- **예측 ②(D 하락에 따라 회복 상승, 60%) = 측정 불가.** 표현을 좁히자 두 셀 모두 D\*=0.96
  **아래로 내려가 주입 컨트롤이 아예 라우팅되지 않았다**(전 셀 `injected_staleness` = null).
  이 손잡이로는 셀 내부 회복 사다리를 만들 수 없다. 인접 측정은 E4(§19.4).
- **부수 관측(사전등록 아님, 관측으로만 기록)**: 같은 셀에서 표현만 바꿔도 denoised가 이동하고,
  homecredit은 k=5에서 D=0.721 = **식별가능 영역**에 들어오면서 양의 sub-floor 신호
  (+0.013)를 낸다 — 전 표현(D=1.000)에서는 인증 불가였던 셀이다. 표현이 식별가능성과 판독
  부호를 동시에 정한다.
- **스코프**: `--mi-k`는 `--metric`과 같은 이유로 **진단 전용**이다(표현을 바꾸면 estimand가
  바뀐다). 지도 판정이 아니다. 이는 사전등록된 규정이 아니라 **지금 명시하는 판독 스코프**이며,
  그렇게 표시해 둔다.

### 19.3 E2 (유형-귀속 프레임 head-to-head) — 예측 2건 모두 성립

| 셀 | 진실 | cov_AUC | ESS% | Y\|X gap |
|---|---|---|---|---|
| synth_concept | 규칙 이동 | 0.500 | 71.3 | **+0.4345** |
| synth_reg_concept | 규칙 이동 | 0.500 | 71.3 | **+0.8207** |
| synth_reg_early_noisy | 규칙 고정(노이즈 감쇠) | 0.500 | 71.3 | +0.0576 |
| synth_reg_xdep_noise | 규칙 고정(x-의존 노이즈) | 0.500 | 71.3 | +0.0615 |
| synth_reg_stable | 규칙 고정(진짜 null) | 0.500 | 71.3 | −0.0208 |

- **예측 ①(규칙 셀이 노이즈 셀을 5배 초과, 65%) = 성립.** 14.2× / 7.5×.
- **예측 ②(노이즈 셀도 양의 Y|X gap → 부호만으론 안 갈림, 75%) = 성립.** +0.058·+0.062가
  진짜 null(−0.021)과 부호로 갈리고 규칙 셀과는 안 갈린다.
- cov_AUC 0.500·ESS 71.3%로 **가중이 건강한 조건**에서 나온 결과다 — 프레임이 불리한 판에서
  진 게 아니다.
- **결론**: "필드 도구가 이걸 오독한다"는 **반증**(크기로는 갈린다). 남는 주장은 좁다 —
  **부호만으로는 안 갈리고, 임계값 판독은 둘을 같이 Y|X로 부르는데 함의된 조치는 한쪽에서만
  듣는다.** 초록 마지막 문장은 이 좁은 형태로만 쓴다.

### 19.4 E4 (앵커 × 주입 패밀리 스윕) — 예측 성립, 그리고 §17·§18 패밀리 진술에 한정어가 붙는다

12/12 전부 회복·학습가능(learn 0.916–0.968):

| 패밀리 | stagger_abrupt | sine_abrupt | hyperplane_incr |
|---|---|---|---|
| topvar@hi | +0.206 | +0.230 | +0.207 |
| lowvar@lo | +0.192 | +0.238 | +0.215 |
| interaction@lo | +0.152 | +0.195 | +0.135 |
| **subpop@lo** | **+0.119** | **+0.079** | **+0.076** |

**★ subpop이 여기서는 회복한다** (앵커 D ≈ 0.49). `weather_fullspan`(고-D)에서 학습가능한데도
미회복(−0.050)이던 것과 대비된다. → **subpop 맹점은 패밀리 단독의 속성이 아니라 패밀리 × 기하**
이다. §17·§18의 패밀리-상대성 진술은 "고-분리도 산업 기하에서"라는 한정어를 달고 읽어야 한다.
같은 방향으로, subpop은 네 패밀리 중 회복 크기가 일관되게 최소(0.076–0.119 vs 0.135–0.238)여서
**D가 올라갈 때 가장 먼저 떨어지는 패밀리**라는 해석과 정합적이다.
