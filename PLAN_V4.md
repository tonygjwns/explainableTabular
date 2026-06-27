# PLAN_V4 — 피벗 (2026-06-27): concept 추격 종료 → positivity-failure regime 후보

> V1(시간-인덱싱 메모리/검색 방법) → V2(측정: covariate 지배) → V3(정직-범위화 측정 + 6번 dissolution) →
> **V4(피벗 후보: covariate near-disjoint support = positivity-failure regime를 *positive*로)**.
> 규율: commit 전 싼 pre-test. 5번+1 dissolution의 원인 = test 전 commit.

## 왜 피벗
6번의 깨끗한 양성이 모두 엄밀 검정서 녹음: ①시간-인덱싱 검색(단조 음성) ②concept 착취(거의 부재/측정불가)
③cov_AUC가 TabReD 퍼즐 예측 못 함 ④공간/시간 대조 ⑤단순 생성법칙 ⑥재발 niche(2차 적대검증서 기각 — 헤드라인
+0.26은 항등식 분해상 (구조 +0.07)+(recency −0.18), 배포될 full method는 실데이터서 음수, STAGGER 단일 의존, HARKing).
RESULTS §24-26.
**살아남은 단 하나의 robust 사실**: 표 시간 shift는 *규칙변화*가 아니라 *입력이 너무 변해 early/late support가 거의
disjoint*(cov-AUC≈1, overlap→0, 배포표현 5/8 un-checkable). 이걸 장애물이 아니라 **발견(positive)**으로.

## 후보 척추 (fresh-LLM 4개 독립 수렴 + 사용자 결정)
**"현실 표 시간 task의 한 class는 positivity-failure regime에 살고, 거기서 표준 shift-보정(IW/DRO/conformal/DA)이
조용히 깨진다 — 우리는 그걸 탐지하고 환각 대신 기권하는 검증된 진단을 준다."** concept 양성도, baseline 이기는
method도 불필요. (제안4 #1 + 제안2 #1: 외삽이 TabReD 퍼즐 설명 + harmful/benign disjointness 구분.)

## ★ 사용자 핵심 반론 = 1급 검증 항목 (전처리 artifact 아니냐?)
이미 절반 확인됨: §6서 de-time-leak/sparse-MI가 4/5 overlap 복원 → disjoint의 상당부분은 *배포 표현(가공 피처)* 탓.
단 (a) 우리 버그 아닌 TabReD 공식 전처리, (b) 진짜 질문 = 그 disjoint가 **유해(예측 방향도 disjoint→보정 깨짐)**냐
**무해(nuisance 방향뿐→예측 멀쩡)**냐. **이게 pivot의 사활.**

## V4.0 pre-test (≤1.5일, model-light) — `run_positivity_regime.py`
TabReD 8개(+elec2/insects)에 3블록:
1. **버그-클린**: n_clocklike(|corr(feat,t)|>0.95 = ID/timestamp 누수), cov_auc_raw vs cov_auc_no_timeproxy(클락이면
   ~0.5로 붕괴 = artifact 확정).
2. **benign vs harmful 판별**: cov_auc_labelrel(top-MI 피처)·covPC(예측좌표 ŷ) — disjoint가 예측 방향에 사나.
3. **★다운스트림 직접 harm**: split-conformal late coverage(nominal 0.90 미달=깨짐) + perf_drop(early-holdout vs late).
   **harmful := conformal under>5pt OR perf_drop 큼**(서류상 disjoint가 *실제로 예측을 해치나*).
- **사전등록 KILL**: disjoint 데이터셋이 harmful 아니면(n_disjoint_AND_harmful≈0 ∧ Spearman(cov_auc,perf_drop) ns)
  → 261-dim disjointness는 **benign 전처리 artifact** → positivity-척추 폐기, 정직한 측정/datasheet 논문(제안1/3)으로 후퇴.
  harmful이면 → regime 진짜 → 척추 viable.
- 로컬 synth 검증 통과: benign(covRAW 1.0/conf 0.89/drop 0.00) vs harmful(covRAW 1.0/conf 0.40/drop 0.49) 정확 구분.
- 서버: `python scripts/run_positivity_regime.py --tabred sberbank_housing homecredit_default ecom_offers
  homesite_insurance weather cooking_time maps_routing delivery_eta --elec2 --insects` → `positivity_regime/summary.json`.

## ★ V4.0 pre-test 결과 (2026-06-27, 서버) — 척추 *기각*(사전등록 kill 충족)
**Spearman(cov_auc_raw, perf_drop) = −0.106 (p=0.77)** — disjointness가 harm 예측 못 함. disjoint(cov>0.9) 7개 중
harmful 2개뿐(elec2=자기상관교락·非TabReD / ecom=drop 0.087인데 conformal 0.987 멀쩡). **간판 disjoint가 benign**:
homecredit(cov1.0/drop0.009/conf0.855), homesite(cov1.0/drop0.008/conf0.904), delivery(cov0.997/drop0.065).
covLR(예측방향 overlap)이 homecredit0.60/weather0.61/delivery0.54 → **disjoint는 nuisance 방향뿐, 예측방향은 겹침.**
→ **261-dim 안 겹침 = benign 전처리 artifact**(사용자 직감 적중). "harmful positivity-failure regime" 척추 사망.
규율 성공: 7번째 dissolution을 commit 전에 1.5일 사전등록 실험으로 차단. RESULTS §27.

## V4-B — 도메인 피벗: 도구킷을 *적대적 도메인*으로 (concept이 실제 사는 곳, 2026-06-27)
경로 A(정직한 측정/datasheet)는 *바닥*으로 확보(천장 낮음). 상방 = 도구킷(도메인 불변, ACS 입증)을 공격자가 규칙을
*실제로 바꾸는* 도메인(fraud/malware/spam)에. 이론상 concept이 식별되는 유일한 곳.
- `run_adversarial_probe.py` (generic CSV loader): 지난 7번 함정을 다 내장한 단일 verdict —
  **REAL-HARMFUL-CONCEPT := 측정가능(ess≥5%) ∧ gap−placebo>0.041 ∧ 자기상관 아님(lagged-label 생존) ∧ 유해(perf_drop/conformal).**
- **사전등록**: 적대적 데이터셋 ≥1개가 REAL-HARMFUL-CONCEPT → 도구킷이 concept 사는 곳 발견 = 양성 방향(방법론 재배치).
  0/N → 적대적 도메인서도 concept 부재 → 더 강한 일반 negative(측정 논문, 도메인-광역).
- 로컬 검증 통과: 회전규칙→REAL-HARMFUL-CONCEPT / covariate-only→unmeasurable.
- 데이터 후보(서버 kaggle API 보유): ① BAF(Bank Account Fraud, NeurIPS'22, month 시간축) ② IEEE-CIS Fraud(TransactionDT)
  ③ EMBER(malware, appeared 날짜 — concept drift 유명). 서버: `kaggle datasets download ...` 후 `--csv --target --time`.

## (역사) 후속 (pre-test 통과 시 — *해당 없음*, 척추 기각됨)
- E1 MDE/검정력 곡선(제안1: 기권이 반증가능한지), E3 표현 dose-response(제안1/제안4#5), overlap-gated conformal
  정리+경험(제안4#3 methods 각도), positivity stress suite(제안3#4). 통과 못 하면 측정/datasheet(제안1·3)로.
- 상세 발상 = `Desktop/ExplainableTab_ReviewPackage/제안1~4` + `IDEATION_BRIEF.md`.
