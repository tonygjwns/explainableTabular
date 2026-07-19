# REVISION_NOTES — 필수 보수 실행 기록 (2026-07-18, 독립 판정 후속)

독립 판정(세 평가 메모 교차검증 + artifacts/ 전수 대조)이 확정한 보수 목록의 실행 기록.
분류: [완료 = 이 클린룸에서 실행됨] / [원고 수정 = LaTeX 반영 필요, 정확한 대체 문안 포함] /
[서버 큐 = 데이터셋·서버 필요, 이 폴더에서 실행 불가].

---

## R1. 서버-버전 환경에서 합성 배터리 재실행 [완료]

**결함**: 유일한 배터리 PASS(`synth_battery_v3_PASS.json`)는 Python 3.14.3 / sklearn 1.8.0
(로컬)에서 실행됐고, 지도를 만든 실데이터 65개 JSON은 전부 Python 3.11.15 / sklearn 1.9.0
(서버)이다. 계기 유효성 증명과 측정이 서로 다른 환경에 있었다 (평가 B 논거 1).

**실행**: Python 3.11 venv에 서버와 동일한 sklearn 1.9.0 / numpy 2.4.6을 설치하고
무수정 코드로 `--synth` 재실행.

**결과**: **PASS — 14/14 셀 판정 일치, 불변식 전부 성립.**
아티팩트: `artifacts/repair_20260718/synth_battery_v3_PASS_sklearn190_py311_runA_unmodified-code.json`
(meta: Python 3.11.5 / sklearn 1.9.0 / numpy 2.4.6, UTC 2026-07-18T15:43).
수치는 3~4째 자리에서만 차이(버전 간 비트 동일은 원래 비주장). 하중을 지는 오탐 채널이
새 버전에서 그대로 차단됨: reg_early_noisy raw +0.0219 → den +0.0045 (gate 3.53),
reg_xdep_noise raw +0.0255 → den +0.0063 (gate 3.72), 공존 셀 reg_concept_earlynoisy는
gate 3.70에서도 den +0.319로 발화 유지 (gate가 veto하지 않음). 평가 B의 뒤집힘 조건
"서버 환경 배터리 FAIL"은 발생하지 않음.

**잔여 갭**: 이 재실행은 서버 머신 자체가 아니라 버전-일치 로컬 venv이다 (Python 패치버전
3.11.5 vs 3.11.15). 하중을 지는 변수(sklearn 1.9.0)는 정확히 일치. 서버에서의 1회 재실행은
[서버 큐]에 유지.

## R2. strict 그림자 캐스케이드의 주입 단계 구현 [완료]

**결함**: [run_deployment_decay.py](code/run_deployment_decay.py) 구 741-745행 — 주입 승격이
`verdict`에만 적용되고 `verdict_strict`는 728행에서 확정된 채 절대 승격되지 않았다. 따라서
모든 INJECTION-RECOVERED 셀(cooking, delivery, kNN/elec2)이 기계적으로 rule-sensitive로
분류되어야 했으나, 논문 Figure 1은 strict 그림자를 "always attached"로 광고하고 §5.2는 두
셀을 "the strongest cells in the map"으로 무플래그 헤드라인화했다 (평가 B 논거 2; 사실관계
원본 확인됨).

**수정 내용** (`code/run_deployment_decay.py`):
1. `_injection_recovers`가 주입 staleness의 CI도 반환.
2. 주입 컨트롤 실행 조건을 `verdict` 또는 `verdict_strict`가 UNIDENTIFIABLE*/CONCEPT인
   경우로 확장 (예: sberbank K=10처럼 rule A는 NOISE-DRIFT-CONFOUNDED, rule B만 UNIDENT로
   라우팅되는 셀에서도 그림자가 완결되도록).
3. strict 그림자가 자기 규칙(B: 주입 CI 하한 > floor)으로 회복을 판정해 승격:
   `verdict_strict → INJECTION-RECOVERED`.
4. 출력 blob에 `injected_staleness_ci`, `injection_recovered_strict` 필드 추가.

**수정 후 배터리 재실행 결과**: **PASS — 커밋본과 14/14 판정 일치, run A와 raw 수치 비트
동일** (수정이 RNG 스트림·기존 계산 경로를 건드리지 않음의 증명). 전 셀에서 verdict·
verdict_strict 불변, 신규 필드 정상 방출. vacuity 규율 보존 확인: covariate_mc는 주입
CI하한 +0.0207 > floor로 strict 회복 기준을 충족하지만 학습불능(vacuous)이므로 승격되지
않음. 배터리에는 INJECTION-RECOVERED 셀이 없으므로 승격 경로 자체는 [D] 실행에서 실증됨.
아티팩트: `artifacts/repair_20260718/synth_battery_v3_PASS_sklearn190_py311_runB_strict-shadow-fix.json`
(UTC 2026-07-18T17:04).

**주의 — 기존 아티팩트의 소급 재라벨은 불가**: 기존 실데이터 JSON에는 주입 per-seed 값이
방출되지 않아(평균만 존재) rule-B 회복 CI를 사후 계산할 수 없다. cooking(+0.546/+0.555),
delivery(+0.332/+0.318)는 floor의 16~27×라 rule-B 회복이 사실상 확실하지만, 확정 라벨은
[서버 큐]의 재실행([D] 주입-패밀리 sweep이 어차피 주입을 재실행하므로 그때 함께 방출)으로만
얻는다. 그 전까지 원고는 R4의 공개 문안으로 처리한다.

## R3. PREREG 신규 섹션 §15 [완료]

PREREG_DEPLOYMENT_V2.md 자체 규칙(기존 텍스트 수정 금지, 새 섹션 추가만 허용)에 따라
§15를 추가: strict-그림자 공백의 발견 경위, R1·R2의 실행 기록, cooking/delivery 라벨의
잠정 지위, MLP/elec2가 (기계적 공백이 아닌) 실질적 rule-sensitive임의 명시.

## R4. 원고 수정 문안 [원고 수정 — LaTeX 반영 필요]

이 클린룸에는 main.pdf만 있고 LaTeX 소스가 없어 문안만 확정한다. 페이지·절 번호는
main.pdf (2026-07-18 11:19 UTC 빌드) 기준.

**(a) Table 4 (p.10) — cooking_time·delivery_eta 행에 각주 추가**:

> "INJECTION-RECOVERED is a primary-rule label: the strict shadow cascade in the committed
> instrument stopped before the injection stage, so it could not confirm recovery under the
> strict rule (a code gap we report and fix in §15 of the pre-registration; the underlying
> quantities are rule-robust—the real staleness is null under both rules and the recovery
> margin is 16–27× the floor—but the strict-confirmed certificate awaits the injection-family
> rerun)."

**(b) §5.2 (p.9-10) — "The strongest cells" 문장 직후에 한 문장 삽입**:

> "One caveat attaches to these two certificates: the strict-rule shadow verdict never
> traversed the injection stage in the committed runs (an implementation gap of the shadow
> cascade, not a property of the cells), so their headline label is primary-rule-only until
> the family sweep rerun; the constituent readings themselves are rule-robust."

**(c) §5.4 (p.10) — "replicated across two independent canary classes" 문장 한정**:

현행: "the flip that matters is replicated across two independent canary classes"
수정: "the flip is replicated across two independent canary classes, with one asymmetry
worth stating: the linear flip survives the strict rule (DEPLOYMENT-CONCEPT under both
readings), while the MLP flip is rule-sensitive (denoised +0.025 clears CI>0 but not the
strict CI>floor bar)—consistent with its canary grade."

근거(원본): `summary_20260704T142045` linear/elec2 verdict = strict = DEPLOYMENT-CONCEPT;
`summary_20260705T163440` MLP/elec2 = DEPLOYMENT-CONCEPT / strict UNIDENTIFIABLE-INERT.

**(d) §4.2 (p.9) — gate 범위 문장을 결정등급으로 한정**:

현행: "every real-data gate value in § 5 falls either clearly below 1.5 or in 2.0–2.9—nowhere
near the envelope edge."
수정: "every decision-grade (HGB/RF) real-data gate value in § 5 falls either clearly below
1.5 or in 2.0–2.9—nowhere near the envelope edge (canary probes reach 1.75 on kNN/sberbank
and 3.40 on linear/sberbank, still inside the 4.7 envelope; canary verdicts are not consumed)."

근거(원본): `summary_20260704T161036` kNN/sberbank noise_ratio 1.7495;
`summary_20260704T141745` linear/sberbank 3.3962.

**(e) §8 Reproducibility (p.13) — 배터리 환경 공개 + 교차버전 결과 반영**:

"The synthetic battery is byte-reproducible ... on the same environment" 문장 뒤에 추가:

> "The committed battery PASS artifact was produced on a local environment (Python 3.14,
> scikit-learn 1.8.0) distinct from the server that produced the map (Python 3.11, 1.9.0).
> A version-matched rerun (scikit-learn 1.9.0 / NumPy 2.4.6) reproduces the PASS with all
> 14 verdicts unchanged (artifact committed), closing the cross-environment validity gap;
> byte-identity across sklearn versions is, as stated, not claimed."

(<<RUN_A_RESULT>>가 PASS일 때의 문안. 다를 경우 §15와 함께 재작성.)

## R5. [서버 큐] — 이 폴더에서 실행 불가, 제출 전 필수

1. **서버에서 `--synth` 재실행** (수 분): PASS 아티팩트를 서버 환경 meta로 커밋 — R1의
   잔여 갭 소거.
2. **[D] injection-family sweep** (PREREG §14에 규정 완료): 수정된 코드로 실행하면
   `injection_recovered_strict`가 자동 방출되어 cooking/delivery의 strict-확정 인증서까지
   한 번에 해결. R4(a)/(b)의 각주를 확정 라벨로 교체.
3. **[C] full-span 감사** (PREREG §14): 0/8 헤드라인의 배포-갭 방어.
4. **git 이력 포함 제출** (§8이 이미 약속): 사전등록 시간순서의 독립 검증 제공.

## R6. 보수 불요로 판정된 항목 (기록용)

- nodrift SUBFLOOR 소급 해석(§9): 결함은 실재하나 방향이 보수적(증거 폐기 방향)이고 논문
  §6·Table 8이 공개함. 요구되는 것은 추정기 교체 실험(블록 부트스트랩 — per-seed 값 방출
  완료)이며 이는 강화 항목이지 차단 항목이 아님.
- B.1 δ(N) 스윕: 사후(post-hoc) 명시 표기 확인, 수치 전부 원본 일치 — 수정 불요.
- 교차렌즈 B.2: 수치 전부 원본 일치 — 수정 불요.
